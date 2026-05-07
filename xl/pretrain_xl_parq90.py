"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import inspect
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

from parq.optim import (
    QuantOptimizer,
    ProxPARQ,
    ProxHardQuant,
    ProxSoftQuant,
    ProxBinaryRelax,
)

# -----------------------------------------------------------------------------
class FixedGridQuantizer:
    """
    Fixed-grid quantizer with optional partial masking.

    ratio = 1.0  -> full quantization
    ratio = 0.9  -> only 90% entries are quantized, 10% stay as original values
    """

    def __init__(
        self,
        step: float,
        ratio: float = 1.0,
        fixed_mask: bool = True,
        mask_seed: int = 2026,
    ):
        if step <= 0:
            raise ValueError(f"step must be > 0, got {step}")
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        self.step = float(step)
        self.ratio = float(ratio)
        self.fixed_mask = bool(fixed_mask)

        # cache masks per parameter object
        self._mask_cache = {}

        # use a dedicated generator so DDP ranks can create the same masks
        self._mask_gen = torch.Generator(device="cpu")
        self._mask_gen.manual_seed(mask_seed)

    def get_quant_size(self, b: int) -> int:
        L = (2 ** (b - 1)) - 1
        return 2 * L + 1

    @torch.no_grad()
    def _get_mask(self, p: torch.Tensor):
        if self.ratio >= 1.0:
            return None

        key = id(p)
        if self.fixed_mask and key in self._mask_cache:
            mask_cpu = self._mask_cache[key]
            if mask_cpu.shape == tuple(p.shape):
                return mask_cpu.to(device=p.device)

        # create mask on CPU with dedicated generator, then move to p.device
        mask_cpu = (
            torch.rand(p.shape, generator=self._mask_gen, device="cpu") < self.ratio
        )

        if self.fixed_mask:
            self._mask_cache[key] = mask_cpu

        return mask_cpu.to(device=p.device)

    @torch.no_grad()
    def quantize(self, p: torch.Tensor, b: int, dim: int | None = None):
        L = (2 ** (b - 1)) - 1
        clip = L * self.step

        # full-grid quantized tensor
        q = p.detach().clone()
        q.div_(self.step).round_().mul_(self.step)
        q.clamp_(-clip, clip)

        # partial mask: only quantize ratio of coordinates
        mask = self._get_mask(p)
        if mask is not None:
            q = torch.where(mask, q, p.detach())

        Q = torch.arange(-L, L + 1, device=p.device, dtype=p.dtype) * self.step

        # keep original compatibility branch
        if dim is not None:
            if p.dim() == 2 and (dim == -1 or dim == 1):
                Q = Q.unsqueeze(0).expand(p.size(0), -1).clone()
            else:
                Q = Q.unsqueeze(0)

        return q, Q


def should_quantize_like_dist4(name: str, p: torch.nn.Parameter) -> bool:
    """
    Match your own selector_mode='gpt2_custom' + exclude_substrings exactly.
    """
    if not p.requires_grad:
        return False
    if not name.endswith(".weight"):
        return False

    n = name.lower()

    exclude_substrings = ("bias", "norm", "ln_", "wte", "wpe", "lm_head")
    if any(s in n for s in exclude_substrings):
        return False

    target_layers = ("c_attn.weight", "c_proj.weight", "c_fc.weight")
    return any(t in n for t in target_layers)

def configure_parq_optimizer(
    model,
    weight_decay,
    learning_rate,
    betas,
    device_type,
    max_iters,
    quant_bits=4,
    fixed_step=(0.10 / 7.0),
    quant_ratio=1.0,
    quant_mask_seed=2026,
    quant_proxmap="parq",
    warmup_steps=0,
    quant_period=10,
    anneal_start=200,
    anneal_end=3000,
    steepness=15,
):
    if anneal_end is None:
        anneal_end = max(anneal_start + 1, max_iters - 200)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    quant_params = []
    decay_params = []
    nodecay_params = []

    quant_names = []
    decay_names = []
    nodecay_names = []

    for pn, p in param_dict.items():
        if should_quantize_like_dist4(pn, p):
            quant_params.append(p)
            quant_names.append(pn)
        elif p.dim() >= 2:
            decay_params.append(p)
            decay_names.append(pn)
        else:
            nodecay_params.append(p)
            nodecay_names.append(pn)

    print("========== PARQ parameter split ==========")
    print(f"fixed_step: {fixed_step}")
    print(f"quant_bits: {quant_bits}")
    print(f"quantized tensors: {len(quant_params)}")
    print(f"normal decay tensors: {len(decay_params)}")
    print(f"no_decay tensors: {len(nodecay_params)}")
    print("first few quantized names:")
    for name in quant_names[:20]:
        print("  ", name)
    if len(quant_names) > 20:
        print(f"  ... and {len(quant_names) - 20} more")
    print("==========================================")

    assert len(quant_params) > 0, "No quantized parameters found. Check layer-name matching."

    param_groups = [
        {"params": quant_params, "quant_bits": quant_bits},
        {"params": decay_params},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = {"fused": True} if use_fused else {}

    base_optimizer = torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
        **extra_args,
    )

    # Replace PARQ's dynamic-range uniform quantizer with your exact fixed grid
    quantizer = FixedGridQuantizer(
        step=fixed_step,
        ratio=quant_ratio,
        fixed_mask=True,
        mask_seed=quant_mask_seed,
    )

    if quant_proxmap == "parq":
        prox_map = ProxPARQ(anneal_start, anneal_end, steepness=steepness)
    elif quant_proxmap == "hard":
        prox_map = ProxHardQuant()
    elif quant_proxmap == "soft":
        prox_map = ProxSoftQuant(anneal_start, anneal_end)
    elif quant_proxmap == "binaryrelax":
        prox_map = ProxBinaryRelax(anneal_start, anneal_end)
    else:
        raise ValueError(f"Unsupported quant_proxmap: {quant_proxmap}")

    optimizer = QuantOptimizer(
        base_optimizer=base_optimizer,
        quantizer=quantizer,
        prox_map=prox_map,
        warmup_steps=warmup_steps,
        quant_period=quant_period,
        quant_per_channel=False,   # keep the grid definition simple and identical
        quant_shrink=False,
    )

    return optimizer

# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out-gpt2large-wikitext103'
eval_interval = 100
log_interval = 20
eval_iters = 50
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'wikitext103'
gradient_accumulation_steps = 8 # used to simulate larger batch sizes
batch_size = 4 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256
# model
n_layer = 48
n_head = 25
n_embd = 1600
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 0.0005 # max learning rate
max_iters = 5000 # total number of training iterations
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 100 # how many steps to warm up for
lr_decay_iters = 5000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', dataset)

class EpochDataLoader:
    def __init__(self, split, batch_size, block_size, device, device_type, ddp_rank, ddp_world_size):
        self.split = split
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.device_type = device_type
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size

        if split == 'train':
            self.data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            self.data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

        self.num_chunks = (len(self.data) - block_size) // block_size
        self.epoch = 0
        self.reset()

    def reset(self):
        g = torch.Generator()
        g.manual_seed(1337 + self.epoch)
        all_chunk_indices = torch.randperm(self.num_chunks, generator=g) * self.block_size

        self.indices = all_chunk_indices[self.ddp_rank :: self.ddp_world_size]
        self.current_pos = 0
        self.epoch += 1

    def get_batch(self):
        if self.current_pos + self.batch_size > len(self.indices):
            self.reset()

        ix = self.indices[self.current_pos : self.current_pos + self.batch_size]
        self.current_pos += self.batch_size

        x = torch.stack([torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])

        if self.device_type == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y

current_rank = ddp_rank if ddp else 0
train_loader = EpochDataLoader('train', batch_size, block_size, device, device_type, current_rank, ddp_world_size)
val_loader = EpochDataLoader('val', batch_size, block_size, device, device_type, current_rank, ddp_world_size)

def get_batch(split):
    if split == 'train':
        return train_loader.get_batch()
    else:
        return val_loader.get_batch()

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))


# ---- dist4-style diagnostics only: DO NOT affect training updates ----
qat_n_bits_w = 4
qat_step_w = 0.035 / 7.0
qat_atol = 0.001
quant_ratio = 0.90
quant_mask_seed = 1337
hard_eval_interval = 100
qrate_interval = 100

current_qrate = 0.0
current_sat_rate = 0.0
current_hard_train_loss = float("nan")
current_hard_val_loss = float("nan")


# optimizer
#optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

optimizer = configure_parq_optimizer(
    model=model,
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas=(beta1, beta2),
    device_type=device_type,
    max_iters=max_iters,
    quant_bits=qat_n_bits_w,
    fixed_step=qat_step_w,
    quant_ratio=quant_ratio,
    quant_mask_seed=quant_mask_seed,
    quant_proxmap="parq",
    warmup_steps=0,
    quant_period=10,
    anneal_start=200,
    anneal_end=max(201, max_iters - 200),
    steepness=10,
)

if init_from == 'resume':
    #optimizer.load_state_dict(checkpoint['optimizer'])
    if isinstance(optimizer, QuantOptimizer):
        optimizer.load_state_dict(checkpoint['optimizer'], start_step=iter_num)
    else:
        optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))


    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
