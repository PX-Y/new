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
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

from qat_prox2.config_small import QATConfig
from qat_prox2.controllers2 import DualController
from qat_prox2.param_filter import QuantParamSelector
from qat_prox2.sensitivity import SensitivityEMA
#from qat_prox2.quant_ops import hard_quantize_model_inplace, restore_model_from_backup
#from qat_prox2.quant_ops import hard_quantize_model_inplace, restore_model_from_backup
from qat_prox2.quant_ops_2b import (
    hard_quantize_model_inplace,
    selective_hard_quantize_model_inplace,
    restore_model_from_backup,
)
from qat_prox2.quant_stats_2b import compute_quantization_rate_fast
from qat_prox2.utils4_2b_sens import prepare_theory_matched_quant_update_with_sensitivity, apply_prepared_quant_update

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out-gpt2xl-wikitext103'
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
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 0.0031 # max learning rate
max_iters = 2000 # total number of training iterations
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 200 # how many steps to warm up for
lr_decay_iters = 2000 # should be ~= max_iters per Chinchilla
min_lr = 0 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

# explicit 2-bit / 4-level quantization levels
n_bits_w = 4
#quant_levels_w = (-0.06, 0.0, 0.06)
quant_levels_w = (
    -0.035,
    -0.030,
    -0.025,
    -0.020,
    -0.015,
    -0.010,
    -0.005,
     0.000,
     0.005,
     0.010,
     0.015,
     0.020,
     0.025,
     0.030,
     0.035,
)
qrate_atol = 0.0011

# sensitivity-weighted quantization
sens_enable = True
sens_ema = 0.95
sens_eps = 1e-8
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, tuple))]
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
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
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

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
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

    # 固定同一批 train/val batch，保证 fp/q/fsq 公平比较
    fixed_batches = {}
    for split in ['train', 'val']:
        batches = []
        for _ in range(eval_iters):
            X, Y = get_batch(split)
            batches.append((X.cpu(), Y.cpu()))
        fixed_batches[split] = batches

    def eval_on_batches(split):
        losses = torch.zeros(eval_iters)
        for k, (X_cpu, Y_cpu) in enumerate(fixed_batches[split]):
            X = X_cpu.to(device, non_blocking=True)
            Y = Y_cpu.to(device, non_blocking=True)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        return losses.mean()

    # 1) full-precision loss
    for split in ['train', 'val']:
        out[f'{split}_fp'] = eval_on_batches(split)

    # 2) full hard-quantized loss
    backup_q_eval = hard_quantize_model_inplace(
        model, qat.quant_levels_w,
        selector=q_selector, exclude_substrings=qat.exclude_substrings
    )

    for split in ['train', 'val']:
        out[f'{split}_q'] = eval_on_batches(split)

    restore_model_from_backup(model, backup_q_eval)

    # 3) selective hard-quantized loss / fsq
    backup_fsq_eval = selective_hard_quantize_model_inplace(
        model, qat.quant_levels_w, atol=qat.atol,
        selector=q_selector, exclude_substrings=qat.exclude_substrings
    )

    for split in ['train', 'val']:
        out[f'{split}_fsq'] = eval_on_batches(split)

    restore_model_from_backup(model, backup_fsq_eval)

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


# Configuration parameters: explicit 2-bit / 4-level quantization.
# The actual quantization points are controlled by quant_levels_w, e.g.
# (-0.09, -0.03, 0.03, 0.09). n_bits_w is kept for logging/config clarity.
qat = QATConfig()
qat.n_bits_w = n_bits_w
qat.quant_levels_w = tuple(quant_levels_w)
qat.atol = qrate_atol

q_selector = QuantParamSelector(mode=qat.selector_mode)
dual_ctl = DualController(beta=qat.beta, dual_lr=qat.dual_lr, lambda_init=qat.lambda_init, lambda_max=qat.lambda_max)

if master_process:
    print(f"[quant-levels] n_bits_w={qat.n_bits_w}, quant_levels_w={qat.quant_levels_w}, atol={qat.atol}")

dual_lambda = dual_ctl.lam
current_qrate = 0.0
current_sat_rate = 0.0

# Sensitivity EMA: stores one gradient-squared EMA value per selected tensor.
sens = SensitivityEMA(momentum=sens_ema, eps=sens_eps) if sens_enable else None

# training loop starts
X, Y = get_batch('train') 
t0 = time.time()
local_iter_num = 0 
raw_model = model.module if ddp else model 
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # One-time full hard quantization right after warmup.
    # Since this file uses explicit 2-bit levels, pass qat.quant_levels_w, not n_bits/step.
    if iter_num == warmup_iters:
        if master_process:
            print("Warmup finished, hard quantization once before continuing training...")
        hard_quantize_model_inplace(
            raw_model,
            qat.quant_levels_w,
            selector=q_selector,
            exclude_substrings=qat.exclude_substrings,
        )

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}:")
        print(f"  [fp]  train loss {losses['train_fp']:.4f}, val loss {losses['val_fp']:.4f}")
        print(f"  [ q]  train loss {losses['train_q']:.4f}, val loss {losses['val_q']:.4f}")
        print(f"  [fsq] train loss {losses['train_fsq']:.4f}, val loss {losses['val_fsq']:.4f}")

        track_val_loss = losses['val_q']   # 继续用量化后的 val loss 选 checkpoint
        if track_val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = track_val_loss
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # === Forward/Backward pass (with quantization) ===
    loss_unquant_val = 0.0
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        
        # 1. Run forward pass with full-precision weights
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        # 2. Backward pass on full-precision model
        scaler.scale(loss).backward()

        # 3. Use the same full-precision loss for dual controller
        if micro_step == gradient_accumulation_steps - 1:
            with torch.no_grad():
                with ctx:
                    _, loss_unq = model(X, Y)
                    loss_unquant_val = loss_unq.item()

        # Fetch next batch of data
        X, Y = get_batch('train')

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Update tensor-level sensitivity EMA from current gradients.
    # This must happen after backward and before optimizer.zero_grad().
    if sens is not None:
        sens.update_from_grads(
            raw_model,
            selector=q_selector,
            exclude_substrings=qat.exclude_substrings,
        )

    quant_step_pkg = prepare_theory_matched_quant_update_with_sensitivity(
        model,
        optimizer,
        qat=qat,
        selector=q_selector,
        dual_lambda=dual_lambda,
        sens_ema=sens,
    )

    # Step the optimizer
    scaler.step(optimizer)
    scaler.update()
    
    # Override parameters with our custom calculations
    apply_prepared_quant_update(quant_step_pkg)
    optimizer.zero_grad(set_to_none=True)

    # === Update quantization controller parameters ===
    dual_lambda = dual_ctl.step(loss_unquant_val)
    if iter_num % 50 == 0:
        qrate, sat_rate, _ = compute_quantization_rate_fast(
            model, qat.quant_levels_w, atol=qat.atol,
            selector=q_selector, exclude_substrings=qat.exclude_substrings
        )
        current_qrate = qrate
        current_sat_rate = sat_rate

    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        sens_mean, sens_max = sens.mean_max() if sens is not None else (0.0, 0.0)
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, "
            f"mfu {running_mfu*100:.2f}% | lambda: {dual_lambda:.4f}, "
            f"qrate: {current_qrate*100:.2f}%, sat_rate: {current_sat_rate*100:.2f}%, "
            f"sens_mean: {sens_mean:.3e}, sens_max: {sens_max:.3e}"
        )

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
