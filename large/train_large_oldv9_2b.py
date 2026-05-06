"""
Single-stage soft quantization training for nanoGPT GPT-2 models.

Behavior:
- Uses only the original first-stage soft quantization logic:
  Lagrange + qrate controller + sensitivity-weighted distance loss.
- Removes all second-stage / hard-phase training logic.
- Keeps qrate monitoring during training.
- Saturated/out-of-range entries can be excluded from quantization and left in FP training.
"""

import os
import time
import math
import pickle
import json
import re
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values
# -----------------------------------------------------------------------------
out_dir = 'out-gpt2xl-wikitext103-gammahist-4b-nosatfp'
eval_interval = 100
log_interval = 20
eval_iters = 50
eval_only = False
always_save_checkpoint = True
init_from = 'gpt2_large'  # 'scratch' | 'resume' | 'gpt2' | 'gpt2-medium' | 'gpt2-large' | 'gpt2-xl'

wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2-xl-gammahist-4b'

dataset = 'wikitext103'
gradient_accumulation_steps = 8
batch_size = 2
block_size = 1024

# model defaults are only used for scratch; pretrained init_from overrides them
n_layer = 36
n_head = 20
n_embd = 1280
dropout = 0.0
bias = False

learning_rate = 5e-5
max_iters = 5000
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 100
lr_decay_iters = 1000
min_lr = 5e-6

# quant / qrate monitor
n_bits_w = 4
alpha_w = 0.05 / 8
qrate_every = 100
qrate_atol = 1e-3
# Optional explicit quantization levels. If set by config, these levels override n_bits_w/alpha_w grid.
quant_levels_w = None
# if True, entries whose raw weights are outside the quantization clip range
# are left fully FP and excluded from the quantization penalty/qrate denominator
skip_saturated_fp = True
quant_include_substrings = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
exclude_substrings = ("ln_", "bias", "wte", "wpe")

# lagrange control
use_lagrange = True
beta = 2.6
dual_lr = 3e-4
dual_lambda = 1.4
dual_lambda_max = 1e6
dist_scale = 0.3

# gamma / qrate controller
gamma_start_iter = 300
q_target = 0.95
gamma_lr = 0.2
gamma_max = 5.0
qrate_ema_momentum = 0.9

# sensitivity-weighted dist
sens_enable = True
sens_ema = 0.95
sens_eps = 1e-8
sens_power = 0.5
sens_w_min = 0.2
sens_w_max = 5.0

gradf_every = 100

# lambda PI control
lambda_use_pi = True
lambda_kp = 1.0
lambda_ki = 0.2
lambda_i_clamp = 10.0

# optional layerwise step json; if missing, global alpha_w is used
layer_step_json = ''


# DDP settings
backend = 'nccl'

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, tuple))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(2026 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
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
        g.manual_seed(2026 + self.epoch)
        all_chunk_indices = torch.randperm(self.num_chunks, generator=g) * self.block_size
        self.indices = all_chunk_indices[self.ddp_rank :: self.ddp_world_size]
        self.current_pos = 0
        self.epoch += 1

    def state_dict(self):
        return {
            'epoch': int(self.epoch),
            'current_pos': int(self.current_pos),
            'indices': self.indices.clone(),
        }

    def load_state_dict(self, state):
        self.epoch = int(state['epoch'])
        self.current_pos = int(state['current_pos'])
        self.indices = state['indices'].clone()

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


train_loader = EpochDataLoader('train', batch_size, block_size, device, device_type, ddp_rank, ddp_world_size)
val_loader = EpochDataLoader('val', batch_size, block_size, device, device_type, ddp_rank, ddp_world_size)


def get_batch(split):
    return train_loader.get_batch() if split == 'train' else val_loader.get_batch()


def _loader_for_split(split):
    return train_loader if split == 'train' else val_loader


class _LoaderStateGuard:
    def __init__(self, *loaders):
        self.loaders = loaders
        self.states = None

    def __enter__(self):
        self.states = [loader.state_dict() for loader in self.loaders]
        return self

    def __exit__(self, exc_type, exc, tb):
        for loader, state in zip(self.loaders, self.states):
            loader.load_state_dict(state)
        return False


# init these up here, can override if init_from='resume'
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
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
else:
    raise ValueError(f"Unknown init_from={init_from}")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# -----------------------------------------------------------------------------
# quantization helpers
# -----------------------------------------------------------------------------
alpha_w_layer = None
if layer_step_json and os.path.exists(layer_step_json):
    with open(layer_step_json, 'r', encoding='utf-8') as f:
        _tmp = json.load(f)
    alpha_w_layer = {int(k): float(v['step_sym']) for k, v in _tmp.items()}
    if master_process:
        print(f"[layer-step] loaded {len(alpha_w_layer)} layers from {layer_step_json}")
else:
    if master_process:
        if layer_step_json:
            print(f"[layer-step] NOT found: {layer_step_json}, fallback to global alpha_w={alpha_w}")
        else:
            print(f"[layer-step] not provided, using global alpha_w={alpha_w}")

_layer_pat = re.compile(r"transformer\.h\.(\d+)\.")


def _get_layer_id(name: str):
    m = _layer_pat.search(name)
    return int(m.group(1)) if m else None


def step_for_param(name: str, default_step: float):
    if alpha_w_layer is None:
        return default_step
    lid = _get_layer_id(name)
    if lid is None:
        return default_step
    return float(alpha_w_layer.get(lid, default_step))

def _using_explicit_levels():
    """Return True when config provides explicit quantization levels."""
    return quant_levels_w is not None and len(quant_levels_w) > 0


def _quantize_to_explicit_levels(w, levels):
    """Quantize each entry of w to the nearest value in explicit levels."""
    q_levels = torch.tensor(levels, device=w.device, dtype=w.dtype)
    dist = (w.unsqueeze(-1) - q_levels).abs()
    idx = dist.argmin(dim=-1)
    q = q_levels[idx]
    return q


if master_process:
    if _using_explicit_levels():
        print(f"[quant-levels] using explicit quant_levels_w={quant_levels_w}; alpha_w/n_bits_w grid is ignored for q values")
    else:
        print(f"[quant-grid] using n_bits_w={n_bits_w}, alpha_w={alpha_w}")



def _is_quant_param(name: str, p: torch.nn.Parameter, include_substrings=None, exclude_substrings=("ln_", "bias", "wte", "wpe")):
    if not p.requires_grad:
        return False
    if exclude_substrings is not None and any(s in name for s in exclude_substrings):
        return False
    if include_substrings is not None and (not any(s in name for s in include_substrings)):
        return False
    return True


def _iter_named_quant_params(model, include_substrings=None, exclude_substrings=("ln_", "bias", "wte", "wpe")):
    for name, p in model.named_parameters():
        if _is_quant_param(name, p, include_substrings, exclude_substrings):
            yield name, p


def _quantize_to_grid(w, n_bits, alpha_step):
    if _using_explicit_levels():
        with torch.no_grad():
            return _quantize_to_explicit_levels(w, quant_levels_w)

    L = (2 ** (n_bits - 1)) - 1
    step = alpha_step
    clip = L * step
    with torch.no_grad():
        q = torch.round(w / step) * step
        q = torch.clamp(q, -clip, clip)
    return q


def _quantize_to_grid_with_masks(w, n_bits, alpha_step):
    """Return quantized tensor and masks for active/saturated entries.

    If quant_levels_w is provided in config, use those explicit levels and
    treat all entries as active because there is no fixed clip range.
    Otherwise, use the standard signed symmetric grid controlled by n_bits_w/alpha_w.
    """
    if _using_explicit_levels():
        with torch.no_grad():
            q = _quantize_to_explicit_levels(w, quant_levels_w)
            active_mask = torch.ones_like(w, dtype=torch.bool)
            sat_mask = torch.zeros_like(w, dtype=torch.bool)
            clip = None
        return q, active_mask, sat_mask, clip

    L = (2 ** (n_bits - 1)) - 1
    step = alpha_step
    clip = L * step
    with torch.no_grad():
        q = torch.round(w / step) * step
        q = torch.clamp(q, -clip, clip)
        sat_mask = (w.abs() > clip)
        active_mask = ~sat_mask
    return q, active_mask, sat_mask, clip



@torch.no_grad()
def compute_qrate_layerwise(model, n_bits_w, atol, include_substrings=None, exclude_substrings=("ln_", "bias", "wte", "wpe")):
    total = 0
    active_total = 0
    within = 0
    sat = 0

    for name, p in _iter_named_quant_params(model, include_substrings, exclude_substrings):
        step = step_for_param(name, alpha_w)
        w = p.data
        q, active_mask, sat_mask, clip = _quantize_to_grid_with_masks(w, n_bits_w, step)

        d = (w - q).abs()
        total += d.numel()
        sat += sat_mask.sum().item()

        if skip_saturated_fp:
            active_total += active_mask.sum().item()
            within += ((d <= atol) & active_mask).sum().item()
        else:
            active_total += d.numel()
            within += (d <= atol).sum().item()

    qrate = within / max(active_total, 1)
    sat_rate = sat / max(total, 1)
    return qrate, sat_rate, total, active_total


@torch.no_grad()
def selective_hard_quantize_model_inplace(model, n_bits_w, alpha_w, atol,
                                          include_substrings=None,
                                          exclude_substrings=("ln_", "bias", "wte", "wpe"),
                                          verbose=True):
    backup = {}
    total = 0
    quantized = 0
    active_total = 0
    skipped_sat = 0

    for name, p in _iter_named_quant_params(model, include_substrings, exclude_substrings):
        w = p.data
        step = step_for_param(name, alpha_w)
        q, active_mask, sat_mask, _ = _quantize_to_grid_with_masks(w, n_bits_w, step)
        d = (w - q).abs()

        if skip_saturated_fp:
            mask = (d <= atol) & active_mask
            active_total += int(active_mask.sum().item())
            skipped_sat += int(sat_mask.sum().item())
        else:
            mask = (d <= atol)
            active_total += w.numel()

        backup[name] = w.clone()
        w.copy_(torch.where(mask, q, w))

        total += w.numel()
        quantized += int(mask.sum().item())

    if verbose:
        denom = max(active_total if skip_saturated_fp else total, 1)
        pct = 100.0 * quantized / denom
        if skip_saturated_fp:
            print(
                f"[selective-hard] quantized {pct:.3f}% "
                f"({quantized:,}/{denom:,} active; skipped_sat={skipped_sat:,}, total={total:,}) "
                f"with atol={atol}"
            )
        else:
            print(f"[selective-hard] quantized {pct:.3f}% ({quantized:,}/{total:,}) with atol={atol}")

    return backup


@torch.no_grad()
def restore_model_from_backup(model, backup):
    for name, p in model.named_parameters():
        if name in backup:
            p.data.copy_(backup[name])







def compute_dist_loss(model, n_bits_w, alpha_w,
                      include_substrings=None,
                      exclude_substrings=("ln_", "bias", "wte", "wpe"),
                      sens_state=None,
                      return_debug=False):
    items = []
    skipped_sat_elems = 0
    active_elems = 0
    for name, p in _iter_named_quant_params(model, include_substrings, exclude_substrings):
        step = step_for_param(name, alpha_w)
        q, active_mask, sat_mask, _ = _quantize_to_grid_with_masks(p, n_bits_w, step)
        q = q.detach()

        if skip_saturated_fp:
            active_mask_f = active_mask.to(dtype=p.dtype)
            mse = ((p - q).pow(2) * active_mask_f).sum()
            skipped_sat_elems += int(sat_mask.sum().item())
            active_elems += int(active_mask.sum().item())
            if active_mask.sum().item() == 0:
                continue
        else:
            mse = (p - q).pow(2).mean() * p.numel()
            active_elems += p.numel()

        if (sens_state is not None) and (name in sens_state):
            s = sens_state[name].detach().to(p.device, dtype=torch.float32)
            if (not torch.isfinite(s)) or (s <= 0):
                s = None
        else:
            s = None

        items.append((s, mse))

    if len(items) == 0:
        out = torch.zeros((), device=next(model.parameters()).device)
        return (out, None) if return_debug else out

    K = len(items)
    s_list = [s for (s, _) in items if (s is not None) and torch.isfinite(s) and (s > 0)]

    ws = []
    if len(s_list) == 0:
        for _ in items:
            ws.append(torch.ones((), device=items[0][1].device, dtype=torch.float32))
    else:
        S = torch.stack(s_list).float() + sens_eps
        if (not torch.isfinite(S).all()) or (S <= 0).any():
            ws = [torch.ones((), device=items[0][1].device, dtype=torch.float32) for _ in items]
        else:
            z = torch.log(S)
            mu = z.mean()
            sigma = z.std()
            if (not torch.isfinite(mu)) or (not torch.isfinite(sigma)):
                ws = [torch.ones((), device=items[0][1].device, dtype=torch.float32) for _ in items]
            else:
                sigma = sigma + 1e-6
                for (s, _) in items:
                    if (s is None) or (not torch.isfinite(s)) or (s <= 0):
                        w = torch.ones((), device=mu.device, dtype=torch.float32)
                    else:
                        zz = torch.log(s.to(mu.device) + sens_eps)
                        if not torch.isfinite(zz):
                            w = torch.ones((), device=mu.device, dtype=torch.float32)
                        else:
                            score = (mu - zz) / sigma
                            if not torch.isfinite(score):
                                w = torch.ones((), device=mu.device, dtype=torch.float32)
                            else:
                                w = 1.0 + 4.0 * torch.sigmoid(score)
                    ws.append(w)

    ws = torch.stack([w.detach().float() for w in ws])
    if not torch.isfinite(ws).all():
        ws = torch.ones_like(ws)
    w_mean = ws.mean() + 1e-12

    dist = 0.0
    for i, (_, mse) in enumerate(items):
        rel = ws[i] / w_mean
        if not torch.isfinite(rel):
            rel = torch.ones_like(rel)
        dist = dist + rel * mse

    out = dist / K
    if not torch.isfinite(out):
        out = torch.zeros((), device=next(model.parameters()).device)

    if return_debug:
        rels = (ws / w_mean).detach().cpu()
        dbg = {
            'K': K,
            'w_min': float(ws.min().item()),
            'w_mean': float(ws.mean().item()),
            'w_max': float(ws.max().item()),
            'rel_min': float(rels.min().item()),
            'rel_mean': float(rels.mean().item()),
            'rel_max': float(rels.max().item()),
            'active_elems': int(active_elems),
            'skipped_sat_elems': int(skipped_sat_elems),
        }
        return out, dbg

    return out


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@torch.no_grad()
def estimate_loss_selective_hard():
    with _LoaderStateGuard(train_loader, val_loader):
        backup = selective_hard_quantize_model_inplace(
            raw_model,
            n_bits_w=n_bits_w,
            alpha_w=alpha_w,
            atol=qrate_atol,
            include_substrings=quant_include_substrings,
            exclude_substrings=exclude_substrings,
            verbose=True,
        )
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        restore_model_from_backup(raw_model, backup)
        model.train()
        return out




def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


last_qrate = None
last_sat_rate = None
last_qtotal = None
last_qactive_total = None

gamma = dist_scale
qrate_ema = None
lambda_g_int = 0.0
sens_state = {}

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0


def sens_mean_max(sens_state, device, ddp=False):
    if sens_state is None or len(sens_state) == 0:
        return 0.0, 0.0

    vals_list = []
    for v in sens_state.values():
        vv = v.detach().float().to(device)
        if torch.isfinite(vv).all():
            vals_list.append(vv)

    if len(vals_list) == 0:
        return 0.0, 0.0

    vals = torch.stack(vals_list)
    s_sum = vals.sum()
    s_max = vals.max()
    n = torch.tensor([vals.numel()], device=device, dtype=torch.float32)

    if ddp:
        import torch.distributed as dist
        dist.all_reduce(s_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(s_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)

    mean = (s_sum / (n + 1e-12)).item()
    mx = s_max.item()
    return mean, mx



def grad_norm_l2(grads, device, ddp=False):
    sq = torch.zeros((), device=device, dtype=torch.float32)
    for g in grads:
        if g is None:
            continue
        sq = sq + g.detach().float().pow(2).sum()
    if ddp:
        import torch.distributed as dist
        dist.all_reduce(sq, op=dist.ReduceOp.SUM)
    return torch.sqrt(sq + 1e-12)


while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if (iter_num % qrate_every == 0):
        if iter_num < gamma_start_iter:
            if master_process:
                gamma = 0.0
            if ddp:
                import torch.distributed as dist
                tgam = torch.tensor([gamma], device=device, dtype=torch.float32)
                dist.broadcast(tgam, src=0)
                gamma = float(tgam.item())
        else:
            if master_process:
                last_qrate, last_sat_rate, last_qtotal, last_qactive_total = compute_qrate_layerwise(
                    raw_model,
                    n_bits_w=n_bits_w,
                    atol=qrate_atol,
                    include_substrings=quant_include_substrings,
                    exclude_substrings=exclude_substrings,
                )
                if qrate_ema is None:
                    qrate_ema = float(last_qrate)
                else:
                    qrate_ema = qrate_ema_momentum * qrate_ema + (1.0 - qrate_ema_momentum) * float(last_qrate)

                gamma = float(gamma + gamma_lr * (q_target - qrate_ema))
                gamma = max(0.0, min(gamma, gamma_max))
                print(f"[gamma] qrate={last_qrate:.4f} ema={qrate_ema:.4f} -> gamma={gamma:.6f} (target={q_target:.3f})")

            if ddp:
                import torch.distributed as dist
                tgam = torch.tensor([gamma], device=device, dtype=torch.float32)
                dist.broadcast(tgam, src=0)
                gamma = float(tgam.item())

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        hard_losses = estimate_loss_selective_hard()

        if last_qrate is None:
            qrate_str = 'N/A'
        else:
            qrate_str = f"{last_qrate * 100:.2f}%"
            if last_sat_rate is not None and last_qtotal is not None:
                if last_qactive_total is not None:
                    qrate_str = (
                        f"{last_qrate * 100:.2f}% sat {last_sat_rate * 100:.2f}% "
                        f"(active={last_qactive_total:,}, total={last_qtotal:,})"
                    )
                else:
                    qrate_str = f"{last_qrate * 100:.2f}% sat {last_sat_rate * 100:.2f}% (N={last_qtotal:,})"

        print(f"[eval] iter {iter_num}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}, qrate {qrate_str}")
        print(f"[eval-hard-selective] iter {iter_num}: train_loss {hard_losses['train']:.4f}, val_loss {hard_losses['val']:.4f}")
        print(f"[lagrange] lambda={dual_lambda:.6f} beta={beta:.4f}")

        if wandb_log:
            wandb_payload = {
                'iter': iter_num,
                'train/loss': float(losses['train']),
                'val/loss': float(losses['val']),
                'train/hard_selective_loss': float(hard_losses['train']),
                'val/hard_selective_loss': float(hard_losses['val']),
                'lr': lr,
                'gamma': gamma,
                'lambda': dual_lambda,
            }
            if last_qrate is not None:
                wandb_payload['qrate'] = last_qrate
            if last_sat_rate is not None:
                wandb_payload['sat_rate'] = last_sat_rate
            if last_qactive_total is not None:
                wandb_payload['q_active_total'] = last_qactive_total
            wandb.log(wandb_payload)

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

    optimizer.zero_grad(set_to_none=True)
    grad_f_norm_val = None
    loss_raw_sum = torch.zeros((), device=device)
    dist_loss = torch.zeros((), device=device)
    dist_dbg = None

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            _, loss_raw = model(X, Y)

        loss_raw_sum += loss_raw.detach()

        if (micro_step == gradient_accumulation_steps - 1) and (iter_num % gradf_every == 0):
            params = []
            for name, p in raw_model.named_parameters():
                if not p.requires_grad:
                    continue
                if any(s in name for s in exclude_substrings):
                    continue
                if quant_include_substrings is not None and (not any(s in name for s in quant_include_substrings)):
                    continue
                params.append(p)

            grads = torch.autograd.grad(loss_raw, params, retain_graph=True, allow_unused=True)
            grad_f_norm_val = float(grad_norm_l2(grads, device=loss_raw.device, ddp=ddp).item())

        if use_lagrange:
            loss_for_backward = dual_lambda * (loss_raw - beta)
            loss_for_backward = loss_for_backward / gradient_accumulation_steps

            if (micro_step == gradient_accumulation_steps - 1) and (gamma > 0.0):
                dist_loss, dist_dbg = compute_dist_loss(
                    raw_model,
                    n_bits_w=n_bits_w,
                    alpha_w=alpha_w,
                    include_substrings=quant_include_substrings,
                    exclude_substrings=exclude_substrings,
                    sens_state=sens_state,
                    return_debug=True,
                )
            else:
                dist_loss = torch.zeros((), device=loss_raw.device)

            loss_for_backward = loss_for_backward + (dist_scale * gamma) * dist_loss
        else:
            loss_for_backward = loss_raw / gradient_accumulation_steps

        X, Y = get_batch('train')
        scaler.scale(loss_for_backward).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    if sens_enable:
        for name, p in raw_model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if any(s in name for s in exclude_substrings):
                continue
            if quant_include_substrings is not None and (not any(s in name for s in quant_include_substrings)):
                continue

            g = p.grad.detach()
            if not torch.isfinite(g).all():
                continue

            g2 = g.float().pow(2).mean()
            if not torch.isfinite(g2):
                continue

            old_s = sens_state.get(name, None)
            if old_s is None:
                sens_state[name] = g2
            else:
                old_s = old_s.detach().float().to(g2.device)
                if not torch.isfinite(old_s):
                    sens_state[name] = g2
                else:
                    new_s = sens_ema * old_s + (1.0 - sens_ema) * g2
                    if torch.isfinite(new_s):
                        sens_state[name] = new_s

    scaler.step(optimizer)
    scaler.update()

    if use_lagrange:
        f_mean = (loss_raw_sum / gradient_accumulation_steps).detach().float()
        if not torch.isfinite(f_mean):
            f_mean = torch.tensor(beta, device=device, dtype=torch.float32)

        if ddp:
            import torch.distributed as dist
            dist.all_reduce(f_mean, op=dist.ReduceOp.AVG)

        if master_process:
            f_item = float(f_mean.item())
            if math.isfinite(f_item):
                g = float(f_item - beta)

                if lambda_use_pi:
                    lambda_g_int = lambda_g_int + g
                    lambda_g_int = max(-lambda_i_clamp, min(lambda_i_clamp, lambda_g_int))
                    dual_lambda = float(dual_lambda + dual_lr * (lambda_kp * g + lambda_ki * lambda_g_int))
                else:
                    dual_lambda = float(dual_lambda + dual_lr * g)

                dual_lambda = max(0.0, min(dual_lambda, dual_lambda_max))

        if ddp:
            import torch.distributed as dist
            tlam = torch.tensor([dual_lambda], device=device, dtype=torch.float32)
            dist.broadcast(tlam, src=0)
            dual_lambda = float(tlam.item())

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        f_mean_val = float(f_mean.item()) if use_lagrange else (loss_raw_sum / gradient_accumulation_steps).item()
        dist_val = float(dist_loss.detach().item()) if use_lagrange else 0.0

        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        lam_over_gam = float(dual_lambda / (gamma + 1e-12))
        sens_m, sens_x = sens_mean_max(sens_state, device=device, ddp=ddp)
        gradf_str = 'N/A' if grad_f_norm_val is None else f'{grad_f_norm_val:.3e}'

        if dist_dbg is not None:
            K = dist_dbg['K']
            rel_min, rel_mean, rel_max = dist_dbg['rel_min'], dist_dbg['rel_mean'], dist_dbg['rel_max']
            coef_mean = dist_scale * (1.0 / max(K, 1)) * rel_mean
            coef_min = dist_scale * (1.0 / max(K, 1)) * rel_min
            coef_max = dist_scale * (1.0 / max(K, 1)) * rel_max
            active_e = dist_dbg.get('active_elems', 0)
            skipped_sat_e = dist_dbg.get('skipped_sat_elems', 0)
            coef_str = (
                f"coef(no gamma) min/mean/max=({coef_min:.3g},{coef_mean:.3g},{coef_max:.3g}), "
                f"K={K}, active_elems={active_e:,}, skipped_sat={skipped_sat_e:,}"
            )
        else:
            coef_str = 'coef(no gamma)=N/A'

        print(
            f"iter {iter_num}: f_mean {f_mean_val:.4f}, dist {dist_val:.4f}, "
            f"lambda {dual_lambda:.6f}, gamma {gamma:.6f}, lambda/gamma {lam_over_gam:.3e}, "
            f"sens(mean,max)=({sens_m:.3e},{sens_x:.3e}), ||∇f|| {gradf_str}, "
            f"g_int {lambda_g_int:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        )
        print(f"    {coef_str}")

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
