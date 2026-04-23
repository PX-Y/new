# GPT-2 XL + WikiText-103 + soft quantization (no STE)
# For the training script: train_xl_gammahist_soft.py


# ----------------
# data / init
# ----------------
out_dir = "out_gpt2sl_wikitext103_gammahist_4b"
dataset = "wikitext103"
init_from = "gpt2-large"
compile = False

dtype = "float16"
device = "cuda"

# ----------------
# batch / context
# ----------------
# Conservative default for XL. If you are on 80GB and stable,
# you can try batch_size = 4, gradient_accumulation_steps = 8.
batch_size = 4
block_size = 256
gradient_accumulation_steps = 8

# ----------------
# schedule
# ----------------
max_iters =5000
lr_decay_iters = 5000
warmup_iters = 100

eval_interval = 100
eval_iters = 100
log_interval = 20
always_save_checkpoint = True

# ----------------
# optimizer
# ----------------
learning_rate = 6e-5
min_lr = 5e-6
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True

# ----------------
# quant / qrate monitor
# ----------------
n_bits_w = 8
alpha_w = 0.08/127
qrate_every = 100
qrate_atol = 0.00007
quant_include_substrings = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
exclude_substrings = ("ln_", "bias", "wte", "wpe")


# ----------------
# Lagrange control
# ----------------
use_lagrange = True
beta = 2.6
dual_lr =3e-3
dual_lambda = 0.5
dual_lambda_max = 40
dist_scale = 3

# ----------------
# gamma / qrate controller
# ----------------
q_target = 0.85
gamma_start_iter = 500 #700
gamma_lr = 1
gamma_max = 30.0
qrate_ema_momentum = 0.9

# ----------------
# sensitivity-weighted dist
# ----------------
sens_enable = True
sens_ema = 0.95
sens_eps = 1e-8
sens_power = 0.5
sens_w_min = 0.2
sens_w_max = 7.0
gradf_every = 100

# ----------------
# lambda PI control
# ----------------
lambda_use_pi = True
lambda_kp = 1.0
lambda_ki = 0.2
lambda_i_clamp = 10.0

# optional per-layer step json; leave empty to use global alpha_w
layer_step_json = ""

hard_start = max_iters - 500
