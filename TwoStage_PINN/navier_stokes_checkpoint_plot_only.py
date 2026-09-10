# -*- coding: utf-8 -*-
"""
Navier-Stokes 逆问题：仅加载 checkpoint，重新计算最终结果并生成对比图
--------------------------------------------------------------------

用途：
- 不重新训练 Adam / L-BFGS / NNCG
- 直接读取 v4 已保存的四个 checkpoint
- 重新计算 Total/Data/PDE Loss 与 lambda1/lambda2 识别误差
- 重新生成最终对比图
- 图片保存到原目录：
    ./results/navier_stokes_inverse/

注意：
当前 checkpoint 没有保存完整训练 history，因此：
- 可以重新生成“最终结果类”图
- 不能还原之前完整的 loss/参数训练轨迹曲线
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ============================================================
# 1. 路径与配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "cylinder_nektar_wake.mat"
RESULT_DIR = BASE_DIR / "results" / "navier_stokes_inverse"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "Adam-only": CHECKPOINT_DIR / "navier_stokes_adam_stage1.pt",
    "Adam -> L-BFGS": CHECKPOINT_DIR / "navier_stokes_adam_lbfgs.pt",
    "Adam -> L-BFGS -> L-BFGS\n(matched time)": (
        CHECKPOINT_DIR / "navier_stokes_adam_lbfgs_continue_matched_time.pt"
    ),
    "Adam -> L-BFGS -> NNCG": (
        CHECKPOINT_DIR / "navier_stokes_adam_lbfgs_nncg.pt"
    ),
}

# 注意：
# 第一个 checkpoint 是 Adam 5000，而不是 Adam-only 20000。
# 因为 v4 没有保存 Adam-only 20000 checkpoint。
# 如果你希望把 Adam-only 20000 也放进绘图，需要后续把那个分支也保存 checkpoint。

LAYERS = [3, 64, 64, 64, 64, 2]

LAMBDA1_TRUE = 1.0
LAMBDA2_TRUE = 0.01

# 为了与训练时一致，仍使用 5000 个训练点重新计算训练 Loss。
N_TRAIN = 5000
SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch version:", torch.__version__)
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. 读取数据
# ============================================================

data = loadmat(DATA_PATH)

required_keys = ["X_star", "U_star", "p_star", "t"]

for key in required_keys:
    if key not in data:
        raise KeyError(
            f"数据中缺少 '{key}'。当前 keys = "
            f"{[k for k in data.keys() if not k.startswith('__')]}"
        )

X_star_space = data["X_star"]
U_star = data["U_star"]
p_star = data["p_star"]
t_star = data["t"].flatten()

N = X_star_space.shape[0]
Tn = t_star.shape[0]

XX = np.tile(X_star_space[:, 0:1], (1, Tn))
YY = np.tile(X_star_space[:, 1:2], (1, Tn))
TT = np.tile(t_star.reshape(1, -1), (N, 1))

UU = U_star[:, 0, :]
VV = U_star[:, 1, :]
PP = p_star

x_all = XX.reshape(-1, 1)
y_all = YY.reshape(-1, 1)
t_all = TT.reshape(-1, 1)

u_all = UU.reshape(-1, 1)
v_all = VV.reshape(-1, 1)
p_all = PP.reshape(-1, 1)

n_total = x_all.shape[0]

idx = np.random.choice(
    n_total,
    N_TRAIN,
    replace=False
)

x_train = torch.tensor(
    x_all[idx],
    dtype=torch.float32,
    device=device,
    requires_grad=True
)

y_train = torch.tensor(
    y_all[idx],
    dtype=torch.float32,
    device=device,
    requires_grad=True
)

t_train = torch.tensor(
    t_all[idx],
    dtype=torch.float32,
    device=device,
    requires_grad=True
)

u_train = torch.tensor(
    u_all[idx],
    dtype=torch.float32,
    device=device
)

v_train = torch.tensor(
    v_all[idx],
    dtype=torch.float32,
    device=device
)


# ============================================================
# 3. PINN 模型
# ============================================================

class NavierStokesPINN(nn.Module):

    def __init__(self, layers):
        super().__init__()

        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1])
            for i in range(len(layers) - 1)
        ])

        self.activation = nn.Tanh()

        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.lambda1 = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )

        self.lambda2 = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )

    def forward(self, X):

        H = X

        for layer in self.linears[:-1]:
            H = self.activation(layer(H))

        out = self.linears[-1](H)

        psi = out[:, 0:1]
        p = out[:, 1:2]

        return psi, p


def grad(outputs, inputs):

    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]


# ============================================================
# 4. Navier-Stokes residual 与 Loss
# ============================================================

def navier_stokes_terms(model, x, y, t):

    X = torch.cat([x, y, t], dim=1)

    psi, p = model(X)

    psi_x = grad(psi, x)
    psi_y = grad(psi, y)

    u = psi_y
    v = -psi_x

    u_t = grad(u, t)
    u_x = grad(u, x)
    u_y = grad(u, y)

    v_t = grad(v, t)
    v_x = grad(v, x)
    v_y = grad(v, y)

    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)

    v_xx = grad(v_x, x)
    v_yy = grad(v_y, y)

    p_x = grad(p, x)
    p_y = grad(p, y)

    lambda1 = model.lambda1
    lambda2 = model.lambda2

    f_u = (
        u_t
        + lambda1 * (u * u_x + v * u_y)
        + p_x
        - lambda2 * (u_xx + u_yy)
    )

    f_v = (
        v_t
        + lambda1 * (u * v_x + v * v_y)
        + p_y
        - lambda2 * (v_xx + v_yy)
    )

    return u, v, p, f_u, f_v


mse = nn.MSELoss()


def losses_ns(model):

    u_pred, v_pred, _, f_u, f_v = navier_stokes_terms(
        model,
        x_train,
        y_train,
        t_train
    )

    loss_data = (
        mse(u_pred, u_train)
        + mse(v_pred, v_train)
    )

    loss_pde = (
        mse(f_u, torch.zeros_like(f_u))
        + mse(f_v, torch.zeros_like(f_v))
    )

    loss_total = loss_data + loss_pde

    return loss_total, loss_data, loss_pde


def evaluate_model(model):

    loss, loss_data, loss_pde = losses_ns(model)

    lambda1 = model.lambda1.detach().item()
    lambda2 = model.lambda2.detach().item()

    err1 = abs(lambda1 - LAMBDA1_TRUE) / abs(LAMBDA1_TRUE) * 100.0
    err2 = abs(lambda2 - LAMBDA2_TRUE) / abs(LAMBDA2_TRUE) * 100.0

    return {
        "total_loss": loss.detach().item(),
        "data_loss": loss_data.detach().item(),
        "pde_loss": loss_pde.detach().item(),
        "lambda1": lambda1,
        "lambda2": lambda2,
        "lambda1_error": err1,
        "lambda2_error": err2,
    }


# ============================================================
# 5. 加载 checkpoint
# ============================================================

models = {}
results = {}

for name, ckpt_path in CHECKPOINTS.items():

    if not ckpt_path.exists():
        print(f"[Skip] checkpoint 不存在: {ckpt_path}")
        continue

    model = NavierStokesPINN(
        LAYERS
    ).to(device)

    ckpt = torch.load(
        ckpt_path,
        map_location=device
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    models[name] = model

    print(f"\nLoaded: {name}")
    print(f"  {ckpt_path.name}")


# ============================================================
# 6. 重新计算最终指标
# ============================================================

print("\n========== Recomputed Final Results ==========")

for name, model in models.items():

    result = evaluate_model(model)

    results[name] = result

    print(f"\n{name}")
    print(f"Total Loss    = {result['total_loss']:.3e}")
    print(f"Data Loss     = {result['data_loss']:.3e}")
    print(f"PDE Loss      = {result['pde_loss']:.3e}")
    print(f"lambda1       = {result['lambda1']:.6f}")
    print(f"lambda1 error = {result['lambda1_error']:.3f}%")
    print(f"lambda2       = {result['lambda2']:.6f}")
    print(f"lambda2 error = {result['lambda2_error']:.3f}%")


# ============================================================
# 7. 图 1：Total / Data / PDE Loss 对比
# ============================================================

names = list(results.keys())

x_pos = np.arange(
    len(names)
)

width = 0.25

total_losses = [
    results[n]["total_loss"]
    for n in names
]

data_losses = [
    results[n]["data_loss"]
    for n in names
]

pde_losses = [
    results[n]["pde_loss"]
    for n in names
]

plt.figure(
    figsize=(11, 5.8)
)

plt.bar(
    x_pos - width,
    total_losses,
    width,
    label="Total Loss"
)

plt.bar(
    x_pos,
    data_losses,
    width,
    label="Data Loss"
)

plt.bar(
    x_pos + width,
    pde_losses,
    width,
    label="PDE Loss"
)

plt.yscale("log")

plt.xticks(
    x_pos,
    names
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Navier-Stokes Inverse Problem: Final Loss Comparison"
)

plt.legend()
plt.grid(
    axis="y",
    alpha=0.3
)
plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "navier_stokes_final_loss_comparison.png",
    dpi=200
)

plt.show()


# ============================================================
# 8. 图 2：lambda1 参数值局部放大
# ============================================================

lambda1_values = [
    results[n]["lambda1"]
    for n in names
]

plt.figure(
    figsize=(10, 5.5)
)

plt.bar(
    x_pos,
    lambda1_values
)

plt.axhline(
    LAMBDA1_TRUE,
    linestyle="--",
    label="lambda1 true"
)

vmin = min(
    min(lambda1_values),
    LAMBDA1_TRUE
)

vmax = max(
    max(lambda1_values),
    LAMBDA1_TRUE
)

pad = max(
    (vmax - vmin) * 0.25,
    5e-4
)

plt.ylim(
    vmin - pad,
    vmax + pad
)

plt.xticks(
    x_pos,
    names
)

plt.ylabel(
    "Identified lambda1"
)

plt.title(
    "Navier-Stokes Parameter Identification: lambda1 (Zoomed)"
)

plt.legend()
plt.grid(
    axis="y",
    alpha=0.3
)
plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "navier_stokes_lambda1_values_zoomed.png",
    dpi=200
)

plt.show()


# ============================================================
# 9. 图 3：lambda2 参数值局部放大
# ============================================================

lambda2_values = [
    results[n]["lambda2"]
    for n in names
]

plt.figure(
    figsize=(10, 5.5)
)

plt.bar(
    x_pos,
    lambda2_values
)

plt.axhline(
    LAMBDA2_TRUE,
    linestyle="--",
    label="lambda2 true"
)

vmin = min(
    min(lambda2_values),
    LAMBDA2_TRUE
)

vmax = max(
    max(lambda2_values),
    LAMBDA2_TRUE
)

pad = max(
    (vmax - vmin) * 0.25,
    2e-4
)

plt.ylim(
    vmin - pad,
    vmax + pad
)

plt.xticks(
    x_pos,
    names
)

plt.ylabel(
    "Identified lambda2"
)

plt.title(
    "Navier-Stokes Parameter Identification: lambda2 (Zoomed)"
)

plt.legend()
plt.grid(
    axis="y",
    alpha=0.3
)
plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "navier_stokes_lambda2_values_zoomed.png",
    dpi=200
)

plt.show()


# ============================================================
# 10. 图 4：参数相对误差
# ============================================================

lambda1_errors = [
    results[n]["lambda1_error"]
    for n in names
]

lambda2_errors = [
    results[n]["lambda2_error"]
    for n in names
]

width = 0.35

plt.figure(
    figsize=(11, 5.8)
)

plt.bar(
    x_pos - width / 2,
    lambda1_errors,
    width,
    label="lambda1 error"
)

plt.bar(
    x_pos + width / 2,
    lambda2_errors,
    width,
    label="lambda2 error"
)

plt.xticks(
    x_pos,
    names
)

plt.ylabel(
    "Relative Error (%)"
)

plt.title(
    "Navier-Stokes Parameter Identification Error"
)

plt.legend()
plt.grid(
    axis="y",
    alpha=0.3
)
plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "navier_stokes_parameter_error_recomputed.png",
    dpi=200
)

plt.show()


# ============================================================
# 11. 图 5：只比较 L-BFGS endpoint 与两个后续分支
# ============================================================

post_names = [
    n for n in names
    if n != "Adam-only"
]

if len(post_names) >= 2:

    post_total = [
        results[n]["total_loss"]
        for n in post_names
    ]

    post_pde = [
        results[n]["pde_loss"]
        for n in post_names
    ]

    xp = np.arange(
        len(post_names)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 5.5)
    )

    plt.bar(
        xp - width / 2,
        post_total,
        width,
        label="Total Loss"
    )

    plt.bar(
        xp + width / 2,
        post_pde,
        width,
        label="PDE Loss"
    )

    plt.yscale("log")

    plt.xticks(
        xp,
        post_names
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Navier-Stokes: Post-L-BFGS Final Comparison"
    )

    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.3
    )
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR
        / "navier_stokes_post_lbfgs_final_comparison.png",
        dpi=200
    )

    plt.show()


print(
    f"\nFigures saved to: "
    f"{RESULT_DIR.resolve()}"
)

print(
    "\n注意：v4 checkpoint 没有保存完整 history，"
    "因此本脚本不能恢复完整训练轨迹。"
)
