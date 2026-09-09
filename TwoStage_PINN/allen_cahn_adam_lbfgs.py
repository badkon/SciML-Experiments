# -*- coding: utf-8 -*-
"""
Allen-Cahn 方程正问题：Adam-only vs Adam -> L-BFGS

实验目标
--------
验证在比 Burgers 更难优化的 Allen-Cahn PINN 中：
1. Adam 后期是否出现明显平台 / 震荡；
2. 从相同的 Adam-15000 checkpoint 切换到 L-BFGS，
   是否比继续 Adam 到 20000 获得更低 Loss；
3. 是否真正改善完整解场的 Relative L2 Error。

默认数据文件
------------
./data/AC.mat
"""

from pathlib import Path
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ============================================================
# 1. 全局配置
# ============================================================

SEED = 42

# 当前 Python 文件所在目录

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "AC.mat"
RESULT_DIR = BASE_DIR / "results" / "allen_cahn_two_stage"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

N_U = 500
N_F = 20000

ADAM_STAGE1 = 15000
ADAM_CONTINUE = 5000

LEARNING_RATE = 1e-3

EPSILON = 1e-4

LAYERS = [2, 64, 64, 64, 64, 1]


def set_seed(seed: int = SEED):
    """固定随机种子，便于复现实验。"""
    random.seed(seed)
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
# 2. 读取 Allen-Cahn 数据
# ============================================================

data = loadmat(DATA_PATH)

x = data["x"].flatten()        # shape: (512,)
t = data["tt"].flatten()       # shape: (201,)
Exact = np.real(data["uu"])    # shape: (512, 201)

X, T = np.meshgrid(x, t, indexing="ij")

# X_star: (102912, 2)，每一行是 [x_i, t_i]
# u_star: (102912, 1)，对应完整参考解
X_star = np.hstack(
    (X.reshape(-1, 1), T.reshape(-1, 1))
)
u_star = Exact.reshape(-1, 1)

print("x shape:", x.shape)
print("t shape:", t.shape)
print("Exact shape:", Exact.shape)
print("X_star shape:", X_star.shape)


# ============================================================
# 3. 构造训练点
# ============================================================

idx_u = np.random.choice(
    X_star.shape[0],
    N_U,
    replace=False
)

X_u = X_star[idx_u, :]      # shape: (N_U, 2)
u_u = u_star[idx_u, :]      # shape: (N_U, 1)

X_u_tensor = torch.tensor(
    X_u,
    dtype=torch.float32,
    device=device
)

u_u_tensor = torch.tensor(
    u_u,
    dtype=torch.float32,
    device=device
)

lb = X_star.min(axis=0)
ub = X_star.max(axis=0)

X_f = lb + (ub - lb) * np.random.rand(N_F, 2)

X_f_tensor = torch.tensor(
    X_f,
    dtype=torch.float32,
    device=device,
    requires_grad=True
)


# ============================================================
# 4. PINN 网络
# ============================================================

class PINN(nn.Module):
    """输入 (x, t)，输出 Allen-Cahn 解 u(x, t)。"""

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

    def forward(self, X_input):
        H = X_input

        for layer in self.linears[:-1]:
            H = self.activation(layer(H))

        return self.linears[-1](H)


# ============================================================
# 5. Allen-Cahn PDE residual
# ============================================================

def pde_residual_ac(model, X_input):
    """
    Allen-Cahn 方程：
        u_t - epsilon * u_xx + 5u^3 - 5u = 0
    """

    x_in = X_input[:, 0:1]
    t_in = X_input[:, 1:2]

    u = model(torch.cat([x_in, t_in], dim=1))

    grads = torch.autograd.grad(
        u,
        X_input,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]

    grads_x = torch.autograd.grad(
        u_x,
        X_input,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    u_xx = grads_x[:, 0:1]

    f = (
        u_t
        - EPSILON * u_xx
        + 5.0 * u**3
        - 5.0 * u
    )

    return f


mse = nn.MSELoss()


def data_loss(model):
    """观测点上的数据拟合损失。"""
    return mse(
        model(X_u_tensor),
        u_u_tensor
    )


def physics_loss_ac(model):
    """Allen-Cahn PDE residual loss。"""
    f = pde_residual_ac(
        model,
        X_f_tensor
    )
    return mse(
        f,
        torch.zeros_like(f)
    )


def evaluate_ac(model):
    """
    同时计算：
    1. Total / Data / PDE Loss
    2. 完整时空场 Relative L2 Error
    """

    loss_u = data_loss(model)
    loss_f = physics_loss_ac(model)
    total_loss = loss_u + loss_f

    X_star_tensor = torch.tensor(
        X_star,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        u_pred = model(X_star_tensor).cpu().numpy()

    relative_l2 = (
        np.linalg.norm(u_pred - u_star)
        /
        np.linalg.norm(u_star)
    )

    return (
        total_loss.item(),
        loss_u.item(),
        loss_f.item(),
        relative_l2
    )


# ============================================================
# 6. 共同阶段：Adam 训练 15000 轮
# ============================================================

base_model = PINN(LAYERS).to(device)

optimizer_adam = torch.optim.Adam(
    base_model.parameters(),
    lr=LEARNING_RATE
)

adam_history_ac = {
    "total": [],
    "data": [],
    "physics": []
}

print("\n========== Stage 1: Adam 0 -> 15000 ==========")

for epoch in range(ADAM_STAGE1):

    optimizer_adam.zero_grad()

    loss_u = data_loss(base_model)
    loss_f = physics_loss_ac(base_model)
    loss = loss_u + loss_f

    loss.backward()
    optimizer_adam.step()

    adam_history_ac["total"].append(loss.item())
    adam_history_ac["data"].append(loss_u.item())
    adam_history_ac["physics"].append(loss_f.item())

    if epoch % 1000 == 0:
        print(
            f"Adam | Epoch {epoch:5d} | "
            f"Loss: {loss.item():.3e} | "
            f"Data: {loss_u.item():.3e} | "
            f"PDE: {loss_f.item():.3e}"
        )


# ============================================================
# 7. 保存 Adam-15000 checkpoint 并创建两个分支
# ============================================================

adam_15000_state_ac = copy.deepcopy(
    base_model.state_dict()
)

# 分支 A：继续 Adam
adam_only_model_ac = PINN(LAYERS).to(device)
adam_only_model_ac.load_state_dict(adam_15000_state_ac)

# 分支 B：切换 L-BFGS
lbfgs_model_ac = PINN(LAYERS).to(device)
lbfgs_model_ac.load_state_dict(adam_15000_state_ac)


# ============================================================
# 8. 分支 A：Adam-only 15000 -> 20000
# ============================================================

optimizer_adam_only_ac = torch.optim.Adam(
    adam_only_model_ac.parameters(),
    lr=LEARNING_RATE
)

adam_only_history_ac = []

print("\n========== Branch A: Adam-only 15000 -> 20000 ==========")

for epoch in range(ADAM_CONTINUE):

    optimizer_adam_only_ac.zero_grad()

    loss_u = data_loss(adam_only_model_ac)
    loss_f = physics_loss_ac(adam_only_model_ac)
    loss = loss_u + loss_f

    loss.backward()
    optimizer_adam_only_ac.step()

    adam_only_history_ac.append(loss.item())

    if epoch % 1000 == 0:
        print(
            f"Adam-only | Epoch {epoch + ADAM_STAGE1:5d} | "
            f"Loss: {loss.item():.3e}"
        )


# ============================================================
# 9. 分支 B：Adam -> L-BFGS
# ============================================================

lbfgs_history_ac = []

optimizer_lbfgs_ac = torch.optim.LBFGS(
    lbfgs_model_ac.parameters(),
    lr=1.0,
    max_iter=5000,
    max_eval=5000,
    tolerance_grad=1e-9,
    tolerance_change=1e-12,
    history_size=50,
    line_search_fn="strong_wolfe"
)


def closure_lbfgs_ac():
    """
    L-BFGS closure。
    closure 调用次数用于观察内部优化轨迹，
    但不与 Adam epoch 或论文 iteration 严格等价。
    """

    optimizer_lbfgs_ac.zero_grad()

    loss_u = data_loss(lbfgs_model_ac)
    loss_f = physics_loss_ac(lbfgs_model_ac)
    loss = loss_u + loss_f

    loss.backward()

    lbfgs_history_ac.append(loss.item())

    return loss


print("\n========== Branch B: Adam 15000 -> L-BFGS ==========")
optimizer_lbfgs_ac.step(closure_lbfgs_ac)


# ============================================================
# 10. 可选：确认 L-BFGS 是否已基本收敛
# ============================================================

# 这里保持完全相同的 L-BFGS 配置，
# 直接从当前已优化模型继续一次。
# 如果只产生极少 closure 调用且结果不变，
# 可以认为当前配置下已经基本收敛。

extra_history = []

optimizer_lbfgs_ac_extra = torch.optim.LBFGS(
    lbfgs_model_ac.parameters(),
    lr=1.0,
    max_iter=5000,
    max_eval=5000,
    tolerance_grad=1e-9,
    tolerance_change=1e-12,
    history_size=50,
    line_search_fn="strong_wolfe"
)


def closure_lbfgs_ac_extra():
    optimizer_lbfgs_ac_extra.zero_grad()

    loss_u = data_loss(lbfgs_model_ac)
    loss_f = physics_loss_ac(lbfgs_model_ac)
    loss = loss_u + loss_f

    loss.backward()

    extra_history.append(loss.item())

    return loss


optimizer_lbfgs_ac_extra.step(closure_lbfgs_ac_extra)


# ============================================================
# 11. 最终结果
# ============================================================

adam_result = evaluate_ac(adam_only_model_ac)
lbfgs_result = evaluate_ac(lbfgs_model_ac)

print("\n========== Final Results ==========")
print(f"Primary L-BFGS closure calls: {len(lbfgs_history_ac)}")
print(f"Extra L-BFGS closure calls: {len(extra_history)}")

print("\nAdam-only (20000)")
print(f"Total Loss  = {adam_result[0]:.3e}")
print(f"Data Loss   = {adam_result[1]:.3e}")
print(f"PDE Loss    = {adam_result[2]:.3e}")
print(f"Relative L2 = {adam_result[3]:.3e}")

print("\nAdam -> L-BFGS")
print(f"Total Loss  = {lbfgs_result[0]:.3e}")
print(f"Data Loss   = {lbfgs_result[1]:.3e}")
print(f"PDE Loss    = {lbfgs_result[2]:.3e}")
print(f"Relative L2 = {lbfgs_result[3]:.3e}")


# ============================================================
# 12. 生成完整预测场
# ============================================================

X_star_tensor = torch.tensor(
    X_star,
    dtype=torch.float32,
    device=device
)

with torch.no_grad():

    u_pred_adam = (
        adam_only_model_ac(X_star_tensor)
        .cpu()
        .numpy()
        .reshape(Exact.shape)
    )

    u_pred_lbfgs = (
        lbfgs_model_ac(X_star_tensor)
        .cpu()
        .numpy()
        .reshape(Exact.shape)
    )

error_adam = np.abs(u_pred_adam - Exact)
error_lbfgs = np.abs(u_pred_lbfgs - Exact)


# ============================================================
# 13. 对比图 1：Loss 优化轨迹
# ============================================================

adam_stage1_steps = np.arange(
    len(adam_history_ac["total"])
)

adam_only_steps = np.arange(
    ADAM_STAGE1,
    ADAM_STAGE1 + len(adam_only_history_ac)
)

lbfgs_steps = np.arange(
    ADAM_STAGE1,
    ADAM_STAGE1 + len(lbfgs_history_ac)
)

plt.figure(figsize=(10, 5))

plt.semilogy(
    adam_stage1_steps,
    adam_history_ac["total"],
    label="Adam: shared stage"
)

plt.semilogy(
    adam_only_steps,
    adam_only_history_ac,
    label="Adam-only continuation"
)

plt.semilogy(
    lbfgs_steps,
    lbfgs_history_ac,
    label="L-BFGS continuation"
)

plt.axvline(
    ADAM_STAGE1,
    linestyle="--",
    label="Optimizer switch"
)

# 限制纵轴，避免 line search 的少数试探点破坏主要趋势显示
plt.ylim(1e-6, 2)

plt.xlabel("Optimization progress")
plt.ylabel("Total Loss")
plt.title("Allen-Cahn: Adam-only vs Adam -> L-BFGS")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "allen_cahn_loss_comparison.png",
    dpi=200
)
plt.show()


# ============================================================
# 14. 对比图 2：全场绝对误差对比
# ============================================================

# 为了让两张误差图可直接比较，统一色标上限
error_vmax = max(error_adam.max(), error_lbfgs.max())

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4.8),
    constrained_layout=True
)

im0 = axes[0].imshow(
    error_adam,
    aspect="auto",
    origin="lower",
    extent=[
        t.min(),
        t.max(),
        x.min(),
        x.max()
    ],
    vmin=0,
    vmax=error_vmax
)

axes[0].set_xlabel("t")
axes[0].set_ylabel("x")
axes[0].set_title(
    f"Adam-only Absolute Error\nRelative L2 = {adam_result[3]:.3e}"
)

im1 = axes[1].imshow(
    error_lbfgs,
    aspect="auto",
    origin="lower",
    extent=[
        t.min(),
        t.max(),
        x.min(),
        x.max()
    ],
    vmin=0,
    vmax=error_vmax
)

axes[1].set_xlabel("t")
axes[1].set_ylabel("x")
axes[1].set_title(
    f"Adam -> L-BFGS Absolute Error\nRelative L2 = {lbfgs_result[3]:.3e}"
)

fig.colorbar(
    im1,
    ax=axes,
    label="Absolute Error"
)

plt.savefig(
    RESULT_DIR / "allen_cahn_error_comparison.png",
    dpi=200
)
plt.show()


# ============================================================
# 15. 对比图 3：最终指标柱状图
# ============================================================

methods = ["Adam-only", "Adam -> L-BFGS"]

relative_l2_values = [
    adam_result[3],
    lbfgs_result[3]
]

plt.figure(figsize=(7, 5))

plt.bar(
    methods,
    relative_l2_values
)

plt.ylabel("Relative L2 Error")
plt.title("Allen-Cahn Full-field Relative L2 Error")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "allen_cahn_relative_l2.png",
    dpi=200
)
plt.show()

print(f"\nFigures saved to: {RESULT_DIR.resolve()}")
