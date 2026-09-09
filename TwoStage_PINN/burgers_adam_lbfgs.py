# -*- coding: utf-8 -*-
"""
Burgers 方程参数识别：Adam-only vs Adam -> L-BFGS

实验目标
--------
验证论文中的“两阶段优化策略”：
1. 前 15000 轮使用 Adam；
2. 从完全相同的 15000 轮 checkpoint 分叉：
   - 分支 A：继续 Adam 到 20000 轮；
   - 分支 B：切换到 L-BFGS；
3. 比较最终损失与 PDE 参数识别精度。

注意
----
- 本脚本只保留与“Adam -> L-BFGS 两阶段优化”直接相关的正式实验。
- 不包含此前尝试过的“先仅拟合数据、再加入 PDE”的探索性训练。
- 默认数据文件位于 ./data/burgers_shock.mat
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

DATA_PATH = Path("./data/burgers_shock.mat")
# 当前 Python 文件所在目录
BASE_DIR = Path(__file__).resolve().parent

# 无论从哪个终端目录运行，都相对于本脚本寻找数据和保存结果
DATA_PATH = BASE_DIR / "data" / "burgers_shock.mat"
RESULT_DIR = BASE_DIR / "results" / "burgers_two_stage"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

N_U = 2000          # 有真实 u 的观测点数
N_F = 10000         # PDE collocation points
ADAM_STAGE1 = 15000
ADAM_CONTINUE = 5000

LEARNING_RATE = 1e-3

# Burgers 方程真值：
# u_t + lambda_1 * u * u_x - lambda_2 * u_xx = 0
LAMBDA_1_TRUE = 1.0
LAMBDA_2_TRUE = 0.01 / np.pi

LAYERS = [2, 50, 50, 50, 50, 1]


def set_seed(seed: int = SEED):
    """固定随机种子，保证采样与网络初始化尽可能可复现。"""
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
# 2. 读取 Burgers 数据
# ============================================================

data = loadmat(DATA_PATH)

x = data["x"].flatten()                   # shape: (256,)
t = data["t"].flatten()                   # shape: (100,)
Exact = np.real(data["usol"])             # shape: (256, 100)

# indexing="ij" 保证第一维对应 x，第二维对应 t，
# 从而与 Exact 的 (256, 100) 排列一致。
X, T = np.meshgrid(x, t, indexing="ij")

# 将二维时空网格展开成 PINN 常用格式：
# X_star: (25600, 2)，每一行是 [x_i, t_i]
# u_star: (25600, 1)，对应真实解 u(x_i, t_i)
X_star = np.hstack(
    (X.reshape(-1, 1), T.reshape(-1, 1))
)
u_star = Exact.reshape(-1, 1)

print("x shape:", x.shape)
print("t shape:", t.shape)
print("Exact shape:", Exact.shape)
print("X_star shape:", X_star.shape)
print("u_star shape:", u_star.shape)


# ============================================================
# 3. 构造观测点与 PDE 配点
# ============================================================

# 观测点：有真实 u 值，用于 data loss
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

# PDE 配点：不提供真实 u，只用于计算 PDE residual
lb = X_star.min(axis=0)     # [x_min, t_min]
ub = X_star.max(axis=0)     # [x_max, t_max]

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
    """简单全连接 PINN：输入 (x, t)，输出 u(x, t)。"""

    def __init__(self, layers):
        super().__init__()

        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1])
            for i in range(len(layers) - 1)
        ])

        self.activation = nn.Tanh()

        # PINN 中常用 Xavier 初始化
        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, X_input):
        H = X_input

        # 隐藏层使用 tanh
        for layer in self.linears[:-1]:
            H = self.activation(layer(H))

        # 输出层不加激活函数
        return self.linears[-1](H)


# ============================================================
# 5. Burgers PDE residual
# ============================================================

def pde_residual(model, X_input, lambda_1, lambda_2):
    """
    计算 Burgers 方程残差：
        f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx

    X_input:
        shape = (N_F, 2)
        第 0 列是 x，第 1 列是 t
    """

    x_in = X_input[:, 0:1]
    t_in = X_input[:, 1:2]

    u = model(torch.cat([x_in, t_in], dim=1))

    # 一阶导：[u_x, u_t]
    grads = torch.autograd.grad(
        u,
        X_input,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]

    # 二阶空间导 u_xx
    grads_x = torch.autograd.grad(
        u_x,
        X_input,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    u_xx = grads_x[:, 0:1]

    f = (
        u_t
        + lambda_1 * u * u_x
        - lambda_2 * u_xx
    )

    return f


mse = nn.MSELoss()


def data_loss(model):
    """观测数据拟合损失。"""
    u_pred = model(X_u_tensor)
    return mse(u_pred, u_u_tensor)


def physics_loss(model, lambda_1, lambda_2):
    """Burgers PDE residual loss。"""
    f = pde_residual(
        model,
        X_f_tensor,
        lambda_1,
        lambda_2
    )
    return mse(f, torch.zeros_like(f))


def evaluate_loss(model, lambda_1, lambda_2):
    """重新计算最终 Total / Data / PDE Loss。"""
    loss_u = data_loss(model)
    loss_f = physics_loss(model, lambda_1, lambda_2)
    total = loss_u + loss_f

    return (
        total.item(),
        loss_u.item(),
        loss_f.item()
    )


# ============================================================
# 6. 共同阶段：Adam 训练 15000 轮
# ============================================================

base_model = PINN(LAYERS).to(device)

lambda_1 = nn.Parameter(
    torch.tensor(0.0, dtype=torch.float32, device=device)
)
lambda_2 = nn.Parameter(
    torch.tensor(0.0, dtype=torch.float32, device=device)
)

optimizer_adam = torch.optim.Adam(
    list(base_model.parameters()) + [lambda_1, lambda_2],
    lr=LEARNING_RATE
)

adam_history = {
    "total": [],
    "data": [],
    "physics": [],
    "lambda_1": [],
    "lambda_2": []
}

print("\n========== Stage 1: Adam 0 -> 15000 ==========")

for epoch in range(ADAM_STAGE1):

    optimizer_adam.zero_grad()

    loss_u = data_loss(base_model)
    loss_f = physics_loss(base_model, lambda_1, lambda_2)
    loss = loss_u + loss_f

    loss.backward()
    optimizer_adam.step()

    adam_history["total"].append(loss.item())
    adam_history["data"].append(loss_u.item())
    adam_history["physics"].append(loss_f.item())
    adam_history["lambda_1"].append(lambda_1.item())
    adam_history["lambda_2"].append(lambda_2.item())

    if epoch % 1000 == 0:
        print(
            f"Adam | Epoch {epoch:5d} | "
            f"Loss: {loss.item():.3e} | "
            f"Data: {loss_u.item():.3e} | "
            f"PDE: {loss_f.item():.3e} | "
            f"lambda_1: {lambda_1.item():.6f} | "
            f"lambda_2: {lambda_2.item():.6f}"
        )


# ============================================================
# 7. 保存 Adam-15000 checkpoint，并从同一起点分叉
# ============================================================

adam_15000_model_state = copy.deepcopy(base_model.state_dict())
adam_15000_lambda_1 = lambda_1.detach().clone()
adam_15000_lambda_2 = lambda_2.detach().clone()

# 分支 A：继续 Adam
adam_only_model = PINN(LAYERS).to(device)
adam_only_model.load_state_dict(adam_15000_model_state)

adam_only_lambda_1 = nn.Parameter(
    adam_15000_lambda_1.clone().to(device)
)
adam_only_lambda_2 = nn.Parameter(
    adam_15000_lambda_2.clone().to(device)
)

# 分支 B：切换 L-BFGS
lbfgs_model = PINN(LAYERS).to(device)
lbfgs_model.load_state_dict(adam_15000_model_state)

lbfgs_lambda_1 = nn.Parameter(
    adam_15000_lambda_1.clone().to(device)
)
lbfgs_lambda_2 = nn.Parameter(
    adam_15000_lambda_2.clone().to(device)
)


# ============================================================
# 8. 分支 A：Adam-only 继续到 20000
# ============================================================

optimizer_adam_only = torch.optim.Adam(
    list(adam_only_model.parameters())
    + [adam_only_lambda_1, adam_only_lambda_2],
    lr=LEARNING_RATE
)

adam_only_history = {
    "total": [],
    "lambda_1": [],
    "lambda_2": []
}

print("\n========== Branch A: Adam-only 15000 -> 20000 ==========")

for epoch in range(ADAM_CONTINUE):

    optimizer_adam_only.zero_grad()

    loss_u = data_loss(adam_only_model)
    loss_f = physics_loss(
        adam_only_model,
        adam_only_lambda_1,
        adam_only_lambda_2
    )
    loss = loss_u + loss_f

    loss.backward()
    optimizer_adam_only.step()

    adam_only_history["total"].append(loss.item())
    adam_only_history["lambda_1"].append(
        adam_only_lambda_1.item()
    )
    adam_only_history["lambda_2"].append(
        adam_only_lambda_2.item()
    )

    if epoch % 1000 == 0:
        print(
            f"Adam-only | Epoch {epoch + ADAM_STAGE1:5d} | "
            f"Loss: {loss.item():.3e} | "
            f"lambda_1: {adam_only_lambda_1.item():.6f} | "
            f"lambda_2: {adam_only_lambda_2.item():.6f}"
        )


# ============================================================
# 9. 分支 B：Adam -> L-BFGS
# ============================================================

lbfgs_history = {
    "loss": [],
    "lambda_1": [],
    "lambda_2": []
}

optimizer_lbfgs = torch.optim.LBFGS(
    list(lbfgs_model.parameters())
    + [lbfgs_lambda_1, lbfgs_lambda_2],
    lr=1.0,
    max_iter=5000,
    max_eval=5000,
    tolerance_grad=1e-9,
    tolerance_change=1e-12,
    history_size=50,
    line_search_fn="strong_wolfe"
)


def closure_lbfgs():
    """
    PyTorch L-BFGS 需要 closure。
    注意：closure 调用次数不等价于论文中的 iteration 次数。
    """
    optimizer_lbfgs.zero_grad()

    loss_u = data_loss(lbfgs_model)
    loss_f = physics_loss(
        lbfgs_model,
        lbfgs_lambda_1,
        lbfgs_lambda_2
    )
    loss = loss_u + loss_f

    loss.backward()

    lbfgs_history["loss"].append(loss.item())
    lbfgs_history["lambda_1"].append(lbfgs_lambda_1.item())
    lbfgs_history["lambda_2"].append(lbfgs_lambda_2.item())

    return loss


print("\n========== Branch B: Adam 15000 -> L-BFGS ==========")
optimizer_lbfgs.step(closure_lbfgs)


# ============================================================
# 10. 最终结果
# ============================================================

adam_total, adam_data, adam_pde = evaluate_loss(
    adam_only_model,
    adam_only_lambda_1,
    adam_only_lambda_2
)

lbfgs_total, lbfgs_data, lbfgs_pde = evaluate_loss(
    lbfgs_model,
    lbfgs_lambda_1,
    lbfgs_lambda_2
)

print("\n========== Final Results ==========")
print("Ground Truth")
print(f"lambda_1 = {LAMBDA_1_TRUE:.6f}")
print(f"lambda_2 = {LAMBDA_2_TRUE:.6f}")

print("\nAdam-only (20000)")
print(f"lambda_1 = {adam_only_lambda_1.item():.6f}")
print(f"lambda_2 = {adam_only_lambda_2.item():.6f}")
print(f"Total Loss = {adam_total:.3e}")
print(f"Data Loss  = {adam_data:.3e}")
print(f"PDE Loss   = {adam_pde:.3e}")

print("\nAdam -> L-BFGS")
print(f"lambda_1 = {lbfgs_lambda_1.item():.6f}")
print(f"lambda_2 = {lbfgs_lambda_2.item():.6f}")
print(f"Total Loss = {lbfgs_total:.3e}")
print(f"Data Loss  = {lbfgs_data:.3e}")
print(f"PDE Loss   = {lbfgs_pde:.3e}")
print(f"L-BFGS closure calls = {len(lbfgs_history['loss'])}")

adam_err_l1 = abs(adam_only_lambda_1.item() - LAMBDA_1_TRUE) / LAMBDA_1_TRUE
adam_err_l2 = abs(adam_only_lambda_2.item() - LAMBDA_2_TRUE) / LAMBDA_2_TRUE

lbfgs_err_l1 = abs(lbfgs_lambda_1.item() - LAMBDA_1_TRUE) / LAMBDA_1_TRUE
lbfgs_err_l2 = abs(lbfgs_lambda_2.item() - LAMBDA_2_TRUE) / LAMBDA_2_TRUE

print("\nRelative Parameter Error")
print(f"Adam-only lambda_1: {adam_err_l1 * 100:.2f}%")
print(f"Adam-only lambda_2: {adam_err_l2 * 100:.2f}%")
print(f"Adam->L-BFGS lambda_1: {lbfgs_err_l1 * 100:.2f}%")
print(f"Adam->L-BFGS lambda_2: {lbfgs_err_l2 * 100:.2f}%")


# ============================================================
# 11. 对比图 1：Loss 优化轨迹
# ============================================================

adam_stage1_steps = np.arange(len(adam_history["total"]))
adam_only_steps = np.arange(
    ADAM_STAGE1,
    ADAM_STAGE1 + len(adam_only_history["total"])
)

# L-BFGS 横轴表示 closure evaluation 的顺序，
# 只是用于展示优化轨迹，不与 Adam epoch 严格等价。
lbfgs_steps = np.arange(
    ADAM_STAGE1,
    ADAM_STAGE1 + len(lbfgs_history["loss"])
)

plt.figure(figsize=(10, 5))

plt.semilogy(
    adam_stage1_steps,
    adam_history["total"],
    label="Adam: shared stage"
)

plt.semilogy(
    adam_only_steps,
    adam_only_history["total"],
    label="Adam-only continuation"
)

plt.semilogy(
    lbfgs_steps,
    lbfgs_history["loss"],
    label="L-BFGS continuation"
)

plt.axvline(
    ADAM_STAGE1,
    linestyle="--",
    label="Optimizer switch"
)

# 避免 L-BFGS line search 中极端 trial point 将纵轴完全拉爆
plt.ylim(1e-5, 1)

plt.xlabel("Optimization progress")
plt.ylabel("Total Loss")
plt.title("Burgers: Adam-only vs Adam -> L-BFGS")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "burgers_loss_comparison.png",
    dpi=200
)
plt.show()


# ============================================================
# 12. 对比图 2：参数相对误差
# ============================================================

methods = ["Adam-only", "Adam -> L-BFGS"]

lambda_1_errors = [
    adam_err_l1 * 100,
    lbfgs_err_l1 * 100
]

lambda_2_errors = [
    adam_err_l2 * 100,
    lbfgs_err_l2 * 100
]

x_pos = np.arange(len(methods))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    x_pos - width / 2,
    lambda_1_errors,
    width,
    label="lambda_1 error"
)

plt.bar(
    x_pos + width / 2,
    lambda_2_errors,
    width,
    label="lambda_2 error"
)

plt.xticks(x_pos, methods)
plt.ylabel("Relative Error (%)")
plt.title("Burgers Parameter Identification Error")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "burgers_parameter_error.png",
    dpi=200
)
plt.show()

print(f"\nFigures saved to: {RESULT_DIR.resolve()}")
