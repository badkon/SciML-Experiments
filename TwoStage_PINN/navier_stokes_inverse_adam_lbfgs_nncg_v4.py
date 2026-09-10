# -*- coding: utf-8 -*-
"""
Navier-Stokes 逆问题：
Adam-only vs Adam -> L-BFGS vs Adam -> L-BFGS -> NNCG

数据
----
默认使用 Raissi PINN 仓库中的：
    ./data/cylinder_nektar_wake.mat

逆问题目标
----------
从稀疏的速度观测 (u, v) 中同时学习流场与 Navier-Stokes 方程参数：

    u_t + lambda1 (u u_x + v u_y) + p_x
        - lambda2 (u_xx + u_yy) = 0

    v_t + lambda1 (u v_x + v v_y) + p_y
        - lambda2 (v_xx + v_yy) = 0

真实参数：
    lambda1 = 1.0
    lambda2 = 0.01

网络输入：
    (x, y, t)

网络输出：
    (psi, p)

通过流函数 psi 自动满足不可压缩连续性方程：
    u = psi_y
    v = -psi_x

实验设计
--------
1. 共同 Adam 训练 5000 轮；
2. 从同一个 Adam-5000 checkpoint 分叉：
   A. 继续 Adam 15000 轮，总计 20000；
   B. 切换到 L-BFGS，主优化阶段 15000 次内部迭代上限；
3. 从 B 的 L-BFGS 最终点再分成两个公平对照：
   C1. 继续 L-BFGS，额外运行时间尽量匹配 NNCG；
   C2. 切换 NNCG 200 步；
4. 比较 lambda1 / lambda2 参数识别误差、Loss 和运行时间。

注意
----
- NNCG 很重，默认只跑 60 步。
- 为了适合本地 RTX 5060 首轮验证，默认训练点 N_TRAIN=5000。
- 如果显存不足，可先把 N_TRAIN 改为 3000。
"""

from pathlib import Path
from functools import reduce
import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torch.optim import Optimizer


# ============================================================
# 1. 全局配置
# ============================================================

SEED = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cylinder_nektar_wake.mat"
RESULT_DIR = BASE_DIR / "results" / "navier_stokes_inverse"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Raissi 原始逆问题常用 5000 个速度观测点
N_TRAIN = 5000

ADAM_STAGE1 = 5000
ADAM_CONTINUE = 15000
ADAM_LR = 1e-3

# 输入 (x,y,t)，输出 (psi,p)
LAYERS = [3, 64, 64, 64, 64, 2]

LAMBDA1_TRUE = 1.0
LAMBDA2_TRUE = 0.01

# ---------------- L-BFGS ----------------
LBFGS_MAX_ITER = 15000

# ---------------- NNCG ----------------
# Allen-Cahn 实验表明主要收益集中在前几十步，因此这里先用 60 步。
NNCG_STEPS = 200
NNCG_LR = 1.0
NNCG_RANK = 20
NNCG_MU = 1e-3
NNCG_CG_TOL = 1e-6
NNCG_CG_MAX_ITERS = 200
NNCG_PRECOND_UPDATE_FREQ = 20
NNCG_LINE_SEARCH = "armijo"

# ---------------- 公平对照：继续 L-BFGS ----------------
# 从完全相同的 L-BFGS checkpoint 出发，
# 让“继续 L-BFGS”的额外运行时间尽量匹配 NNCG 的运行时间。
# 每次只跑一小段 L-BFGS，便于在接近目标时间时停止。
LBFGS_CONTINUE_CHUNK = 50


def set_seed(seed=SEED):
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
# 2. 读取 Raissi cylinder wake 数据
# ============================================================

data = loadmat(DATA_PATH)

required_keys = ["X_star", "U_star", "p_star", "t"]
for key in required_keys:
    if key not in data:
        raise KeyError(
            f"数据中缺少 '{key}'。当前 keys = "
            f"{[k for k in data.keys() if not k.startswith('__')]}"
        )

X_star_space = data["X_star"]          # (N, 2)
U_star = data["U_star"]                # (N, 2, T)
p_star = data["p_star"]                # (N, T)
t_star = data["t"].flatten()           # (T,)

N = X_star_space.shape[0]
Tn = t_star.shape[0]

print("X_star:", X_star_space.shape)
print("U_star:", U_star.shape)
print("p_star:", p_star.shape)
print("t:", t_star.shape)

# 构造全部 (x,y,t,u,v,p) 时空数据
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
print("Total spatiotemporal points:", n_total)

if N_TRAIN > n_total:
    raise ValueError("N_TRAIN 大于数据总点数。")

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
# 3. PINN 网络
# ============================================================

class NavierStokesPINN(nn.Module):
    """
    输入 (x,y,t)
    输出 (psi,p)

    lambda1 / lambda2 作为可训练物理参数，与网络权重共同优化。
    """

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

        # Raissi 逆问题中这两个系数需要从数据识别
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


# ============================================================
# 4. 自动微分工具
# ============================================================

def grad(outputs, inputs):
    """对 inputs 求一阶导数，保留计算图供高阶导继续使用。"""
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]


# ============================================================
# 5. Navier-Stokes residual
# ============================================================

def navier_stokes_terms(model, x, y, t):
    """
    由 psi 得到：
        u = psi_y
        v = -psi_x

    再构造二维不可压缩 Navier-Stokes 动量方程残差。
    """

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
    """
    数据项：速度 u,v
    物理项：两个 Navier-Stokes 动量方程 residual
    """

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


def parameter_metrics(model):
    lambda1 = model.lambda1.detach().item()
    lambda2 = model.lambda2.detach().item()

    err1 = (
        abs(lambda1 - LAMBDA1_TRUE)
        / abs(LAMBDA1_TRUE)
        * 100.0
    )

    err2 = (
        abs(lambda2 - LAMBDA2_TRUE)
        / abs(LAMBDA2_TRUE)
        * 100.0
    )

    return lambda1, lambda2, err1, err2


def evaluate_model(model):
    loss, loss_data, loss_pde = losses_ns(model)
    l1, l2, e1, e2 = parameter_metrics(model)

    return (
        loss.detach().item(),
        loss_data.detach().item(),
        loss_pde.detach().item(),
        l1,
        l2,
        e1,
        e2
    )


# ============================================================
# 6. NysNewton-CG
# ============================================================

def _apply_nys_precond_inv(U, S_mu_inv, mu, lambda_r, vec):
    z = U.T @ vec

    return (
        (lambda_r + mu) * (U @ (S_mu_inv * z))
        + (vec - U @ z)
    )


def _nystrom_pcg(
    hvp_fn,
    b,
    x0,
    mu,
    U,
    S,
    rank,
    tol,
    max_iters
):
    """
    用 Nyström 预条件共轭梯度近似求：
        (H + mu I) d = g
    """

    lambda_r = S[rank - 1]
    S_mu_inv = 1.0 / (S + mu)

    x = x0.clone()
    residual = b - (hvp_fn(x) + mu * x)

    with torch.no_grad():
        z = _apply_nys_precond_inv(
            U,
            S_mu_inv,
            mu,
            lambda_r,
            residual
        )
        p = z.clone()

    i = 0

    while (
        torch.norm(residual) > tol
        and i < max_iters
    ):
        v = hvp_fn(p) + mu * p

        with torch.no_grad():
            rz = torch.dot(residual, z)
            denom = torch.dot(p, v)

            if torch.abs(denom) < 1e-30:
                break

            alpha = rz / denom

            x = x + alpha * p
            residual_new = residual - alpha * v

            z_new = _apply_nys_precond_inv(
                U,
                S_mu_inv,
                mu,
                lambda_r,
                residual_new
            )

            beta = (
                torch.dot(residual_new, z_new)
                /
                (rz + 1e-30)
            )

            p = z_new + beta * p

            residual = residual_new
            z = z_new

        i += 1

    return x, i, torch.norm(residual).item()


class NysNewtonCG(Optimizer):
    """简化移植版 NysNewton-CG。"""

    def __init__(
        self,
        params,
        lr=1.0,
        rank=20,
        mu=1e-3,
        cg_tol=1e-6,
        cg_max_iters=200,
        line_search_fn="armijo"
    ):
        params = list(params)

        defaults = dict(
            lr=lr,
            rank=rank,
            mu=mu,
            cg_tol=cg_tol,
            cg_max_iters=cg_max_iters,
            line_search_fn=line_search_fn
        )

        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("NNCG 只支持一个 parameter group。")

        self._params = self.param_groups[0]["params"]

        self.rank = rank
        self.mu = mu
        self.cg_tol = cg_tol
        self.cg_max_iters = cg_max_iters
        self.line_search_fn = line_search_fn

        self.U = None
        self.S = None
        self._numel_cache = None

        self.old_dir = torch.zeros(
            self._numel(),
            device=self._params[0].device,
            dtype=self._params[0].dtype
        )

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(),
                self._params,
                0
            )
        return self._numel_cache

    def _flatten_grad_tuple(self, grad_tuple):
        return torch.cat([
            g.reshape(-1)
            for g in grad_tuple
            if g is not None
        ])

    def _hvp(self, flat_grad, vector):
        hv = torch.autograd.grad(
            flat_grad,
            self._params,
            grad_outputs=vector,
            retain_graph=True,
            allow_unused=False
        )

        return torch.cat([
            h.detach().reshape(-1)
            for h in hv
        ])

    def update_preconditioner(self, grad_tuple):
        """构造当前 Hessian 的随机 Nyström 低秩近似。"""

        flat_grad = self._flatten_grad_tuple(
            grad_tuple
        )

        p = flat_grad.numel()
        rank = min(self.rank, p)
        self.rank = rank

        Phi = torch.randn(
            rank,
            p,
            device=flat_grad.device,
            dtype=flat_grad.dtype
        ) / np.sqrt(p)

        Phi = torch.linalg.qr(
            Phi.T,
            mode="reduced"
        )[0].T

        # 逐个 HVP，降低峰值显存
        Y_rows = []

        for j in range(rank):
            Y_rows.append(
                self._hvp(
                    flat_grad,
                    Phi[j]
                )
            )

        Y = torch.stack(
            Y_rows,
            dim=0
        )

        shift = torch.finfo(Y.dtype).eps

        Y_shifted = (
            Y
            + shift * Phi
        )

        target = (
            Y_shifted
            @ Phi.T
        )

        target = (
            0.5
            * (target + target.T)
        )

        eig_min = torch.linalg.eigvalsh(
            target
        ).min()

        if eig_min <= 0:
            extra_shift = (
                -eig_min
                + 10.0 * shift
            )

            target = (
                target
                + extra_shift
                * torch.eye(
                    rank,
                    device=target.device,
                    dtype=target.dtype
                )
            )

            shift = (
                shift
                + extra_shift
            )

        C = torch.linalg.cholesky(
            target
        )

        B = torch.linalg.solve_triangular(
            C,
            Y_shifted,
            upper=False,
            left=True
        )

        _, singular_values, Vh = (
            torch.linalg.svd(
                B,
                full_matrices=False
            )
        )

        self.U = Vh.T

        self.S = torch.clamp(
            singular_values.square()
            - shift,
            min=0.0
        )

    def _clone_params(self):
        return [
            p.detach().clone()
            for p in self._params
        ]

    def _set_params(self, params_data):
        with torch.no_grad():
            for p, pdata in zip(
                self._params,
                params_data
            ):
                p.copy_(pdata)

    def _add_flat_direction(
        self,
        alpha,
        direction
    ):
        offset = 0

        with torch.no_grad():
            for p in self._params:
                n = p.numel()

                p.add_(
                    direction[
                        offset:offset+n
                    ].view_as(p),
                    alpha=alpha
                )

                offset += n

    def step(self, closure):

        if self.U is None:
            raise RuntimeError(
                "先调用 update_preconditioner()。"
            )

        with torch.enable_grad():
            loss, grad_tuple = closure()

        flat_grad = self._flatten_grad_tuple(
            grad_tuple
        )

        def hvp_fn(v):
            return self._hvp(
                flat_grad,
                v
            )

        direction, cg_iters, cg_residual = (
            _nystrom_pcg(
                hvp_fn,
                flat_grad.detach(),
                self.old_dir.detach(),
                self.mu,
                self.U,
                self.S,
                self.rank,
                self.cg_tol,
                self.cg_max_iters
            )
        )

        self.old_dir = direction.detach()

        descent_direction = (
            -direction.detach()
        )

        directional_derivative = (
            torch.dot(
                flat_grad.detach(),
                descent_direction
            )
        )

        fallback = False

        if directional_derivative >= 0:
            fallback = True

            descent_direction = (
                -flat_grad.detach()
            )

            directional_derivative = (
                -torch.dot(
                    flat_grad.detach(),
                    flat_grad.detach()
                )
            )

        step_size = (
            self.param_groups[0]["lr"]
        )

        if self.line_search_fn == "armijo":

            params_init = (
                self._clone_params()
            )

            loss0 = loss.detach().item()

            c1 = 0.1
            beta = 0.5

            for _ in range(20):

                self._set_params(
                    params_init
                )

                self._add_flat_direction(
                    step_size,
                    descent_direction
                )

                with torch.enable_grad():
                    trial_loss, _ = closure()

                rhs = (
                    loss0
                    + c1
                    * step_size
                    * directional_derivative.item()
                )

                if (
                    trial_loss.detach().item()
                    <= rhs
                ):
                    break

                step_size *= beta

            self._set_params(
                params_init
            )

        self._add_flat_direction(
            step_size,
            descent_direction
        )

        return (
            loss.detach().item(),
            flat_grad.detach(),
            cg_iters,
            cg_residual,
            step_size,
            fallback
        )


# ============================================================
# 7. Stage 1：共同 Adam 0 -> 5000
# ============================================================

base_model = NavierStokesPINN(
    LAYERS
).to(device)

optimizer_adam = torch.optim.Adam(
    base_model.parameters(),
    lr=ADAM_LR
)

adam_history = []
lambda1_history_adam = []
lambda2_history_adam = []

print(
    "\n========== Stage 1: "
    "Adam 0 -> 5000 =========="
)

t0 = time.perf_counter()

for epoch in range(
    ADAM_STAGE1
):
    optimizer_adam.zero_grad()

    loss, loss_data, loss_pde = (
        losses_ns(base_model)
    )

    loss.backward()
    optimizer_adam.step()

    adam_history.append(
        loss.item()
    )

    lambda1_history_adam.append(
        base_model.lambda1.item()
    )

    lambda2_history_adam.append(
        base_model.lambda2.item()
    )

    if epoch % 1000 == 0:
        print(
            f"Adam | Epoch {epoch:5d} | "
            f"Loss: {loss.item():.3e} | "
            f"Data: {loss_data.item():.3e} | "
            f"PDE: {loss_pde.item():.3e} | "
            f"lambda1: {base_model.lambda1.item():.5f} | "
            f"lambda2: {base_model.lambda2.item():.5f}"
        )

adam_stage_time = (
    time.perf_counter() - t0
)

torch.save(
    {
        "model_state_dict": base_model.state_dict(),
        "epoch": ADAM_STAGE1,
        "lambda1": base_model.lambda1.detach().item(),
        "lambda2": base_model.lambda2.detach().item(),
    },
    CHECKPOINT_DIR / "navier_stokes_adam_stage1.pt"
)


# ============================================================
# 8. 从同一个 Adam-5000 checkpoint 分叉
# ============================================================

adam_state = copy.deepcopy(
    base_model.state_dict()
)

# A：继续 Adam
adam_only_model = (
    NavierStokesPINN(LAYERS)
    .to(device)
)

adam_only_model.load_state_dict(
    adam_state
)

# B：切换 L-BFGS
lbfgs_model = (
    NavierStokesPINN(LAYERS)
    .to(device)
)

lbfgs_model.load_state_dict(
    adam_state
)


# ============================================================
# 9. Branch A：Adam-only 5000 -> 20000
# ============================================================

optimizer_adam_only = torch.optim.Adam(
    adam_only_model.parameters(),
    lr=ADAM_LR
)

adam_only_history = []
adam_only_lambda1 = []
adam_only_lambda2 = []

print(
    "\n========== Branch A: "
    "Adam-only 5000 -> 20000 =========="
)

t0 = time.perf_counter()

for epoch in range(
    ADAM_CONTINUE
):
    optimizer_adam_only.zero_grad()

    loss, _, _ = losses_ns(
        adam_only_model
    )

    loss.backward()
    optimizer_adam_only.step()

    adam_only_history.append(
        loss.item()
    )

    adam_only_lambda1.append(
        adam_only_model.lambda1.item()
    )

    adam_only_lambda2.append(
        adam_only_model.lambda2.item()
    )

    if epoch % 1000 == 0:
        print(
            f"Adam-only | "
            f"Epoch {epoch + ADAM_STAGE1:5d} | "
            f"Loss: {loss.item():.3e} | "
            f"lambda1: {adam_only_model.lambda1.item():.5f} | "
            f"lambda2: {adam_only_model.lambda2.item():.5f}"
        )

adam_continue_time = (
    time.perf_counter() - t0
)


# ============================================================
# 10. Branch B：Adam -> L-BFGS
# ============================================================

lbfgs_history = []
lbfgs_lambda1 = []
lbfgs_lambda2 = []

optimizer_lbfgs = torch.optim.LBFGS(
    lbfgs_model.parameters(),
    lr=1.0,
    max_iter=LBFGS_MAX_ITER,
    max_eval=LBFGS_MAX_ITER,
    tolerance_grad=1e-9,
    tolerance_change=1e-12,
    history_size=100,
    line_search_fn="strong_wolfe"
)


def closure_lbfgs():

    optimizer_lbfgs.zero_grad()

    loss, _, _ = losses_ns(
        lbfgs_model
    )

    loss.backward()

    lbfgs_history.append(
        loss.item()
    )

    lbfgs_lambda1.append(
        lbfgs_model.lambda1.item()
    )

    lbfgs_lambda2.append(
        lbfgs_model.lambda2.item()
    )

    return loss


print(
    "\n========== Branch B: "
    "Adam 15000 -> L-BFGS =========="
)

t0 = time.perf_counter()

optimizer_lbfgs.step(
    closure_lbfgs
)

lbfgs_time = (
    time.perf_counter() - t0
)

lbfgs_final_state = copy.deepcopy(
    lbfgs_model.state_dict()
)

torch.save(
    {
        "model_state_dict": lbfgs_model.state_dict(),
        "lambda1": lbfgs_model.lambda1.detach().item(),
        "lambda2": lbfgs_model.lambda2.detach().item(),
        "lbfgs_closure_calls": len(lbfgs_history),
    },
    CHECKPOINT_DIR / "navier_stokes_adam_lbfgs.pt"
)


# ============================================================
# 11. Branch C：Adam -> L-BFGS -> NNCG
# ============================================================

# Branch C1：从相同 L-BFGS checkpoint 继续使用 L-BFGS
lbfgs_continue_model = (
    NavierStokesPINN(LAYERS)
    .to(device)
)
lbfgs_continue_model.load_state_dict(
    lbfgs_final_state
)

# Branch C2：从相同 L-BFGS checkpoint 切换到 NNCG
nncg_model = (
    NavierStokesPINN(LAYERS)
    .to(device)
)

nncg_model.load_state_dict(
    lbfgs_final_state
)

optimizer_nncg = NysNewtonCG(
    nncg_model.parameters(),
    lr=NNCG_LR,
    rank=NNCG_RANK,
    mu=NNCG_MU,
    cg_tol=NNCG_CG_TOL,
    cg_max_iters=NNCG_CG_MAX_ITERS,
    line_search_fn=NNCG_LINE_SEARCH
)

nncg_history = []
nncg_lambda1 = []
nncg_lambda2 = []

fallback_count = 0

print(
    "\n========== Branch C: "
    "Adam -> L-BFGS -> NNCG =========="
)

print(
    f"NNCG | steps={NNCG_STEPS} | "
    f"rank={NNCG_RANK} | "
    f"mu={NNCG_MU:.1e} | "
    f"cg_max_iters={NNCG_CG_MAX_ITERS}"
)

t0 = time.perf_counter()

for step in range(
    NNCG_STEPS
):

    def closure_nncg():

        optimizer_nncg.zero_grad(
            set_to_none=True
        )

        loss, _, _ = losses_ns(
            nncg_model
        )

        params = tuple(nncg_model.parameters())

        raw_grads = torch.autograd.grad(
            loss,
            params,
            create_graph=True,
            allow_unused=True
        )

        # Navier-Stokes 的流函数/压力只通过导数进入目标，
        # 某些常数偏置可能天然不参与当前计算图。
        # 用 p*0.0 保持二阶计算图连通，而不是简单丢弃这些参数。
        grad_tuple = tuple(
            g if g is not None else p * 0.0
            for g, p in zip(raw_grads, params)
        )

        return loss, grad_tuple

    # 第 0 步以及每隔固定频率更新 Nyström preconditioner
    if (
        step
        % NNCG_PRECOND_UPDATE_FREQ
        == 0
    ):
        optimizer_nncg.zero_grad(
            set_to_none=True
        )

        loss_pre, _, _ = losses_ns(
            nncg_model
        )

        params = tuple(nncg_model.parameters())

        raw_grads_pre = torch.autograd.grad(
            loss_pre,
            params,
            create_graph=True,
            allow_unused=True
        )

        grad_pre = tuple(
            g if g is not None else p * 0.0
            for g, p in zip(raw_grads_pre, params)
        )

        print(
            f"\nNNCG | Step {step:3d} | "
            "updating preconditioner..."
        )

        optimizer_nncg.update_preconditioner(
            grad_pre
        )

        del grad_pre
        del loss_pre

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (
        _,
        flat_grad,
        cg_iters,
        cg_res,
        step_size,
        fallback
    ) = optimizer_nncg.step(
        closure_nncg
    )

    if fallback:
        fallback_count += 1

    loss_after, _, _ = losses_ns(
        nncg_model
    )

    nncg_history.append(
        loss_after.detach().item()
    )

    nncg_lambda1.append(
        nncg_model.lambda1.item()
    )

    nncg_lambda2.append(
        nncg_model.lambda2.item()
    )

    if (
        step % 5 == 0
        or step == NNCG_STEPS - 1
    ):
        print(
            f"NNCG | Step {step:3d} | "
            f"Loss: {loss_after.item():.3e} | "
            f"|g|: {torch.norm(flat_grad).item():.3e} | "
            f"CG: {cg_iters:3d} | "
            f"CG-res: {cg_res:.3e} | "
            f"step: {step_size:.3e} | "
            f"lambda1: {nncg_model.lambda1.item():.5f} | "
            f"lambda2: {nncg_model.lambda2.item():.5f}"
        )

nncg_time = (
    time.perf_counter() - t0
)

torch.save(
    {
        "model_state_dict": nncg_model.state_dict(),
        "lambda1": nncg_model.lambda1.detach().item(),
        "lambda2": nncg_model.lambda2.detach().item(),
        "fallback_count": fallback_count,
        "nncg_steps": NNCG_STEPS,
        "runtime_seconds": nncg_time,
    },
    CHECKPOINT_DIR / "navier_stokes_adam_lbfgs_nncg.pt"
)


# ============================================================
# 12. Branch C1：继续 L-BFGS（与 NNCG 等 wall-clock）
# ============================================================

lbfgs_continue_history = []
lbfgs_continue_lambda1 = []
lbfgs_continue_lambda2 = []
lbfgs_continue_closure_calls = 0

print(
    "\n========== Branch C1: "
    "L-BFGS continuation with matched wall-clock =========="
)
print(
    f"Target extra runtime ~= NNCG runtime = {nncg_time:.1f} s"
)

t0 = time.perf_counter()

# 通过多个小 chunk 继续 L-BFGS，直到达到 NNCG 的运行时间预算。
while True:
    elapsed = time.perf_counter() - t0
    if elapsed >= nncg_time:
        break

    optimizer_lbfgs_continue = torch.optim.LBFGS(
        lbfgs_continue_model.parameters(),
        lr=1.0,
        max_iter=LBFGS_CONTINUE_CHUNK,
        max_eval=LBFGS_CONTINUE_CHUNK,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe"
    )

    def closure_lbfgs_continue():
        global lbfgs_continue_closure_calls

        optimizer_lbfgs_continue.zero_grad()

        loss, _, _ = losses_ns(
            lbfgs_continue_model
        )

        loss.backward()

        lbfgs_continue_history.append(
            loss.item()
        )
        lbfgs_continue_lambda1.append(
            lbfgs_continue_model.lambda1.item()
        )
        lbfgs_continue_lambda2.append(
            lbfgs_continue_model.lambda2.item()
        )

        lbfgs_continue_closure_calls += 1

        return loss

    optimizer_lbfgs_continue.step(
        closure_lbfgs_continue
    )

    # 每个 chunk 后输出一次状态
    current_loss, _, _ = losses_ns(
        lbfgs_continue_model
    )
    current_l1, current_l2, _, _ = parameter_metrics(
        lbfgs_continue_model
    )

    elapsed = time.perf_counter() - t0
    print(
        f"L-BFGS continue | "
        f"time: {elapsed:7.1f}/{nncg_time:.1f} s | "
        f"closure calls: {lbfgs_continue_closure_calls:6d} | "
        f"Loss: {current_loss.item():.3e} | "
        f"lambda1: {current_l1:.6f} | "
        f"lambda2: {current_l2:.6f}"
    )

lbfgs_continue_time = time.perf_counter() - t0

torch.save(
    {
        "model_state_dict": lbfgs_continue_model.state_dict(),
        "lambda1": lbfgs_continue_model.lambda1.detach().item(),
        "lambda2": lbfgs_continue_model.lambda2.detach().item(),
        "closure_calls": lbfgs_continue_closure_calls,
        "runtime_seconds": lbfgs_continue_time,
    },
    CHECKPOINT_DIR / "navier_stokes_adam_lbfgs_continue_matched_time.pt"
)


# ============================================================
# 13. 最终结果
# ============================================================

adam_result = evaluate_model(
    adam_only_model
)

lbfgs_result = evaluate_model(
    lbfgs_model
)

nncg_result = evaluate_model(
    nncg_model
)

lbfgs_continue_result = evaluate_model(
    lbfgs_continue_model
)

print(
    "\n========== Final Results =========="
)

print(
    f"L-BFGS closure calls: "
    f"{len(lbfgs_history)}"
)

print(
    f"NNCG fallback count: "
    f"{fallback_count}/{NNCG_STEPS}"
)

print(
    f"Matched-time L-BFGS extra closure calls: "
    f"{lbfgs_continue_closure_calls}"
)

names = [
    "Adam-only (20000)",
    "Adam -> L-BFGS",
    "Adam -> L-BFGS -> L-BFGS (matched time)",
    "Adam -> L-BFGS -> NNCG"
]

results = [
    adam_result,
    lbfgs_result,
    lbfgs_continue_result,
    nncg_result
]

for name, result in zip(
    names,
    results
):
    (
        total,
        data_l,
        pde_l,
        l1,
        l2,
        e1,
        e2
    ) = result

    print(f"\n{name}")
    print(
        f"Total Loss    = {total:.3e}"
    )
    print(
        f"Data Loss     = {data_l:.3e}"
    )
    print(
        f"PDE Loss      = {pde_l:.3e}"
    )
    print(
        f"lambda1       = {l1:.6f}"
    )
    print(
        f"lambda1 error = {e1:.2f}%"
    )
    print(
        f"lambda2       = {l2:.6f}"
    )
    print(
        f"lambda2 error = {e2:.2f}%"
    )

print(
    "\n========== Runtime =========="
)

print(
    f"Shared Adam stage : "
    f"{adam_stage_time:.1f} s"
)

print(
    f"Adam continuation : "
    f"{adam_continue_time:.1f} s"
)

print(
    f"L-BFGS stage      : "
    f"{lbfgs_time:.1f} s"
)

print(
    f"NNCG stage        : "
    f"{nncg_time:.1f} s"
)

print(
    f"L-BFGS matched    : "
    f"{lbfgs_continue_time:.1f} s"
)


# ============================================================
# 14. 图 1：Loss 对比
# ============================================================

adam_stage_steps = np.arange(
    len(adam_history)
)

adam_only_steps = np.arange(
    ADAM_STAGE1,
    ADAM_STAGE1
    + len(adam_only_history)
)

lbfgs_steps = np.arange(
    ADAM_STAGE1,
    ADAM_STAGE1
    + len(lbfgs_history)
)

nncg_start = (
    ADAM_STAGE1
    + len(lbfgs_history)
)

nncg_steps_plot = np.arange(
    nncg_start,
    nncg_start
    + len(nncg_history)
)

lbfgs_continue_steps_plot = np.arange(
    nncg_start,
    nncg_start
    + len(lbfgs_continue_history)
)

plt.figure(
    figsize=(11, 5.5)
)

plt.semilogy(
    adam_stage_steps,
    adam_history,
    label="Adam: shared stage"
)

plt.semilogy(
    adam_only_steps,
    adam_only_history,
    label="Adam-only continuation"
)

plt.semilogy(
    lbfgs_steps,
    lbfgs_history,
    label="L-BFGS continuation"
)

plt.semilogy(
    nncg_steps_plot,
    nncg_history,
    label="NNCG fine-tuning"
)

plt.semilogy(
    lbfgs_continue_steps_plot,
    lbfgs_continue_history,
    label="L-BFGS continuation (matched time)"
)

plt.axvline(
    ADAM_STAGE1,
    linestyle="--",
    label="Adam -> L-BFGS"
)

plt.axvline(
    nncg_start,
    linestyle=":",
    label="L-BFGS -> NNCG"
)

plt.xlabel(
    "Optimization progress"
)
plt.ylabel(
    "Total Loss"
)
plt.title(
    "Navier-Stokes Inverse Problem: Optimizer Comparison"
)
plt.legend()
plt.grid(
    alpha=0.3
)
plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "navier_stokes_loss_comparison.png",
    dpi=200
)

plt.show()


# ============================================================
# 15. 图 2：lambda1 / lambda2 参数识别结果（分开显示）
# ============================================================

method_labels = [
    "Adam-only",
    "Adam -> L-BFGS",
    "Adam -> L-BFGS\n-> L-BFGS\n(matched time)",
    "Adam -> L-BFGS\n-> NNCG"
]

lambda1_values = [
    adam_result[3],
    lbfgs_result[3],
    lbfgs_continue_result[3],
    nncg_result[3]
]

lambda2_values = [
    adam_result[4],
    lbfgs_result[4],
    lbfgs_continue_result[4],
    nncg_result[4]
]

x_pos = np.arange(
    len(method_labels)
)
width = 0.35

# lambda1 单独作图，并围绕真值 1.0 自动缩放
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

lambda1_min = min(
    min(lambda1_values),
    LAMBDA1_TRUE
)
lambda1_max = max(
    max(lambda1_values),
    LAMBDA1_TRUE
)

lambda1_pad = max(
    (lambda1_max - lambda1_min) * 0.25,
    5e-4
)

plt.ylim(
    lambda1_min - lambda1_pad,
    lambda1_max + lambda1_pad
)

plt.xticks(
    x_pos,
    method_labels
)

plt.ylabel(
    "Identified lambda1"
)

plt.title(
    "Navier-Stokes Parameter Identification: lambda1"
)

plt.legend()
plt.grid(
    axis="y",
    alpha=0.3
)
plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "navier_stokes_lambda1_values.png",
    dpi=200
)

plt.show()


# lambda2 单独作图，并围绕真值 0.01 自动缩放
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

lambda2_min = min(
    min(lambda2_values),
    LAMBDA2_TRUE
)
lambda2_max = max(
    max(lambda2_values),
    LAMBDA2_TRUE
)

lambda2_pad = max(
    (lambda2_max - lambda2_min) * 0.25,
    2e-4
)

plt.ylim(
    lambda2_min - lambda2_pad,
    lambda2_max + lambda2_pad
)

plt.xticks(
    x_pos,
    method_labels
)

plt.ylabel(
    "Identified lambda2"
)

plt.title(
    "Navier-Stokes Parameter Identification: lambda2"
)

plt.legend()
plt.grid(
    axis="y",
    alpha=0.3
)
plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "navier_stokes_lambda2_values.png",
    dpi=200
)

plt.show()


# ============================================================
# 16. 图 3：参数相对误差
# ============================================================

lambda1_errors = [
    adam_result[5],
    lbfgs_result[5],
    lbfgs_continue_result[5],
    nncg_result[5]
]

lambda2_errors = [
    adam_result[6],
    lbfgs_result[6],
    lbfgs_continue_result[6],
    nncg_result[6]
]

plt.figure(
    figsize=(9, 5.5)
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
    method_labels
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
    / "navier_stokes_parameter_error.png",
    dpi=200
)

plt.show()


# ============================================================
# 17. 图 4：lambda1 / lambda2 后期局部放大轨迹
# ============================================================

# Adam-only 后半段轨迹
adam_post_lambda1 = adam_only_lambda1
adam_post_lambda2 = adam_only_lambda2

# L-BFGS checkpoint 后的两个公平分支
lbfgs_matched_post_lambda1 = lbfgs_continue_lambda1
lbfgs_matched_post_lambda2 = lbfgs_continue_lambda2

nncg_post_lambda1 = nncg_lambda1
nncg_post_lambda2 = nncg_lambda2

# lambda1 后期放大
plt.figure(
    figsize=(10, 5.5)
)

plt.plot(
    np.arange(len(adam_post_lambda1)),
    adam_post_lambda1,
    label="Adam-only continuation"
)

plt.plot(
    np.arange(len(lbfgs_matched_post_lambda1)),
    lbfgs_matched_post_lambda1,
    label="L-BFGS continuation (matched time)"
)

plt.plot(
    np.arange(len(nncg_post_lambda1)),
    nncg_post_lambda1,
    label="NNCG fine-tuning"
)

plt.axhline(
    LAMBDA1_TRUE,
    linestyle="--",
    label="lambda1 true"
)

lambda1_zoom_vals = (
    adam_post_lambda1
    + lbfgs_matched_post_lambda1
    + nncg_post_lambda1
    + [LAMBDA1_TRUE]
)

if len(lambda1_zoom_vals) > 0:
    zmin = min(lambda1_zoom_vals)
    zmax = max(lambda1_zoom_vals)
    zpad = max(
        (zmax - zmin) * 0.15,
        2e-4
    )
    plt.ylim(
        zmin - zpad,
        zmax + zpad
    )

plt.xlabel(
    "Post-switch optimization progress"
)
plt.ylabel(
    "lambda1"
)
plt.title(
    "Navier-Stokes: lambda1 Late-stage Zoom"
)
plt.legend()
plt.grid(
    alpha=0.3
)
plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "navier_stokes_lambda1_late_zoom.png",
    dpi=200
)

plt.show()


# lambda2 后期放大
plt.figure(
    figsize=(10, 5.5)
)

plt.plot(
    np.arange(len(adam_post_lambda2)),
    adam_post_lambda2,
    label="Adam-only continuation"
)

plt.plot(
    np.arange(len(lbfgs_matched_post_lambda2)),
    lbfgs_matched_post_lambda2,
    label="L-BFGS continuation (matched time)"
)

plt.plot(
    np.arange(len(nncg_post_lambda2)),
    nncg_post_lambda2,
    label="NNCG fine-tuning"
)

plt.axhline(
    LAMBDA2_TRUE,
    linestyle="--",
    label="lambda2 true"
)

lambda2_zoom_vals = (
    adam_post_lambda2
    + lbfgs_matched_post_lambda2
    + nncg_post_lambda2
    + [LAMBDA2_TRUE]
)

if len(lambda2_zoom_vals) > 0:
    zmin = min(lambda2_zoom_vals)
    zmax = max(lambda2_zoom_vals)
    zpad = max(
        (zmax - zmin) * 0.15,
        1e-4
    )
    plt.ylim(
        zmin - zpad,
        zmax + zpad
    )

plt.xlabel(
    "Post-switch optimization progress"
)
plt.ylabel(
    "lambda2"
)
plt.title(
    "Navier-Stokes: lambda2 Late-stage Zoom"
)
plt.legend()
plt.grid(
    alpha=0.3
)
plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "navier_stokes_lambda2_late_zoom.png",
    dpi=200
)

plt.show()


# ============================================================
# 18. 图 5：L-BFGS checkpoint 后的 Loss 局部对比
# ============================================================

plt.figure(
    figsize=(10, 5.5)
)

plt.semilogy(
    np.arange(len(lbfgs_continue_history)),
    lbfgs_continue_history,
    label="Continue L-BFGS (matched time)"
)

plt.semilogy(
    np.arange(len(nncg_history)),
    nncg_history,
    label="Switch to NNCG"
)

plt.xlabel(
    "Post-L-BFGS optimization progress"
)
plt.ylabel(
    "Total Loss"
)
plt.title(
    "Navier-Stokes: Post-L-BFGS Fine-tuning Comparison"
)
plt.legend()
plt.grid(
    alpha=0.3
)
plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "navier_stokes_post_lbfgs_loss_zoom.png",
    dpi=200
)

plt.show()


# ============================================================
# 19. 图 6：参数完整训练轨迹
# ============================================================

# 把 Adam shared + Adam-only 拼起来
adam_full_lambda1 = (
    lambda1_history_adam
    + adam_only_lambda1
)

adam_full_lambda2 = (
    lambda2_history_adam
    + adam_only_lambda2
)

# 把 Adam shared + L-BFGS + NNCG 拼起来
hybrid_lambda1 = (
    lambda1_history_adam
    + lbfgs_lambda1
    + nncg_lambda1
)

hybrid_lambda2 = (
    lambda2_history_adam
    + lbfgs_lambda2
    + nncg_lambda2
)

lbfgs_matched_lambda1 = (
    lambda1_history_adam
    + lbfgs_lambda1
    + lbfgs_continue_lambda1
)

lbfgs_matched_lambda2 = (
    lambda2_history_adam
    + lbfgs_lambda2
    + lbfgs_continue_lambda2
)

plt.figure(
    figsize=(10, 5.5)
)

plt.plot(
    adam_full_lambda1,
    label="Adam-only lambda1"
)

plt.plot(
    hybrid_lambda1,
    label="Hybrid lambda1"
)

plt.plot(
    lbfgs_matched_lambda1,
    label="L-BFGS matched-time lambda1"
)

plt.axhline(
    LAMBDA1_TRUE,
    linestyle="--",
    label="lambda1 true"
)

plt.xlabel(
    "Optimization progress"
)
plt.ylabel(
    "lambda1"
)
plt.title(
    "Navier-Stokes: lambda1 Identification Trajectory"
)
plt.legend()
plt.grid(
    alpha=0.3
)
plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "navier_stokes_lambda1_trajectory.png",
    dpi=200
)

plt.show()


plt.figure(
    figsize=(10, 5.5)
)

plt.plot(
    adam_full_lambda2,
    label="Adam-only lambda2"
)

plt.plot(
    hybrid_lambda2,
    label="Hybrid lambda2"
)

plt.plot(
    lbfgs_matched_lambda2,
    label="L-BFGS matched-time lambda2"
)

plt.axhline(
    LAMBDA2_TRUE,
    linestyle="--",
    label="lambda2 true"
)

plt.xlabel(
    "Optimization progress"
)
plt.ylabel(
    "lambda2"
)
plt.title(
    "Navier-Stokes: lambda2 Identification Trajectory"
)
plt.legend()
plt.grid(
    alpha=0.3
)
plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "navier_stokes_lambda2_trajectory.png",
    dpi=200
)

plt.show()


print(
    f"\nFigures saved to: "
    f"{RESULT_DIR.resolve()}"
)

print(f"Checkpoints saved to: {CHECKPOINT_DIR.resolve()}")
