# -*- coding: utf-8 -*-
"""
Allen-Cahn 方程正问题：
Adam-only vs Adam -> L-BFGS vs Adam -> L-BFGS -> NNCG

在原有两阶段脚本基础上增加 NNCG 后期精调。
所有图片仍保存到：
    ./results/allen_cahn_two_stage/

NNCG 参考：
Rathore et al., Challenges in Training PINNs: A Loss Landscape Perspective, ICML 2024
官方实现：https://github.com/pratikrathore8/opt_for_pinns

注意：
- 前两阶段配置尽量保持原脚本不变。
- NNCG 默认采用本地验证用的较保守参数，避免一次把显存/时间拉得过高。
- 若本机运行稳定，可再逐步提高 NNCG_RANK 和 NNCG_STEPS。
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

# NNCG：先做小规模验证
NNCG_STEPS = 200
NNCG_LR = 1.0
NNCG_RANK = 20
NNCG_MU = 1e-4
NNCG_CG_TOL = 1e-6
NNCG_CG_MAX_ITERS = 100
NNCG_PRECOND_UPDATE_FREQ = 20
NNCG_LINE_SEARCH = "armijo"


def set_seed(seed: int = SEED):
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

x = data["x"].flatten()

if "tt" in data:
    t = data["tt"].flatten()
elif "t" in data:
    t = data["t"].flatten()
else:
    raise KeyError("AC.mat 中未找到时间变量 'tt' 或 't'。")

Exact = np.real(data["uu"])

X, T = np.meshgrid(x, t, indexing="ij")
X_star = np.hstack((X.reshape(-1, 1), T.reshape(-1, 1)))
u_star = Exact.reshape(-1, 1)

print("x shape:", x.shape)
print("t shape:", t.shape)
print("Exact shape:", Exact.shape)
print("X_star shape:", X_star.shape)


# ============================================================
# 3. 构造训练点
# ============================================================

idx_u = np.random.choice(X_star.shape[0], N_U, replace=False)
X_u = X_star[idx_u, :]
u_u = u_star[idx_u, :]

X_u_tensor = torch.tensor(X_u, dtype=torch.float32, device=device)
u_u_tensor = torch.tensor(u_u, dtype=torch.float32, device=device)

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
# 5. Allen-Cahn PDE residual 与 Loss
# ============================================================

def pde_residual_ac(model, X_input):
    """
    Allen-Cahn:
        u_t - epsilon*u_xx + 5u^3 - 5u = 0
    """
    u = model(X_input)

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

    return u_t - EPSILON * u_xx + 5.0 * u**3 - 5.0 * u


mse = nn.MSELoss()


def data_loss(model):
    return mse(model(X_u_tensor), u_u_tensor)


def physics_loss_ac(model):
    f = pde_residual_ac(model, X_f_tensor)
    return mse(f, torch.zeros_like(f))


def total_loss_ac(model):
    loss_u = data_loss(model)
    loss_f = physics_loss_ac(model)
    return loss_u + loss_f, loss_u, loss_f


def evaluate_ac(model):
    loss, loss_u, loss_f = total_loss_ac(model)

    X_star_tensor = torch.tensor(
        X_star,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        u_pred = model(X_star_tensor).cpu().numpy()

    relative_l2 = (
        np.linalg.norm(u_pred - u_star)
        / np.linalg.norm(u_star)
    )

    return loss.item(), loss_u.item(), loss_f.item(), relative_l2


# ============================================================
# 6. NysNewton-CG
# ============================================================

def _apply_nys_precond_inv(U, S_mu_inv, mu, lambda_r, vec):
    z = U.T @ vec
    return (lambda_r + mu) * (U @ (S_mu_inv * z)) + (vec - U @ z)


def _nystrom_pcg(hvp_fn, b, x0, mu, U, S, rank, tol, max_iters):
    lambda_r = S[rank - 1]
    S_mu_inv = 1.0 / (S + mu)

    x = x0.clone()
    residual = b - (hvp_fn(x) + mu * x)

    with torch.no_grad():
        z = _apply_nys_precond_inv(U, S_mu_inv, mu, lambda_r, residual)
        p = z.clone()

    i = 0
    while torch.norm(residual) > tol and i < max_iters:
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
                U, S_mu_inv, mu, lambda_r, residual_new
            )

            beta = torch.dot(residual_new, z_new) / (rz + 1e-30)

            p = z_new + beta * p
            residual = residual_new
            z = z_new

        i += 1

    return x, i, torch.norm(residual).item()


class NysNewtonCG(Optimizer):
    """
    Nyström-preconditioned damped Newton-CG。

    closure 必须返回：
        loss, grad_tuple

    grad_tuple 必须由：
        torch.autograd.grad(loss, params, create_graph=True)
    得到。
    """

    def __init__(
        self,
        params,
        lr=1.0,
        rank=20,
        mu=1e-4,
        cg_tol=1e-6,
        cg_max_iters=100,
        line_search_fn="armijo",
    ):
        params = list(params)
        defaults = dict(
            lr=lr,
            rank=rank,
            mu=mu,
            cg_tol=cg_tol,
            cg_max_iters=cg_max_iters,
            line_search_fn=line_search_fn,
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("NNCG 当前只支持单一 parameter group。")
        if line_search_fn not in (None, "armijo"):
            raise ValueError("NNCG 仅支持 Armijo line search 或 None。")

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

        self.rank = rank
        self.mu = mu
        self.cg_tol = cg_tol
        self.cg_max_iters = cg_max_iters
        self.line_search_fn = line_search_fn

        self.U = None
        self.S = None

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

    @staticmethod
    def _flatten_grad_tuple(grad_tuple):
        return torch.cat([
            g.reshape(-1) for g in grad_tuple if g is not None
        ])

    def _hvp(self, flat_grad, vector):
        hv = torch.autograd.grad(
            flat_grad,
            self._params,
            grad_outputs=vector,
            retain_graph=True,
            allow_unused=False
        )
        return torch.cat([h.detach().reshape(-1) for h in hv])

    def update_preconditioner(self, grad_tuple):
        """
        用随机 Nyström sketch 构造 Hessian 低秩近似。
        为节省显存，rank 个 HVP 串行计算。
        """
        flat_grad = self._flatten_grad_tuple(grad_tuple)
        p = flat_grad.numel()

        rank = min(self.rank, p)
        self.rank = rank

        Phi = torch.randn(
            rank,
            p,
            device=flat_grad.device,
            dtype=flat_grad.dtype
        ) / np.sqrt(p)

        Phi = torch.linalg.qr(Phi.T, mode="reduced")[0].T

        Y = torch.stack([
            self._hvp(flat_grad, Phi[j])
            for j in range(rank)
        ], dim=0)

        shift = torch.finfo(Y.dtype).eps
        Y_shifted = Y + shift * Phi

        target = Y_shifted @ Phi.T
        target = 0.5 * (target + target.T)

        eig_min = torch.linalg.eigvalsh(target).min()

        if eig_min <= 0:
            extra_shift = -eig_min + 10.0 * shift
            target = target + extra_shift * torch.eye(
                rank,
                device=target.device,
                dtype=target.dtype
            )
            shift = shift + extra_shift

        C = torch.linalg.cholesky(target)

        B = torch.linalg.solve_triangular(
            C,
            Y_shifted,
            upper=False,
            left=True
        )

        _, singular_values, Vh = torch.linalg.svd(
            B,
            full_matrices=False
        )

        self.U = Vh.T
        self.S = torch.clamp(
            singular_values.square() - shift,
            min=0.0
        )

    def _clone_params(self):
        return [p.detach().clone() for p in self._params]

    def _set_params(self, params_data):
        with torch.no_grad():
            for p, pdata in zip(self._params, params_data):
                p.copy_(pdata)

    def _add_flat_direction(self, alpha, direction):
        offset = 0
        with torch.no_grad():
            for p in self._params:
                n = p.numel()
                p.add_(
                    direction[offset:offset + n].view_as(p),
                    alpha=alpha
                )
                offset += n

    def step(self, closure):
        if self.U is None or self.S is None:
            raise RuntimeError(
                "NNCG preconditioner 尚未初始化，请先 update_preconditioner()。"
            )

        with torch.enable_grad():
            loss, grad_tuple = closure()

        flat_grad = self._flatten_grad_tuple(grad_tuple)

        def hvp_fn(v):
            return self._hvp(flat_grad, v)

        direction, cg_iters, cg_residual = _nystrom_pcg(
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

        self.old_dir = direction.detach()

        descent_direction = -direction.detach()
        directional_derivative = torch.dot(
            flat_grad.detach(),
            descent_direction
        )

        # 非正定/数值异常时退化成负梯度方向
        if directional_derivative >= 0:
            print(
                "NNCG warning: Newton direction is not descent; "
                "fallback to -gradient."
            )
            descent_direction = -flat_grad.detach()
            directional_derivative = -torch.dot(
                flat_grad.detach(),
                flat_grad.detach()
            )

        step_size = self.param_groups[0]["lr"]

        if self.line_search_fn == "armijo":
            params_init = self._clone_params()
            loss0 = loss.detach().item()

            c1 = 0.1
            beta = 0.5

            for _ in range(20):
                self._set_params(params_init)
                self._add_flat_direction(
                    step_size,
                    descent_direction
                )

                with torch.enable_grad():
                    trial_loss, _ = closure()

                rhs = (
                    loss0
                    + c1 * step_size * directional_derivative.item()
                )

                if trial_loss.detach().item() <= rhs:
                    break

                step_size *= beta

            self._set_params(params_init)

        self._add_flat_direction(
            step_size,
            descent_direction
        )

        return (
            loss.detach().item(),
            flat_grad.detach(),
            cg_iters,
            cg_residual,
            step_size
        )


# ============================================================
# 7. 共同阶段：Adam 训练 15000 轮
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

t0 = time.perf_counter()

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

adam_stage_time = time.perf_counter() - t0


# ============================================================
# 8. 从 Adam-15000 checkpoint 创建两个分支
# ============================================================

adam_15000_state_ac = copy.deepcopy(base_model.state_dict())

adam_only_model_ac = PINN(LAYERS).to(device)
adam_only_model_ac.load_state_dict(adam_15000_state_ac)

lbfgs_model_ac = PINN(LAYERS).to(device)
lbfgs_model_ac.load_state_dict(adam_15000_state_ac)


# ============================================================
# 9. 分支 A：Adam-only 15000 -> 20000
# ============================================================

optimizer_adam_only_ac = torch.optim.Adam(
    adam_only_model_ac.parameters(),
    lr=LEARNING_RATE
)

adam_only_history_ac = []

print("\n========== Branch A: Adam-only 15000 -> 20000 ==========")

t0 = time.perf_counter()

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

adam_continue_time = time.perf_counter() - t0


# ============================================================
# 10. 分支 B：Adam -> L-BFGS
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
    optimizer_lbfgs_ac.zero_grad()

    loss_u = data_loss(lbfgs_model_ac)
    loss_f = physics_loss_ac(lbfgs_model_ac)
    loss = loss_u + loss_f

    loss.backward()
    lbfgs_history_ac.append(loss.item())

    return loss


print("\n========== Branch B: Adam 15000 -> L-BFGS ==========")

t0 = time.perf_counter()
optimizer_lbfgs_ac.step(closure_lbfgs_ac)
lbfgs_time = time.perf_counter() - t0

# NNCG 从 L-BFGS 最终点继续，不再额外跑一次 L-BFGS，
# 避免第三阶段起点被“额外 L-BFGS 验证”改变。
lbfgs_final_state_ac = copy.deepcopy(lbfgs_model_ac.state_dict())


# ============================================================
# 11. 分支 C：Adam -> L-BFGS -> NNCG
# ============================================================

nncg_model_ac = PINN(LAYERS).to(device)
nncg_model_ac.load_state_dict(lbfgs_final_state_ac)

optimizer_nncg_ac = NysNewtonCG(
    nncg_model_ac.parameters(),
    lr=NNCG_LR,
    rank=NNCG_RANK,
    mu=NNCG_MU,
    cg_tol=NNCG_CG_TOL,
    cg_max_iters=NNCG_CG_MAX_ITERS,
    line_search_fn=NNCG_LINE_SEARCH
)

nncg_history_ac = []
nncg_cg_iters = []
nncg_cg_residuals = []
nncg_step_sizes = []

print("\n========== Branch C: Adam -> L-BFGS -> NNCG ==========")
print(
    f"NNCG config | steps={NNCG_STEPS} | rank={NNCG_RANK} | "
    f"mu={NNCG_MU:.1e} | cg_tol={NNCG_CG_TOL:.1e} | "
    f"cg_max_iters={NNCG_CG_MAX_ITERS}"
)

t0 = time.perf_counter()

for step in range(NNCG_STEPS):

    def closure_nncg_ac():
        optimizer_nncg_ac.zero_grad(set_to_none=True)

        loss_u = data_loss(nncg_model_ac)
        loss_f = physics_loss_ac(nncg_model_ac)
        loss = loss_u + loss_f

        grad_tuple = torch.autograd.grad(
            loss,
            tuple(nncg_model_ac.parameters()),
            create_graph=True
        )

        return loss, grad_tuple

    # 第 0 步以及之后每隔固定步数重建 Nyström 预条件器
    if step % NNCG_PRECOND_UPDATE_FREQ == 0:
        optimizer_nncg_ac.zero_grad(set_to_none=True)

        loss_u = data_loss(nncg_model_ac)
        loss_f = physics_loss_ac(nncg_model_ac)
        loss_precond = loss_u + loss_f

        grad_tuple_precond = torch.autograd.grad(
            loss_precond,
            tuple(nncg_model_ac.parameters()),
            create_graph=True
        )

        print(
            f"\nNNCG | Step {step:4d} | "
            f"updating Nyström preconditioner..."
        )

        optimizer_nncg_ac.update_preconditioner(
            grad_tuple_precond
        )

        del grad_tuple_precond
        del loss_precond

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    loss_before, grad_flat, cg_iter, cg_res, step_size = (
        optimizer_nncg_ac.step(closure_nncg_ac)
    )

    # 记录更新后的真实 Loss
    loss_after, _, _ = total_loss_ac(nncg_model_ac)
    loss_after_value = loss_after.detach().item()

    nncg_history_ac.append(loss_after_value)
    nncg_cg_iters.append(cg_iter)
    nncg_cg_residuals.append(cg_res)
    nncg_step_sizes.append(step_size)

    if step % 10 == 0 or step == NNCG_STEPS - 1:
        print(
            f"NNCG | Step {step:4d} | "
            f"Loss: {loss_after_value:.3e} | "
            f"|g|: {torch.norm(grad_flat).item():.3e} | "
            f"CG: {cg_iter:3d} | "
            f"CG-res: {cg_res:.3e} | "
            f"step_size: {step_size:.3e}"
        )

nncg_time = time.perf_counter() - t0


# ============================================================
# 12. 最终结果
# ============================================================

adam_result = evaluate_ac(adam_only_model_ac)
lbfgs_result = evaluate_ac(lbfgs_model_ac)
nncg_result = evaluate_ac(nncg_model_ac)

print("\n========== Final Results ==========")
print(f"Primary L-BFGS closure calls: {len(lbfgs_history_ac)}")

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

print("\nAdam -> L-BFGS -> NNCG")
print(f"Total Loss  = {nncg_result[0]:.3e}")
print(f"Data Loss   = {nncg_result[1]:.3e}")
print(f"PDE Loss    = {nncg_result[2]:.3e}")
print(f"Relative L2 = {nncg_result[3]:.3e}")

print("\n========== Runtime ==========")
print(f"Shared Adam stage : {adam_stage_time:.1f} s")
print(f"Adam continuation : {adam_continue_time:.1f} s")
print(f"L-BFGS stage      : {lbfgs_time:.1f} s")
print(f"NNCG stage        : {nncg_time:.1f} s")


# ============================================================
# 13. 生成完整预测场
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

    u_pred_nncg = (
        nncg_model_ac(X_star_tensor)
        .cpu()
        .numpy()
        .reshape(Exact.shape)
    )

error_adam = np.abs(u_pred_adam - Exact)
error_lbfgs = np.abs(u_pred_lbfgs - Exact)
error_nncg = np.abs(u_pred_nncg - Exact)


# ============================================================
# 14. 对比图 1：Loss 优化轨迹
# ============================================================

adam_stage1_steps = np.arange(len(adam_history_ac["total"]))

adam_only_steps = np.arange(
    ADAM_STAGE1,
    ADAM_STAGE1 + len(adam_only_history_ac)
)

lbfgs_steps = np.arange(
    ADAM_STAGE1,
    ADAM_STAGE1 + len(lbfgs_history_ac)
)

nncg_start = ADAM_STAGE1 + len(lbfgs_history_ac)

nncg_steps_plot = np.arange(
    nncg_start,
    nncg_start + len(nncg_history_ac)
)

plt.figure(figsize=(11, 5.5))

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

plt.semilogy(
    nncg_steps_plot,
    nncg_history_ac,
    label="NNCG fine-tuning"
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

plt.xlabel("Optimization progress")
plt.ylabel("Total Loss")
plt.title("Allen-Cahn: Adam-only vs Adam -> L-BFGS -> NNCG")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "allen_cahn_loss_comparison_nncg.png",
    dpi=200
)
plt.show()


# ============================================================
# 15. 对比图 2：全场绝对误差
# ============================================================

error_vmax = max(
    error_adam.max(),
    error_lbfgs.max(),
    error_nncg.max()
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 4.8),
    constrained_layout=True
)

errors = [error_adam, error_lbfgs, error_nncg]

titles = [
    f"Adam-only\nRelative L2 = {adam_result[3]:.3e}",
    f"Adam -> L-BFGS\nRelative L2 = {lbfgs_result[3]:.3e}",
    f"Adam -> L-BFGS -> NNCG\nRelative L2 = {nncg_result[3]:.3e}"
]

ims = []

for ax, err, title in zip(axes, errors, titles):
    im = ax.imshow(
        err,
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

    ims.append(im)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(title)

fig.colorbar(
    ims[-1],
    ax=axes,
    label="Absolute Error"
)

plt.savefig(
    RESULT_DIR / "allen_cahn_error_comparison_nncg.png",
    dpi=200
)
plt.show()


# ============================================================
# 16. 对比图 3：Relative L2
# ============================================================

methods = [
    "Adam-only",
    "Adam -> L-BFGS",
    "Adam -> L-BFGS\n-> NNCG"
]

relative_l2_values = [
    adam_result[3],
    lbfgs_result[3],
    nncg_result[3]
]

plt.figure(figsize=(8, 5))

plt.bar(
    methods,
    relative_l2_values
)

plt.ylabel("Relative L2 Error")
plt.title("Allen-Cahn Full-field Relative L2 Error")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "allen_cahn_relative_l2_nncg.png",
    dpi=200
)
plt.show()


# ============================================================
# 17. NNCG 精调阶段单独观察
# ============================================================

plt.figure(figsize=(9, 5))

plt.semilogy(
    np.arange(len(nncg_history_ac)),
    nncg_history_ac
)

plt.xlabel("NNCG Step")
plt.ylabel("Total Loss")
plt.title("Allen-Cahn: NNCG Fine-tuning")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "allen_cahn_nncg_finetuning.png",
    dpi=200
)
plt.show()


print(f"\nFigures saved to: {RESULT_DIR.resolve()}")
