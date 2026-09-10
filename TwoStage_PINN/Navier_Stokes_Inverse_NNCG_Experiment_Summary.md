# Navier--Stokes 逆问题：Adam → L-BFGS → NNCG 优化实验

整理时间：2026-09-10

## 1. 实验目标

本实验基于二维不可压缩 Navier--Stokes 方程构建 PINN
逆问题，在利用速度场观测数据约束网络的同时识别未知物理参数：

\[ `\lambda`{=tex}\_1 = 1,`\qquad `{=tex}`\lambda`{=tex}\_2 = 0.01 \]

其中 (`\lambda`{=tex}\_1) 控制非线性对流项，(`\lambda`{=tex}\_2)
对应黏性系数。

实验重点不是重新验证 PINN 能否求解该问题，而是考察不同优化阶段对 PINN
后期收敛和参数识别精度的影响，重点测试 NNCG 是否能在 L-BFGS
已经充分优化后继续改善结果。

------------------------------------------------------------------------

## 2. 数据与模型

数据使用 Raissi PINN 示例中的：

`cylinder_nektar_wake.mat`

网络输入：

\[ (x,y,t) \]

网络输出：

\[ (`\psi`{=tex},p) \]

通过流函数得到速度：

\[ u=`\psi`{=tex}\_y,`\qquad `{=tex}v=-`\psi`{=tex}\_x \]

并将二维不可压缩 Navier--Stokes 方程作为物理约束：

\[
u_t+`\lambda`{=tex}*1(uu_x+vu_y)+p_x-`\lambda`{=tex}*2(u*{xx}+u*{yy})=0
\]

\[
v_t+`\lambda`{=tex}*1(uv_x+vv_y)+p_y-`\lambda`{=tex}*2(v*{xx}+v*{yy})=0
\]

总损失由速度数据误差和 PDE residual 组成：

\[ L=L\_{`\text{data}`{=tex}}+L\_{`\text{PDE}`{=tex}} \]

训练采样点数为 5000。

------------------------------------------------------------------------

## 3. 优化器设计

根据 *Challenges in Training PINNs: A Loss Landscape Perspective*
中"先一阶、再准二阶/二阶优化"的思路，本实验没有让 Adam
占据主要训练预算，而是采用：

**Adam 少量预优化 → L-BFGS 主优化 → NNCG 后期精调。**

主要分支为：

``` text
Adam 5000
    ↓
L-BFGS 15000
    ↓
同一个 L-BFGS checkpoint
    ├── 继续 L-BFGS（与 NNCG 等 wall-clock）
    └── NNCG 200
```

同时保留 Adam-only 20000 作为基线。

最终比较四种结果：

1.  Adam-only (20000)
2.  Adam → L-BFGS
3.  Adam → L-BFGS → L-BFGS（matched time）
4.  Adam → L-BFGS → NNCG

NNCG 阶段设置为 200
steps。为了避免单纯因为"额外训练时间"导致结果改善，从相同的 L-BFGS
checkpoint 继续运行 L-BFGS，并使其额外 wall-clock 与 NNCG 基本一致。

------------------------------------------------------------------------

## 4. 最终结果

  -----------------------------------------------------------------------------------------------------------------
  方法             Total Loss       Data Loss        PDE Loss             λ1    λ1 Error             λ2    λ2 Error
  ----------- --------------- --------------- --------------- -------------- ----------- -------------- -----------
  Adam-only         2.821e-04       1.284e-04       1.537e-04       0.995972       0.40%       0.011085      10.85%
  (20000)                                                                                               

  Adam →            1.696e-05       7.324e-06       9.633e-06       0.999811       0.02%       0.010257       2.57%
  L-BFGS                                                                                                

  Adam →            1.696e-05       7.323e-06       9.633e-06       0.999811       0.02%       0.010256       2.56%
  L-BFGS →                                                                                              
  L-BFGS                                                                                                
  (matched                                                                                              
  time)                                                                                                 

  Adam →        **1.470e-05**   **6.537e-06**   **8.161e-06**   **0.999920**   **0.01%**   **0.010247**   **2.47%**
  L-BFGS →                                                                                              
  NNCG                                                                                                  
  -----------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 5. 结果分析

### Adam → L-BFGS

L-BFGS 是本实验中最主要的性能提升来源。

相较 Adam-only，Total Loss 从：

\[ 2.821`\times10`{=tex}\^{-4} `\rightarrow
1.696`{=tex}`\times10`{=tex}\^{-5} \]

下降约 **94%**。

其中较难识别的 (`\lambda`{=tex}\_2) 相对误差由：

\[ 10.85% `\rightarrow
2.57`{=tex}% \]

说明在 Adam 已经完成初步优化后，L-BFGS 对 PINN
的高精度收敛和逆问题参数识别具有明显作用。

### L-BFGS 继续训练

原始 L-BFGS 阶段达到设定的 15000 closure calls 后，从该 checkpoint
继续运行 L-BFGS。

额外运行时间与 NNCG 均为：

**412.2 s**

继续 L-BFGS 共进行了 18243 次额外 closure calls，但最终：

\[ 1.696`\times10`{=tex}\^{-5} `\rightarrow
1.696`{=tex}`\times10`{=tex}\^{-5} \]

几乎没有进一步改善。

这说明在当前设置下，L-BFGS
已基本进入平台区域，单纯增加相同量级的计算时间难以继续显著降低目标函数。

### NNCG 后期精调

从完全相同的 L-BFGS checkpoint 切换到 NNCG 200 steps 后：

\[ 1.696`\times10`{=tex}\^{-5} `\rightarrow
1.470`{=tex}`\times10`{=tex}\^{-5} \]

Total Loss 进一步下降约 **13.3%**。

PDE Loss 从：

\[ 9.633`\times10`{=tex}\^{-6} `\rightarrow
8.161`{=tex}`\times10`{=tex}\^{-6} \]

进一步下降约 **15.3%**。

参数识别也继续小幅改善：

\[ `\lambda`{=tex}\_1`\text{ Error}`{=tex}:
0.02%`\rightarrow0.01`{=tex}% \]

\[ `\lambda`{=tex}\_2`\text{ Error}`{=tex}:
2.57%`\rightarrow2.47`{=tex}% \]

NNCG 200 steps 中：

`fallback count = 0 / 200`

说明本次 Navier--Stokes 实验中 NNCG 的 Newton/CG
更新整体较稳定，没有出现此前 Allen--Cahn 后期频繁退回负梯度方向的现象。

------------------------------------------------------------------------

## 6. 运行时间

  阶段                                    时间
  ---------------------------------- ---------
  Shared Adam stage                     94.7 s
  Adam continuation                    283.8 s
  L-BFGS stage                         496.8 s
  NNCG 200                             412.2 s
  Matched-time L-BFGS continuation     412.2 s

由于不同优化器单步计算成本差异很大，因此本实验没有用"相同步数"判断优化器优劣，而是在后期精调阶段增加了等
wall-clock 对照。

------------------------------------------------------------------------

## 7. 实验结论

本实验得到三个主要结论：

**1. Adam → L-BFGS 的两阶段训练明显优于 Adam-only。**

Adam 适合完成前期快速优化，而 L-BFGS
是本实验高精度收敛和参数识别改善的主要来源。

**2. L-BFGS 在长时间优化后出现明显平台。**

在已有 L-BFGS checkpoint 上继续投入约 412 s、18243 次额外 closure
calls，Loss 和参数识别结果几乎不再改善。

**3. NNCG 能够在 L-BFGS 平台之后继续提供有限但稳定的优化增益。**

在完全相同的起点和相同额外 wall-clock 下，继续 L-BFGS 基本无改善，而
NNCG 将 Total Loss 进一步降低约 13.3%，PDE Loss 降低约
15.3%，并进一步改善两个未知参数的识别结果。

因此，在本次 Navier--Stokes PINN 逆问题上，更合适的优化流程可以概括为：

\[ `\boxed{\text{Adam 前期探索}\rightarrow
\text{L-BFGS 主优化}\rightarrow
\text{NNCG 后期精调}}`{=tex} \]

需要注意的是，本实验仍属于单一问题、单次训练设置下的验证，当前结果支持
NNCG 作为后期精调优化器具有价值，但不足以据此认为 NNCG 在所有 PINN
问题上都优于继续 L-BFGS。

------------------------------------------------------------------------

## 8. 实验状态

**本实验已完成，可以告一段落。**

已验证内容：

-   Navier--Stokes PINN 逆问题参数识别；
-   Adam-only 与 Adam → L-BFGS 对比；
-   L-BFGS 后接 NNCG；
-   NNCG 200 steps 稳定性；
-   NNCG 与继续 L-BFGS 的等 wall-clock 公平对照；
-   Loss、PDE Loss 与未知参数识别精度比较。

后续若重新研究 NNCG，可进一步扩展到多随机种子、不同 PDE
或更复杂逆问题，而无需继续增加当前实验的训练轮数。
