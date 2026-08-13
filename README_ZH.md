# ComfyUI-sol-attn

**[English](./README.md)** | **中文**

**版本: v0.6.2**

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange)](https://github.com/comfyanonymous/ComfyUI)
[![GPU](https://img.shields.io/badge/tested-RTX%205090%20(SM120)-76b900)](https://www.nvidia.com/)
[![Triton](https://img.shields.io/badge/Triton-3.6.0-blue)](https://github.com/triton-lang/triton)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

ComfyUI 视频扩散模型的稀疏注意力与显存优化节点包,基于 NVIDIA 的 **Sol-Attn** Triton 参考内核构建,并针对原生 Windows 消费级 Blackwell(SM120 / RTX 50 系列)调优。包含一个通用的逐模型 Sol-Attn 补丁,以及四个 MiniMax H3 专用节点:零拷贝注意力、调度稀疏、逐位精确的调制融合和前馈峰值显存削减。

> 法律说明:`sol_kernel/` 中的内核为 NVIDIA 源代码(Apache-2.0)的本地修改副本。NVIDIA 现已为 Linux 提供可选的 SM120 CuTe 后端;本仓库的原生 Windows Triton 指针路径和残差 int8 扩展仍是本地改动。

## v0.6.2

- **支持 SM86 / RTX 30 系列** —— MiniMax H3 Sol Attention 现通过 SM89 与 SM120 共用的指针内核族支持 Ampere SM86 GPU。SM90/SM100/SM121 继续使用原有 TMA 路径。
- **社区硬件验证** —— RTX 3090 Ti 在严格模式下完成了 25,323-token 的 MiniMax H3 工作流并确认 Sol 已激活,随后又在同时关闭 `int8_qk` 与 `int8_pv` 的情况下成功生成。残差 int8 QK/PV 路径也已成功运行。
- **回归测试覆盖** —— 适用于 SM86 的 6 项仓库测试及贡献者本地的运行时版本检查全部通过;仅用于 SM120 的 pointer-vs-TMA 对比按预期跳过。合并后,仓库全部 7 项测试也在 SM120 上通过。覆盖架构分发、指针 INT8 路径、H3 调制融合和 KJNodes 交接。
- **数值不变** —— 本版本仅启用新增架构,不改变注意力数学、权重、稀疏设置或输出质量行为。现有 SM89、SM90、SM100、SM120 与 SM121 分发保持不变。SM86 性能尚未进行正式基准测试。

## v0.6.1

- **兼容 KJNodes 低显存节点** —— 两个 MiniMax H3 Sol 节点现已支持 KJNodes `MiniMax H3 Low VRAM Attention` 使用的单元素激活列表交接。Sol 在判断是否接管调用时只读取张量;稠密回退时保留交接列表,运行 Sol 时则消费并释放它。
- **组合回归测试** —— `KJ MiniMax H3 Low VRAM Attention → MiniMax H3 Memory Efficient Sol Attention` 及 Scheduled Sol 变体现在可以共同运行,不再出现 `'list' object has no attribute 'shape'`。稀疏与稠密调用均保留 KJ 的提前释放激活机制。
- **数值与内核不变** —— 注意力数学、模型权重、SM89/SM120 指针分发、SM90/SM100/SM121 TMA 分发及输出精度均未改变。全部 7 项回归测试通过,真实 KJ 低显存 block-forward GPU 集成测试也已通过。v0.6.0 的基准矩阵仍然有效。
- **原有限制不变** —— KJNodes `MiniMax H3 Low VRAM Attention` 仍不应与 `MiniMax H3 Fused Modulation` 同时使用,因为两者都会修改完整 H3 block forward。本版本修复的是它与本仓库两个 H3 **Sol Attention** 节点的组合。

## v0.6.0

- **更快的 SM120 forward 分发** —— RTX 5090 现在默认使用指针 forward;SM89 保持指针路径,SM90/100/121 保持 TMA。H3 形状 `B=1,T=8192,H=56,D=128` 下,bf16 指针路径吞吐量为 TMA 的 1.25×,输出逐位一致;残差 int8 同样逐位一致。
- **内联残差 int8 Q 预处理** —— SM89/SM120 的 `diag` 指针内核直接利用 forward 已加载的 BF16 Q tile 完成 Q 量化与路由阈值计算,不再生成 Q-int8/Q-scale/threshold 中间张量。32K H3 tokens 下实测峰值分配减少 189 MiB;整除/非整除长度、精确汇聚和 `int8_pv` 开关均与旧路径逐位一致。
- **逐位精确的 H3 调制融合** —— 新增 `MiniMax H3 Fused Modulation` 节点,融合全部 50 个 DiT 块的分段 AdaLN scale/shift 与门控残差更新。它显式复现 eager BF16 中间舍入,真实 ComfyUI `DiTBlock` 测试逐位一致。在 38,247 × 5,376 形状下,scale/shift 独立实测 1.91×,gate/add 1.22×。
- **注意力补丁组合保持不变** —— 融合节点在运行时动态解析每个块的 attention 与 MLP,因此推荐的 `全局 KJ Sage → H3 显存高效 Sage → 本地 H3 Sol` 链仍可共同工作,且不会改变 Sol 之外的注意力调用。
- **全新完整发布矩阵** —— 在 autotune 缓存热身后重测 8K/16K/32K/65K:bf16 吞吐量为 SageAttention 的 1.38–1.65×,残差 `int8_qk` 为 1.73–1.97×,按需开启的 `int8_qk+pv` 为 1.98–2.33×。相对 bf16 Sol 路径的 L2 误差保持为 `0.00802`/`0.01396`。

## v0.5.9

- **更快的残差 int8 预处理** —— K 的 64-token 块均值归约与残差量化现已合并到单个 Triton 内核中,只读取一次 K。在 RTX 5090 上,独立的 K/V 汇总 + K 量化预处理阶段于 8K、16K、65K tokens 实测提速 26–36%(32K 结果波动较大)。残差 int8 公式与 FP32 累加方式不变;验证中未发现路由变化。
- **所有受支持架构保持原有 forward 路径** —— 这是共享预处理优化。SM89 仍使用指针 forward 内核;SM90/100/120/121 仍使用 TMA forward 内核。架构分发逻辑未改动。
- **明确记录 KJNodes 组合方式** —— MiniMax H3 三补丁组合必须依次应用全局 KJ Sage、KJ 的 MiniMax 显存高效 Sage 补丁,最后应用本仓库的 MiniMax Sol 补丁。Sol 拒绝处理的 tokens 使用已捕获的显存高效 Sage forward;该 H3 对象补丁之外的注意力调用继续使用全局 Sage 覆盖。

## 为什么选择本仓库

在 RTX 5090 上实测(当前本地发布矩阵见 [BENCHMARKS.md](BENCHMARKS.md),2026-08-09;第三方对比作为历史数据保留):

- **bf16 吞吐量为 Sage 的 1.38–1.65×**(8K–65K);残差 int8 达 1.73–1.97×,按需的 P·V int8 达 1.98–2.33×。
- **最佳的 int8 精度** —— 本内核仅量化 K 的块内*残差*,均值项以 bf16 精确保留:相对 L2 误差 0.008,比全键 int8 设计(0.029)接近精确路径约 3.6 倍。
- **零拷贝设计** —— 内核直接读取 H3 融合 qkv 投影的视图;其他 TMA 实现会先拷贝 q/k/v(长序列下额外增加 1.3–2.7 GiB 峰值显存与拷贝时间)。
- **数学实现交叉验证** —— 本仓库 bf16 路径与 kijai 的独立实现逐位一致(0.000000),全精确模式与 SDPA 一致(0.00097)。
- **不止于内核** —— 带曲线预览的调度 tau、条件精确 KV 汇聚、前馈分块(MLP 峰值 −37%)、int8 q/k 与 P·V 量化、SM86–SM121 支持,以及诚实的逐调用回退。

## 功能特性

- **按需逐模型打补丁** —— 只有接入节点的模型受影响,工作流其余部分不受影响。
- **两条 Sol-Attn 集成路径** —— 适用于任意模型的通用钩子节点,以及 MiniMax H3 专用节点:将融合 qkv 投影的跨步视图直接送入内核,零 q/k/v 拷贝,并为 H3 打包的条件行提供精确 KV 汇聚。
- **SM86 至 SM121** —— SM86、SM89 与 SM120 使用指针内核;SM90、SM100 与 SM121 使用 TMA 描述符内核。SM121 覆盖 DGX Spark。
- **调度稀疏** —— 在采样过程中渐变 `tau`(前期稀疏、后期致密),并输出调度曲线预览图。
- **逐位精确的 H3 调制融合** —— 合并分段 AdaLN 与门控残差逐元素操作,不改变 eager BF16 输出或所选注意力后端。
- **前馈分块** —— 压低 MiniMax H3 的 MLP 峰值激活显存,输出逐位一致。
- **诚实回退** —— 内核无法处理的形状或 GPU 自动回退到你原有的注意力后端并记录原因;`strict` 模式则直接抛错,用于验证新环境。
- **无新增依赖** —— 仅需 torch 和 Triton;matplotlib 仅用于调度预览图(可选)。

## 前置条件

- NVIDIA GPU:**SM86、SM89、SM90、SM100、SM120 或 SM121** —— SM86/89/120 运行指针 forward,SM90/100/121 运行 TMA。SM120 已在本地完成测试和基准;SM86 已在 RTX 30 系列上完成硬件冒烟测试;SM89 已由社区在 RTX 4080 SUPER 上完成硬件冒烟测试([issue #2](https://github.com/Saganaki22/ComfyUI-sol-attn/issues/2));SM121(DGX Spark)也已由社区测试。SM86/SM89 尚未在本仓库完成性能基准。
- 支持 CUDA 与 **bfloat16** 的 PyTorch
- 带有 `triton.tools.tensor_descriptor`(TMA)的 **Triton** —— 已在 3.6.0 上验证
- ComfyUI(基于 0.30.0 开发)
- 使用 MiniMax H3 节点时:需要 MiniMax H3 检查点,例如放置于 `ComfyUI/models/diffusion_models/` 的 `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- matplotlib(可选 —— 仅用于 tau 调度预览图)

## 安装

```bash
cd ComfyUI/custom_nodes
git clone <本仓库地址>
```

重启 ComfyUI。除前置条件外无需 pip 安装任何内容。

## 测试环境

```text
RTX 5090 (SM120)  ·  torch 2.10.0+cu130  ·  Triton 3.6.0
Python 3.12.10    ·  ComfyUI 0.30.0      ·  Windows 11
MiniMax H3(56 头 × 128,bf16,mask=None)—— 满足全部内核约束
```

社区测试于 **NVIDIA DGX Spark**(SM121,aarch64,CUDA 13.0,Triton 3.6.0)—— DGX Spark 数据见 [BENCHMARKS.md](BENCHMARKS.md)。Sol-Attn 相对 SageAttention 的加速比在该设备上更高(1.48–1.92×),因为 GB10 的 LPDDR5X 统一内存受限于带宽,而 Sol-Attn 节省带宽。

Sol-Attn 运行时约束:`head_dim` 必须恰好为 128、bf16、无注意力掩码、4D q/k/v 且为连续或 TMA 兼容跨步布局。不满足时将回退,并按原因各记录一次日志。

## 节点接入顺序

在 MiniMax H3 工作流中的位置:

```text
UNETLoader → (LoRA / 其他模型补丁)
          → Patch Sage Attention(KJNodes,可选的全局回退)
          → MiniMax H3 Memory Efficient Sage Attention Patch(KJNodes,可选的 H3 回退)
          → MiniMax H3 Scheduled Sol Attention Patch(或 Memory Efficient 版)
          → MiniMax H3 Fused Modulation
          → MiniMax H3 Chunk FeedForward
          → EasyCache(可选,核心节点)
          → guider / sampler
```

- **两个 H3 注意力节点二选一,切勿同时使用。** 调度节点是显存高效节点的超集(`tau_start = tau_end` 时两者完全等价)。
- **同时使用三个注意力补丁时,顺序必须为:**`Patch Sage Attention (KJNodes) → MiniMax H3 Memory Efficient Sage Attention Patch (KJNodes) → MiniMax H3 Sol 补丁(本仓库)`。本仓库的 H3 包装器会捕获已打补丁的显存高效 Sage forward,并在短序列、门控步骤、不支持的形状和非严格模式内核失败时使用它。其他注意力调用使用 KJNodes 的全局 Sage 覆盖。若把 H3 Sage 补丁放在 Sol 之后,它会完全覆盖 Sol。
- **在此组合中,通用 `SolAttentionPatch` 不能替代 MiniMax 专用 Sol 节点。** KJNodes 的 MiniMax 显存高效 Sage 补丁会直接替换各 H3 注意力模块的 `forward`,绕过通用节点使用的全局注意力覆盖。
- **Fused Modulation 与注意力后端无关。** 它可放在注意力补丁前后,并动态调用已安装的 H3 attention/MLP 对象补丁。为清晰起见建议使用上图顺序。KJNodes 的 `MiniMax H3 Low VRAM Attention` 会替换整个 block forward,因此不能与本融合叠加;遇到已有的未知 block-forward 补丁时,本节点会保留原补丁。
- 非 H3 模型请改用通用的 `SolAttentionPatch`:`UNETLoader → Sol-Attn → guider`。

## 节点

<details>
<summary><strong>1. Sol-Attn(稀疏注意力)</strong> —— 通用逐模型补丁,<code>model_patches/attention</code></summary>

通过 ComfyUI 的 `optimized_attention_override` 钩子,将任意模型的自注意力路由到 Sol-Attn。内核无法处理的情况会回退到你已有的注意力覆盖或 ComfyUI 所选后端。

```text
UNETLoader → Sol-Attn → BasicGuider
```

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| `model` | MODEL | 必填 | 要打补丁的模型。 |
| `enabled` | BOOLEAN | `True` | 设为 `False` 可在不改线的情况下 A/B 对比。 |
| `tau` | FLOAT | `1.3` | 路由阈值,以块分数均值之上的标准差计。越高 = 越多 KV 块走近似路径 = 更快、保真度更低。`1.0` 为 Sol-Attn 论文默认值;`1.3` 为本仓库调优后的默认值。 |
| `min_tokens` | INT | `4096` | 低于此序列长度时使用常规后端。 |
| `strict` | BOOLEAN | `False` | 内核报错时抛出而非回退。验证新 GPU 或 Triton 版本时开启。 |
| `thresh_type` | COMBO | `diag` | `diag`(评估默认值)或 `exact` —— 使用二阶矩统计获得更精确的路由阈值,代价是额外预计算。 |
| `int8_qk` | BOOLEAN | `False` | 将精确注意力路径的 q/k 量化为 int8。SM120 内联 Q 指针路径从 8K 起实测更快,32K 时峰值分配减少 189 MiB,额外数值误差约 1%。这是本仓库的新增功能。 |
| `int8_pv` | BOOLEAN | `False` | 同时将 P·V 点积量化为 int8(逐 token P、逐通道 V)。需要 `int8_qk`。当前硬件上速度与 int8_qk 基本持平;精度降至 rel L2 0.014(int8_qk 单独为 0.008)。按需开启。 |

**输出:** `model`(`MODEL`)

</details>

<details>
<summary><strong>2. MiniMax H3 显存高效 Sol 注意力补丁</strong> —— 零拷贝 H3 注意力,<code>model_patches/attention</code></summary>

通用节点的 H3 专用替代。通用节点从 ComfyUI 注意力钩子接收 BHSD 布局的 q/k/v,必须为内核做连续化拷贝;本节点直接替换注意力模块的 forward,将模型融合 qkv 投影的跨步 NHD 视图送入内核 —— 零 q/k/v 拷贝,并使用与原版模型相同的原地融合 RMSNorm+RoPE。

```text
UNETLoader → MiniMax H3 Memory Efficient Sol Attention Patch → BasicGuider
```

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| `model` | MODEL | 必填 | MiniMax H3 模型;其他模型将警告并原样返回。 |
| `enabled` | BOOLEAN | `True` | 设为 `False` 可在不改线的情况下 A/B 对比。 |
| `tau` | FLOAT | `1.3` | 与通用节点相同的路由阈值。 |
| `min_tokens` | INT | `4096` | 低于此打包序列长度时使用原版注意力 forward。 |
| `strict` | BOOLEAN | `False` | 内核报错时抛出而非回退。 |
| `thresh_type` | COMBO | `diag` | 与节点 1 相同的估计器选择。 |
| `int8_qk` | BOOLEAN | `False` | 与节点 1 相同的 int8 q/k 开关。 |
| `int8_pv` | BOOLEAN | `False` | 与节点 1 相同的 int8 P·V 开关。需要 `int8_qk`。 |
| `sink_conditioning` | COMBO | `exact_kv` | 保持 H3 打包的文本/条件/参考/音频 KV 块精确(约 3% 开销,保护提示词遵循与音画同步)。`exact_kv_and_rows` 同时让这些查询行走完全稠密路径(约 20% 开销)。`off` 关闭。 |
| `dense_blocks` | STRING | 空 | 保持稠密的 Transformer 块,如 `0-2,-1` 表示前三个与最后一个(负数从末尾计数)。首尾块对近似误差最敏感。留空则全部稀疏化。 |

仅修补 50 个主 DiT 块;token refiner 与短序列行为与原版完全一致。本节点可以接在显存高效 sage 注意力补丁(如 KJNodes 的 MiniMax H3 补丁)**之后**:此时它会将 sage forward 作为回退路径 —— 被门控或不符合条件的步骤运行显存高效 sage,符合条件的步骤运行 Sol-Attn。若顺序相反(本节点在前),sage 补丁会完全覆盖本节点 —— 顺序很重要。

**输出:** `model`(`MODEL`)

</details>

<details>
<summary><strong>3. MiniMax H3 调度 Sol 注意力补丁</strong> —— 带曲线预览的 tau 渐变,<code>model_patches/attention</code></summary>

与节点 2 相同的零拷贝注意力路径,但 `tau` 随采样渐变:在早期高噪声步(注意力结构松散)更稀疏,在后期细节成形步更致密。当前步数由扩散时间步自动跟踪,调度可自适应任意步数。

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| `model` | MODEL | 必填 | MiniMax H3 模型。 |
| `enabled` | BOOLEAN | `True` | 设为 `False` 可在不改线的情况下 A/B 对比。 |
| `tau_start` | FLOAT | `1.3` | 第一步(噪声最高)时的 tau。 |
| `tau_end` | FLOAT | `0.8` | 最后几步(低噪声)时的 tau。 |
| `curve` | COMBO | `linear` | `linear`、`cosine`、`sqrt`、`smoothstep`、`exponential` 或 `step`(中点硬切换)—— 两端之间的插值方式。 |
| `min_tokens` | INT | `4096` | 低于此打包序列长度时使用原版注意力 forward。 |
| `strict` | BOOLEAN | `False` | 内核报错时抛出而非回退。 |
| `dense_percent` | FLOAT | `0.0` | 在采样的前此比例内保持原版稠密注意力 —— Sol-Attn 论文的配方为 `0.2`。`0` 表示关闭。 |
| `thresh_type` | COMBO | `diag` | 与节点 1 相同的估计器选择。 |
| `int8_qk` | BOOLEAN | `False` | 与节点 1 相同的 int8 q/k 开关。 |
| `int8_pv` | BOOLEAN | `False` | 与节点 1 相同的 int8 P·V 开关。需要 `int8_qk`。 |
| `sink_conditioning` | COMBO | `exact_kv` | 与节点 2 相同的条件汇聚选择。 |
| `dense_blocks` | STRING | 空 | 与节点 2 相同的稠密块规格。 |

**输出:** `model`(`MODEL`)、`tau_graph`(`IMAGE`)—— 接入 Preview Image 节点即可查看调度曲线。

</details>

<details>
<summary><strong>4. MiniMax H3 Fused Modulation</strong> —— 逐位精确的 DiT 逐元素融合,<code>model_patches/optimization</code></summary>

将 H3 的逐分段 eager AdaLN scale/shift 和门控残差更新替换为每个块四次 Triton 启动。token 到 AdaLN 行的查找表按打包布局只构建一次,由全部 50 个块共享。内核显式复现 BF16 中间舍入;随机张量与真实 ComfyUI `DiTBlock` 均通过 `torch.equal` 验证。

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| `model` | MODEL | 必填 | MiniMax H3 模型;其他模型原样通过。 |
| `enabled` | BOOLEAN | `True` | 设为 `False` 可在不改线的情况下 A/B 对比。 |

本节点不选择或包装注意力后端。全局 Sage、KJNodes H3 显存高效 Sage、本地 H3 Sol 与分块 MLP 都在运行时动态解析,无论补丁先后顺序都可组合。若存在未知的整个 block `forward` 补丁,本节点会保留它而不会绕过。

**输出:** `model`(`MODEL`)

</details>

<details>
<summary><strong>5. MiniMax H3 分块前馈</strong> —— MLP 峰值显存削减,<code>model_patches/memory</code></summary>

将 H3 的逐 token 前馈沿打包序列维度分块,每个分块内保留 ComfyUI 的 `linear_input_act` 实现。独立于 Sol-Attn —— 可搭配任意注意力后端。在 INT8 ConvRot 检查点上效果最佳:swiglu 激活被融合进 INT8 量化器,`[tokens, 28672]` 的第一次投影主导 MLP 峰值显存;在普通 bf16 检查点上,eager swiglu 路径会保留额外中间张量,收益很小。

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| `model` | MODEL | 必填 | MiniMax H3 模型。 |
| `enabled` | BOOLEAN | `True` | 设为 `False` 可在不改线的情况下 A/B 对比。 |
| `chunks` | INT | `2` | 更多分块可进一步压低 MLP 峰值激活显存。 |
| `min_tokens` | INT | `8192` | 低于此打包序列长度时保持原版全宽 MLP。 |

分块在数学上逐 token 独立,并保留 H3 的 INT8 ConvRot 路径(其激活缩放为逐行);测试中输出逐位一致。需要梯度的输入将使用原始未分块 MLP。

收益随打包序列长度线性增长:中间张量每 token 占 56 KB,因此节省量从 8K tokens 的约 238 MiB 增长到 65K 的约 1.9 GiB(实测,×2 分块,int8 检查点)。低于 `min_tokens` 时本节点完全不生效 —— 想在短视频上也获得缓解就调低,只关心长视频就调高。

**输出:** `model`(`MODEL`)

</details>

## 基准测试

跨仓库对比表(kijai、KingGore、SageAttention、SDPA):[BENCHMARKS.md](BENCHMARKS.md)。

以下数据均来自单台机器(见"测试环境")与单次内核构建。请将其视为附带真实数字的冒烟测试,而非基准测试套件。

<details>
<summary><strong>注意力速度 —— Sol-Attn vs SageAttention vs PyTorch SDPA</strong></summary>

H3 尺寸(B=1,H=56,D=128,bf16 输入),随机张量,`tau=1.0`,3 次热身后取 20 次迭代中位数。下表为 autotune 结果缓存后的第二次进程实测:

| tokens | PyTorch SDPA (ms) | SageAttention (ms) | Sol bf16 | 残差 int8_qk | int8_qk+pv |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 23.97 | 3.93 | 2.84 | 2.27 | **1.98** |
| 16,384 | 94.85 | 14.68 | 9.57 | 8.10 | **7.05** |
| 32,768 | 378.41 | 58.09 | 35.72 | 30.06 | **25.91** |
| 65,536 | 1,516.16 | 229.63 | 139.28 | 116.64 | **98.46** |

- bf16 在整张表中比 Sage 快 1.38–1.65×。残差 `int8_qk` 达到 Sage 的 1.73–1.97× 吞吐量,仍是默认的 INT8 质量选择。
- `int8_qk+pv` 是当前本地路径中最快的,吞吐量为 Sage 的 1.98–2.33×。相对 bf16 的 L2 误差从 `0.00802` 升到 `0.01396`,因此仍为按需开启且默认关闭。
- 约 4K tokens 以下 Sage 明显占优。`min_tokens` 默认 4,096;若只想要已实测的赢面,可提高到 8,192。
- SageAttention 是公平基线。列出 PyTorch SDPA 仅因其他 Sol-Attn 插件引用它 —— 在这些尺寸下它并未使用有竞争力的内核路径。
- 随机高斯输入是 Sol-Attn 内容相关路由的最差情况,真实提示词应达到或超过这些比值;多次运行间存在几个百分点的波动。
- 通用节点的 q/k/v 拷贝每次调用额外消耗 0.2–2 ms(视长度而定,见测试输出中的 "Sol generic")。

全模型参考:一组受控对照(MiniMax H3,15 秒,480×864,20 步,`res_multistep`,固定种子,同一输入图)测得 Sage 9.91 s/it → Sol 8.92 s/it(−10%,使用钩子式节点)。

</details>

<details>
<summary><strong>峰值显存 —— 注意力,通用拷贝路径 vs 跨步视图</strong></summary>

单次 H3 注意力调用的峰值激活显存(扣除常驻部分):

| tokens | 通用路径 (MiB) | 跨步路径 (MiB) | 节省 (MiB) |
|---:|---:|---:|---:|
| 8,192 | 452 | 116 | 336 |
| 16,384 | 903 | 231 | 672 |
| 32,768 | 1,806 | 462 | 1,344 |
| 65,536 | 3,612 | 924 | 2,688 |

节省量与跨步路径避免的三份 `[tokens, 7168]` bf16 连续拷贝相符。

</details>

<details>
<summary><strong>峰值显存 —— 前馈分块(真实 INT8 检查点)</strong></summary>

通过 ComfyUI 常规扩散模型加载器加载 `minimax_h3_fl2va_pruned_int8_convrot.safetensors`,取真实的首块 MLP。两条路径使用同一输入张量,热身后测量峰值(扣除常驻部分),并以 `torch.testing.assert_close` 验证输出相等:

| Tokens | FFN 完整 | FFN 分块 ×2 | 节省 MiB | 节省 % |
|---:|---:|---:|---:|---:|
| 8,192 | 644 MiB | 406 MiB | 238 MiB | 37.0% |
| 16,384 | 1,288 MiB | 812 MiB | 476 MiB | 37.0% |
| 32,768 | 2,576 MiB | 1,624 MiB | 952 MiB | 37.0% |
| 65,536 | 5,152 MiB | 3,248 MiB | 1,904 MiB | 37.0% |

单独测试吞吐量基本持平(±4% 噪声内)。这些数字针对 INT8 ConvRot 路径;bf16 的注意事项见节点 5 说明。

</details>

<details>
<summary><strong>正确性检查</strong></summary>

在"测试环境"所列机器上:

- 跨步视图内核输出与连续输入**逐位一致**(最大绝对差 0),包括非整除序列长度(8,191 / 12,345 / 38,247 tokens)。
- 全精确模式(`tau=-100`,仅用于验证)与 PyTorch SDPA 的相对 L2 误差为 `0.00097`。
- 完整修补的 H3 注意力模块与原版 forward 的相对 L2 误差为 `0.00009`,符合 bf16 累加差异。
- 共享的 SM86/SM89/SM120 指针实现通过 SM120 强制分发与 TMA 内核交叉验证后逐位一致(bf16 与 int8_qk);强制精确的汇聚模式与稠密输出一致。
- 内联 Q 残差 int8 与旧的中间张量路径逐位一致,覆盖整除/非整除长度、条件精确 KV、致密查询行和 `int8_pv` 开关。
- H3 调制/门控融合在 BF16 与 FP32 AdaLN 表上均与 eager BF16 逐位一致,并通过真实 ComfyUI `DiTBlock` 集成测试。
- 分块 ×2 的 MLP 输出与完整 MLP 完全一致(`assert_close`,rtol=atol=0)。

</details>

## 与 EasyCache 搭配

ComfyUI 核心自带 **EasyCache**/`LazyCache` 节点(`comfy_extras/nodes_easycache.py`),在输入变化小时跳过整次模型求值 —— 是 TeaCache 的官方维护替代,且支持 H3 的音频/视频双输出。它与本包所有节点兼容:EasyCache 整体跳过部分步数,而 Sol-Attn 和分块让剩余步更快、更省。不要同时拉满两种近似 —— 激进的 `reuse_threshold` 加高 `tau` 会在画面中显现。请固定种子做 A/B。

## 控制台输出

```text
[Sol-Attn] patched (tau=1.30, min_tokens=4096, strict=False)
[Sol-Attn] active
[Sol-Attn] dense fallback: <reason>
[MiniMax H3 Sol] patched 50 attention blocks (tau=1.30, min_tokens=4096, strict=False)
[MiniMax H3 fusion] patched 50 of 50 blocks
[MiniMax H3 fusion] active (38247 tokens, 8 modulation segments)
[MiniMax H3 FFN] patched 62 MLPs (chunks=2, min_tokens=8192)
```

每种回退原因每次运行只记录一次。另请注意编译开销:Triton 以 `key=["T"]` 做 autotune,**每个新 token 数的首次运行都会在采样循环内支付一次 JIT 扫描** —— 计时结果会缓存到磁盘,因此同一 token 数全局只付一次,但改分辨率或时长后新尺寸仍需再付一次。请在第二次运行时测量。

## 注意事项

- **Sol-Attn 是近似方法。** 输出不会与稠密注意力逐位一致;是否影响画面由你判断 —— 用 `enabled` 做 A/B。
- **MiniMax H3 不在 Sol-Attn 论文评估范围内。** H3 使用联合打包序列(文本、条件、音频、视频);`sink_conditioning` 选项已实现论文的精确条件 K/V 处理,但首层稠密调度等论文中更保守配方的其余内容未实现。
- **原生 Windows SM120 指针路径与 SM121 集成属于本仓库。** NVIDIA 当前 Sol-Engine 分支为 SM120 提供可选的 Linux CuTe 后端与可移植 Triton 回退;本包仍保留适配 Comfy 的跨步布局和残差 int8 实现。SM121(DGX Spark)支持源自社区 PR #3。
- **H3 专用节点绕过了 ComfyUI 的注意力钩子。** 挂在 `optimized_attention_override` 上的其他补丁在 Sol 激活的块上不会运行,注意力相关的 `transformer_options` 补丁也不会在 Sol 路径上应用。
- **各架构路径有意保持独立。** SM86、SM89 与 SM120 使用指针内核族;SM120 已在 RTX 5090 上验证并完成基准测试;SM86 与 SM89 分别在 RTX 30、RTX 40 系列上完成硬件冒烟测试,但尚未在本仓库完成性能基准。共享的 SM86/SM89/SM120 实现还通过 SM120 强制分发进行了交叉验证。SM90/100/121 保持 TMA。在新环境上请先以 `strict=true` 跑一次。
- NVIDIA 公布的约 2.0–2.3× 数据面向整个 Sol-Engine(CuTe 内核、NVFP4、块融合、数据中心 GPU)。本包仅为 Triton 参考内核 —— 完全是另一回事。
- 已评估 [KingGore Blackwell 分支](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell):其 `flex_attention` 路径在 H3 尺寸、8,192 tokens 下为 4.334 ms,本 Triton 参考为 3.256 ms。它使用硬块掩码(未选中的块被丢弃而非近似),属于不同方法;其导入期修改已安装 PyTorch 包内文件的修复手段在此被有意排除。

## 致谢

- **Sol-Attn** —— Haopeng Li、Yitong Li、Junsong Chen、Tian Ye、Haozhe Liu、Jincheng Yu、Duomin Wang、Ruihua Zhang、Zeke Xie、Enze Xie、Song Han(NVIDIA Research,Efficient AI Team & Singapore Lab)。内核、方法以及 `sol_kernel/` 中的预处理均为他们的工作。
  - 项目主页: https://nvlabs.github.io/Sana/Sol-Attn/
  - 源码: https://github.com/NVlabs/Sana/tree/sol-engine
  - Sol-Attn 论文: https://arxiv.org/abs/2607.24027
  - Sol-Engine 论文: https://arxiv.org/abs/2606.23743

```bibtex
@misc{li2026solattnacceleratingvideogeneration,
      title={Sol-Attn: Accelerating Video Generation Inference via On-the-Fly
      Attention Sparsification},
      author={Haopeng Li and Yitong Li and Junsong Chen and Tian Ye and Haozhe
      Liu and Jincheng Yu and Duomin Wang and Ruihua Zhang and Zeke Xie and
      Enze Xie and Song Han},
      year={2026},
      eprint={2607.24027},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.24027},
}
```

- **[FlashAttention](https://github.com/dao-ailab/flash-attention)**(Tri Dao 等)—— NVIDIA 的第三方声明显示 Sol-Engine 的 SM90/SM100 脚手架部分源自 FlashAttention(BSD-3-Clause)。这些文件未在此再分发。
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)**(comfyanonymous 及贡献者)—— 本节点包所依赖的 `optimized_attention_override` 钩子与对象补丁机制。
- **[ComfyUI-SolAttn](https://github.com/sumeetprashant/ComfyUI-SolAttn)**([@sumeetprashant](https://github.com/sumeetprashant))—— 本仓库在其钩子式集成与 SM120 验证基础上扩展了 MiniMax H3 节点。
- **[ComfyUI_sol-attn_Blackwell](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell)**([@KingGore](https://github.com/KingGore))—— 使用编译版 PyTorch `flex_attention` 的 SM120 替代方案;已评估,未采用。
- **[ComfyUI-RadialAttn](https://github.com/woct0rdho/ComfyUI-RadialAttn)**([@woct0rdho](https://github.com/woct0rdho))—— 确立了本包遵循的按需 `MODEL → MODEL` 稀疏注意力补丁模式;其维护的 SageAttention Windows 构建用作回退后端与基线。
- **[RadialAttention](https://github.com/mit-han-lab/radial-attention)**(MIT Han Lab)—— 上述移植所包装的稀疏注意力方法。
- **[ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)** 与 **[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)**([@kijai](https://github.com/kijai))—— 稀疏注意力补丁节点的平行先例;H3 节点所遵循的显存高效注意力补丁模式来自 KJNodes 的 `MiniMaxH3MemoryEfficientSageAttentionPatch`。
- **[ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton)**([@kijai](https://github.com/kijai))—— kijai 自己的 Triton Sol-Attn 节点。v0.4.0 采用了其中验证过的模式:条件精确 KV 汇聚、经 `transformer_options["sigmas"]` 的 sigma 门控、SM89 指针内核孪生版本,以及精简的 autotune 列表。同一内核的独立实现,未共享代码。
- **[Triton](https://github.com/triton-lang/triton)**(OpenAI 及贡献者)—— 内核的编译器。
- **[SageAttention](https://github.com/thu-ml/SageAttention)**(thu-ml)—— 以上所有数字的稠密基线。

## 许可证

Apache License 2.0 —— 见 [`LICENSE`](LICENSE),继承自 NVlabs/Sana。
