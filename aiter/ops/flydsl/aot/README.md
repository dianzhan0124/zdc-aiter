# FlyDSL MoE AOT (co-file) 交付与使用手册

> **代码分两个分支交付**：
> - `zdc/flydsl-moe-aot` — 要合入 aiter 的 serve 侧运行时（`co_abi.py` / `flydsl_aot_runtime.py` / `__init__.py` + `moe_kernels.py` 派发 + launcher 名字戳）。
> - `zdc/flydsl-aot-tools`（**本分支**）— 离线产 co 的工具（`build_flydsl_aot.py` / `gate_dispatch_unique.py`）+ 本手册。不用交付给最终 serve 环境。
>
> 本目录（`aiter/ops/flydsl/aot/`）在两个分支里是同一路径：运行时文件随 aiter 发布，工具文件只在产 co 时用。
>
> 注：`build_flydsl_aot.py` / `gate_dispatch_unique.py` 是**面向 flydsl 的通用 AOT 工具**（从 flydsl 缓存抽 .co），并非 MoE 专用；本手册以 MoE kernel 的交付为例。而 `moe_kernels.py` 派发分支与 launcher 名字戳才是 MoE 专用（在 zdc/flydsl-moe-aot）。

把 FlyDSL MoE kernel 从「serve 时 JIT 编译」改为「预编译成 code object (.co)，serve 时用 HIP 直接加载」。
目标：上游 (vllm + aiter) 在 serve 阶段**不再依赖 flydsl 编译器**，只需一批 .co 文件 + 本目录的纯 ctypes 运行时。

---

## 0. 一句话全景

- **产出**：一个 `drop/` 目录 = 若干 `.co` (ELF) + 一个 `index.json`（记录每个 kernel 的符号名、grid/block 表达式、kernarg ABI、shared_bytes、来源等）。
- **消费**：serve 进程里 aiter 的 stage1/2 派发层，按 kernel 符号名从 drop 里取 .co，用 ctypes + libamdhip64 启动。命中则走 AOT，未命中回退 JIT。
- **门控**：一个环境变量 `AITER_FLYDSL_AOT_DROP` 打开；`AITER_FLYDSL_AOT_RUN_ONLY=1` 强制纯 AOT（缺件即报错，不允许静默 JIT）。

---

## 1. 目录内容 · 哪些必须交付，哪些不用

| 文件 | 作用 | 交付时 | 依赖 |
|---|---|---|---|
| `flydsl_aot_runtime.py` | 运行时：`AotDrop`（按名/来源解析）、`AotKernel`（ctypes 启动）。 | **必须随 aiter 一起发**（serve 时被 import） | 无 flydsl/torch，只用 stdlib + libamdhip64 |
| `co_abi.py` | 从 .co 读 kernarg ABI（偏移/大小/对齐），供打包 kernarg。 | **必须随 aiter 一起发** | 无 |
| `__init__.py` | 包标记 | **必须随 aiter 一起发** | 无 |
| `build_flydsl_aot.py` | 从 flydsl 缓存抽取 .co、解析 host wrapper 得到 grid/block/kernarg，产出 `drop/` + `index.json`。 | 只在**产 co 阶段**用，不用给最终使用方 | 需在容器内（读缓存） |
| `gate_dispatch_unique.py` | 交付门：校验同一符号名不会映射到多个无法区分的 .co。 | 只在**产 co 阶段**用，不用给最终使用方 | 无 |
| `README.md` | 本手册 | 随源码 | 无 |

**一句话**：前三个是「运行时」，是 `moe_kernels.py` 在 serve 时真正 import 并调用的代码，必须跟着部署的 aiter 一起在。后两个是「制作工具」，只在你离线产 co 时跑一次，产完 drop 后最终 serve 环境既不需要它们、也不需要 flydsl 编译器。

---

## 2. 交付一个模型的 co：完整流程

> **和路径 A 的区别（重要）**：路径 A 是「run-only 缓存」交付——发一份剥源码的 flydsl 缓存，serve 时仍走 flydsl 的加载器。路径 B（本目录）是「纯 .co + ctypes」交付——发 `drop/` 目录，serve 时不碰 flydsl。

**产 co 分两段，无论用哪种方式都一样**：

```
① 把 kernel 编进 flydsl 缓存  →  ② build_flydsl_aot.py 从缓存抽出 .co（drop/）
```

第 ① 段有两种走法（下面 2.A / 2.B），第 ② 段完全共用。

### 产 co 时的环境变量

| 变量 | 值 | 作用 | 哪种方式需要 |
|---|---|---|---|
| `FLYDSL_RUNTIME_CACHE_DIR` | 一个**全新空目录**，如 `/tmp/fc_csv` | JIT 缓存写这里，co 从这里抽 | 两种都要 |
| `ARCH`（或 `GPU_ARCHS`） | `gfx950` | 目标架构 | 两种都要 |
| `AITER_SITUV2_A8W4` | `1` | 让 **serve 运行时**走 A8W4（a=fp8,b=fp4）dispatch 分支。若目标是 a4w4 则改设 `AITER_SITUV2_A4W4=1` | **只有方式 B（跑 serve）要**；CSV 方式**不需要** |

> **为什么 CSV 方式不需要 `AITER_SITUV2_A8W4`**：这个开关是 serve 运行时 `fused_moe.py` 用来选走哪条 mixed_moe 分支的。CSV 编译器 `aiter.aot.flydsl.moe` **不读**它——每个 kernel 的 `a_dtype`/`b_dtype` 是从 CSV 的列（`q_type`/`dtype`/`q_dtype_w`）和 kernel 名解析出来的，A8W4 这个组合就体现在 CSV 那些行里。只有方式 B 真的在跑 serve，才要靠这个开关让 serve 走 A8W4 路径去 JIT 出对应 kernel。

---

### 2.A 用 CSV 编译（推荐，最方便）

aiter 自带一个 CSV 驱动的 AOT 编译器 `aiter.aot.flydsl.moe`，直接读 tuned CSV 把每一行展开成 stage1+stage2+epilogue 的编译 job 写进缓存——**不用起 serve、不用发请求**。

```bash
export FLYDSL_RUNTIME_CACHE_DIR=/tmp/fc_csv
export ARCH=gfx950
# 注意：CSV 方式不需要 AITER_SITUV2_A8W4——dtype 从 CSV 列里读，见上面的说明

# --csv 可以接多个文件；每行一个 (token, model_dim, inter_dim, expert, topk) 组合
python3 -m aiter.aot.flydsl.moe --csv /path/to/dsv3_fp4_tuned_fmoe.csv
```

CSV 需要的列（用 `csv.DictReader` 读表头）：
- 必需：`token, model_dim, inter_dim, expert, topk`
- 可选：`doweight_stage1, cu_num, block_m, act_type, q_type, dtype, q_dtype_w`

直接用你调优产出的 tuned CSV 即可，列名一致。

> **CSV 方式的唯一注意点**：编什么就有什么 co——CSV 漏一行，就少一个 co，serve 时那个 shape 会回退 JIT（run-only 模式则报错）。所以 CSV 的完整性由你负责。稳妥做法见 2.C 的覆盖校验。

### 2.B 空缓存跑一遍 serve（覆盖面天然精确）

不手写清单，靠「服务实际请求什么，缓存里就有什么」。适合你不确定 CSV 是否列全、或没有现成 CSV 的场景。

```bash
export FLYDSL_RUNTIME_CACHE_DIR=/tmp/fc_jit_ref
export AITER_SITUV2_A8W4=1
export ARCH=gfx950
# ... 正常起这个模型的 serve，打一批覆盖各 seqlen / M 的请求，
#     让 stage1/2 会遇到的每个 shape 都真的跑到 ...
# 跑完 /tmp/fc_jit_ref 就是这个模型的「need 集合」。
```

### 2.C 两种走法对比

| | 2.A CSV | 2.B 跑 serve |
|---|---|---|
| 触发 | `moe.py --csv config.csv` | 起 serve 打请求 |
| 覆盖面靠 | CSV 里列了哪些行 | 请求真打到各 shape |
| 优点 | 方便、可复现、不用起服务 | need 集合天然精确、不会漏 |
| 风险 | CSV 漏行 → 少 co | 要保证请求覆盖全 |

**最稳做法**：用 2.A 的 CSV 产 co（方便），再用 2.B 跑一次空缓存 serve 得到 need 集合，然后跑覆盖校验确认 CSV 没漏：

```bash
# k3run/aot 里现成的工具：need=serve缓存，have=csv缓存
python3 coverage_audit.py --need /tmp/fc_jit_ref --have /tmp/fc_csv
```

---

### 2.② 从缓存抽 .co 产 drop（两种走法共用）

```bash
cd aiter/ops/flydsl/aot
python3 build_flydsl_aot.py /tmp/fc_csv -o /tmp/drop --arch gfx950
python3 gate_dispatch_unique.py /tmp/drop
```

- 参数：`build_flydsl_aot.py <缓存目录> -o <输出drop> --arch <目标架构>`
- `--arch gfx950` **必须指定**：gemm 预热会把 gfx1250/gfx942 也编进同一缓存，混着交付会在目标机加载失败。非目标架构的产物会被丢弃、不进 drop。
- 产物：
  - `/tmp/drop/<符号名>.<sha12>.co`：每个 kernel 的 code object
  - `/tmp/drop/index.json`：`{kernels: {符号名: [ {co, co_sha256_12, shared_bytes, grid, block, kernel_args, wrapper_arg_types, ...} ]}}`
- `gate_dispatch_unique.py` 通过 = 每个符号名都能被 `(name[, shared_bytes])` 唯一解析。失败意味着同名有两个不同 .co 且 shared_bytes 也无法区分——派发时无从选择，会加载错 kernel，用一次干净单模型预热重产即可。

**`/tmp/drop` 就是交付物**（含 `index.json` 和一堆 `.co`）。把它连同带 aot 运行时的 aiter 一起交出去。

### 2.③（可选，强烈建议）全套关卡 + 清单

kimi_k3 的 `k3run/aot/make_delivery.sh` 把上面几步串成 6+ 道关卡并产 `DELIVERY.json`（记录验了什么、在什么 ROCm/aiter 栈上验的），关卡包括：覆盖闭包（预热必须包含 need 的每一条）、剥离源码、剥离后覆盖不变、ABI 自洽 + 本机加载、派发唯一性、冻结语料回放。示例：

```bash
NEED=/tmp/fc_jit_ref CACHE=/tmp/fc_csv ARCH=gfx950 OUT=/tmp/delivery \
  bash make_delivery.sh
# 产物 /tmp/delivery/{drop, cache, corpus, DELIVERY.json}
```

本分支带核心两件工具（build + gate）；CSV 编译器是 aiter 自带的 `aiter.aot.flydsl.moe`；完整关卡脚本在 kimi_k3 侧。

---

## 3. serve 时如何使用 co

把 `drop/` 放到部署机某路径，设两个环境变量即可。派发层已内建在 `aiter/ops/flydsl/moe_kernels.py` 的 stage1/2 里：

```bash
# 打开 AOT：命中就用 .co，未命中回退 JIT（安全默认）
export AITER_FLYDSL_AOT_DROP=/path/to/drop

# 可选：纯 AOT 模式。任何缺件/歧义直接报错，绝不静默 JIT（用于验收交付完整性）
export AITER_FLYDSL_AOT_RUN_ONLY=1
```

不设 `AITER_FLYDSL_AOT_DROP` 时，行为与改动前**完全一致**（纯 JIT），零影响，可灰度。

**部署方需要满足的两个前提**（缺一不可）：
1. 部署的 aiter 里带着本目录的 aot 运行时（`flydsl_aot_runtime.py` / `co_abi.py` / `__init__.py`）；
2. 设了 `AITER_FLYDSL_AOT_DROP` 指向 drop 目录。

满足后就是「co 直接放目录，moe_kernels.py 自动调用」，制作工具（build/gate）无需交付、目标机也无需 flydsl 编译器。

### 3.1 派发是怎么工作的

1. stage1/2 编译（或取缓存）得到 launcher `exe`，其上带有 `exe._aot_module_name`——就是 `@flyc.kernel(name=...)` 的符号名（编译期唯一真源，无字符串复制、无漂移）。
2. `_maybe_run_aot(exe, args)` 用该名字 `AotDrop.get(name)` 取到 `AotKernel`，直接 `k(*args)`：args 元组与 JIT 启动器一致，`AotKernel` 按 .co 自带 ABI 打包 kernarg 后 ctypes 启动。
3. 命中返回 True；未命中：run-only 模式报错，否则告警一次并回退 `_run_compiled`（JIT）。

### 3.2 什么是「ctypes 运行时」

运行时（`flydsl_aot_runtime.py` + `co_abi.py`）加载和启动 kernel 时**不经过 flydsl 编译器**，而是用 Python 标准库的 `ctypes` 直接调 HIP 的 C API：

- `ctypes.CDLL("libamdhip64.so")` 加载 ROCm 的 HIP 运行库；
- ctypes 调 `hipModuleLoad`（读 .co 文件）、`hipModuleGetFunction`（按符号名取 kernel）、`hipModuleLaunchKernel`（启动）；
- kernel 参数（kernarg）按 .co 自带的 ABI（偏移/大小/对齐，由 `co_abi.py` 从 ELF 里读出）在内存里打包好，以裸指针传给 HIP。

**为什么强调这点**：整个加载器只依赖 Python 标准库 + `libamdhip64.so`（装了 ROCm 就有），**不 import flydsl、不 import torch 的编译路径**。所以交付到客户机时客户无需装 flydsl 编译器——这正是路径 B 的核心价值。对比之下 JIT 路径要靠 flydsl 把 kernel 编出来再启动。一句话：**ctypes 运行时 = 一个纯 Python + HIP C API 的极薄加载器，把预编译好的 .co 直接拉起来跑，绕开编译器。**

### 3.3 打进 Docker 交付给客户

co 就是交付物本身，必须交付给客户；客户 serve 时 aiter 读的就是这些 co。最省心的做法是把 `drop/` 打进镜像、路径在镜像里固定死并设好环境变量，客户 `docker run` 起来就开箱即用、连环境变量都不用管：

```dockerfile
# 基于已有的 vllm+aiter 镜像（aiter 里已含 aot 运行时那三个文件）
COPY drop/ /opt/flydsl_drop/
ENV AITER_FLYDSL_AOT_DROP=/opt/flydsl_drop
# 可选，纯 AOT 验收（缺件直接报错，不静默回退 JIT）：
# ENV AITER_FLYDSL_AOT_RUN_ONLY=1
```

镜像里要同时具备两样（缺一不可）：
1. `drop/` 目录（co 文件）——你 `COPY` 进去；
2. 带 aot 运行时的 aiter（`flydsl_aot_runtime.py` / `co_abi.py` / `__init__.py`）——通常已在你 build 镜像用的 aiter 里，不用单独拷。

co 放镜像里**任意目录**都行，没有固定路径要求，关键是 `AITER_FLYDSL_AOT_DROP` 指过去。若想让客户可换 drop，也可以不 COPY、把 drop 挂成 volume 让客户自己设该变量。客户机上**不需要** flydsl 编译器，也不需要 build/gate 工具。

---

## 4. 上游 (aiter) 如何适配

已完成的最小改动（本分支）：

1. **vendored 运行时**：新增 `aiter/ops/flydsl/aot/`（本目录），serve 侧加载 .co 不再需要 flydsl。
2. **名字通道**：在 4 个（本树 2 个）launcher return 处，把已算出的 `module_name` 盖到 launcher 上作 `_aot_module_name`。复用编译器自己的符号名，杜绝命名漂移。
3. **派发分支**：`moe_kernels.py` 两个 `_run_compiled(exe, args)` 站点前插入 `if not _maybe_run_aot(exe, args): _run_compiled(exe, args)`，env 门控。

对上游的要求：仅需把本目录随包发布、在 serve 环境设好 `AITER_FLYDSL_AOT_DROP`。无门控时旧行为不变。

---

## 5. 验证记录

在 MI355X (gfx950) 容器内：

- `validate_aot.py --rounds 2`：stage1+stage2 各 shape **8/8 通过，逐位同=True，最大差=0**（by_source 通道，证明 ctypes AOT 加载正确）。
- 名字通道等价性：`AotDrop.get(name)` 与已验证的 `by_source(ident)` 对全部 kernel 解析到**同一 co 对象**，正确性传递保证。
- `_aot_module_name` 确实粘在 `@flyc.jit` launcher 上。
- 4 条 fail-safe 语义：无 env→JIT；命中失败→回退；run-only 缺件→报错；run-only 无名→报错。

---

## 6. 已知边界

- serve 配置为 **A8W4**（a_dtype=fp8, b_dtype=fp4）；a16w4 (bf16×fp4) 路径的名字戳/派发是 dtype 无关的，逻辑上可用，但逐 bit 未单独验证。
- `build_flydsl_aot.py` 需在能读 flydsl 缓存的容器内运行；`flydsl_aot_runtime.py`/`co_abi.py` 则完全无编译器依赖，可在纯净 serve 环境运行。
- need 集合靠「空缓存 JIT 跑一遍该模型」自然得到，不维护 kernel name 清单；换模型/换 shape 覆盖面就重跑 2.1 取新的 need 集合。
