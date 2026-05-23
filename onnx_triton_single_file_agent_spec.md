# ONNX-Guided Triton 单文件代码生成 Agent Spec (v2)

> **变更说明（v2 vs v1）**
>
> - 新增 Phase -1：DeepSeek frontier model baseline，先验证 ceiling 再训练小模型
> - Diff analyzer 输出从原始 op diff 升级为「语义化变化标签」
> - Patch ops 拆出 kernel-body / kernel-meta 细粒度类型，避免 region replace 在 kernel 上退化
> - 删除脆弱奖励项「no old hardcoded shape」，新增 anti-hacking 守卫
> - MVP 切两档：MVP-easy（纯常量替换） + MVP-medium（语义改动）
> - Correctness check 强制 subprocess 沙盒
> - Trajectory 增加 `failure_class` 字段
> - Phase 3 数据生成改为「先 20 个手写精品，验证后扩 100」
> - 增补 reward shaping、错误回填策略、Triton 编译缓存等工程细节

---

## 1. 总目标

训练小模型 A，使其在只输入三个对象的情况下，生成适配新 ONNX 的 Triton 单文件 Python 代码。

**Agent 可见输入：**

```text
base.onnx
target.onnx
old_model_triton.py
```

**Agent 输出：**

```text
new_model_triton.py
```

`old_model_triton.py` 和 `new_model_triton.py` 是单文件代码，允许同时包含：

```text
import 块
@triton.jit kernel
Python wrapper
model_forward
helper functions
```

Agent 通过 LangGraph 多步分析两个 ONNX 的图差异，读取旧 Triton 单文件代码，生成 patch ops，得到新 Triton 单文件代码，并通过验证器反馈形成 trajectory，用于 frontier baseline 评估和后续 SFT / DPO / RL 训练。

---

## 2. 边界

### 本项目做

```text
ONNX diff 驱动的 Triton 单文件 Python 代码迁移
old_model_triton.py -> new_model_triton.py
基于 patch ops 的多步 agent 修复
trajectory 收集
DeepSeek frontier baseline 评估
小模型 A 后训练（SFT / DPO / RL）
```

### 本项目第一阶段不做

```text
完整编译器级 ONNX -> Triton 自动 lowering
任意大模型全图高性能 kernel 自动生成
跨多个代码文件的复杂工程重构
NVIDIA Triton Inference Server 部署配置生成
只靠 LLM 判断 correctness
直接用 latency 作为唯一 reward
```

### Agent 可见输入只有

```text
base.onnx
target.onnx
old_model_triton.py
```

### Evaluator 可使用隐藏材料

```text
test_inputs.pt
target_outputs.pt
reference_forward.py
oracle_new_model_triton.py
oracle_patch_ops.json
benchmark script
semantic_change_labels.json   # 由 diff analyzer 自动产出，可被 evaluator 校验
```

---

## 3. 任务定义

每个任务是一个 ONNX migration case：

```text
给定 base.onnx、target.onnx 和 old_model_triton.py，
生成 new_model_triton.py，
使其计算语义、输入输出 shape、dtype 和执行逻辑适配 target.onnx。
```

### 任务目录结构

```text
task_000001/
  base.onnx
  target.onnx
  old_model_triton.py

  meta.json                    # 任务元数据：变化类型 / 难度 / 目标硬件
  
  hidden_eval/
    input_specs.json
    test_inputs.pt
    reference_forward.py
    target_outputs.pt
    semantic_change_labels.json

  oracle/
    new_model_triton.py
    patch_ops.json
    diff_summary.json
```

### `meta.json` 字段示例

```json
{
  "task_id": "task_000001",
  "tier": "MVP-easy",
  "change_types": ["hidden_size_change"],
  "base_model": "ln_linear_gelu_linear",
  "target_dtype": "fp32",
  "device": "cuda",
  "min_gpu_capability": "sm_70",
  "estimated_difficulty": 1
}
```

---

## 4. 工作流程

```text
base.onnx + target.onnx + old_model_triton.py
        |
        v
ONNX diff analyzer (raw diff + semantic labels)
        |
        v
Triton single-file code analyzer
        |
        v
LangGraph agent decides patch
        |
        v
apply patch to candidate file
        |
        v
static check (AST + lint + danger scan)
        |
        v
sandboxed correctness check (subprocess)
        |
        v
optional benchmark (correctness 通过后)
        |
        v
reward + trajectory + failure_class
        |
        v
Frontier baseline / SFT / DPO / RL training
```

---

## 5. 每一步做什么

### 5.1 ONNX diff analyzer

读取 `base.onnx` 和 `target.onnx`，产出**两层 diff**：

**第一层：raw structural diff**

```text
input shape / dtype changes
output shape / dtype changes
op count changes (per op type)
op attribute changes
node-level add / remove / modify list
```

**第二层：semantic change labels**（关键改进）

预定义标签集合，由模式匹配规则集自动产出。一个 task 可命中多个标签。

```text
shape_param_change         # seq_len / hidden_size / intermediate_size 等
dtype_change               # fp32 -> fp16 / bf16 / int8
activation_change          # GELU -> BiasGELU / ReLU 等
norm_change                # LayerNorm epsilon / RMSNorm 切换
attention_pattern_change   # MHA -> GQA 等（后期）
op_fusion_change           # 是否 fuse bias add 等
unknown                    # 未命中任何规则，需要 fallback 全图 diff
```

**输出格式**

```json
{
  "raw_diff": {
    "input_shape_changes": [...],
    "op_count_delta": {"Gelu": -1, "BiasGelu": 1},
    "node_modifications": [...]
  },
  "semantic_labels": [
    {
      "label": "activation_change",
      "from": "GELU",
      "to": "BiasGELU",
      "affected_region_hint": "function:model_forward / kernel:fused_ffn"
    }
  ],
  "summary_text": "GELU 替换为 BiasGELU，hidden_size 768 -> 1024"
}
```

**验收标准**

```text
能稳定解析 base.onnx 和 target.onnx
raw_diff 是结构化 JSON，体积 < 50KB
semantic_labels 至少覆盖 MVP 的 6 类变化
diff 结果不包含原始 ONNX protobuf
未命中规则时输出 unknown 标签而非崩溃
```

---

### 5.2 Code analyzer

读取 `old_model_triton.py`，提取代码摘要。

**抽取项**

```text
imports                                       # import 块
triton_kernels[]                              # 每个 @triton.jit kernel：函数名 / 参数 / 行号 / body
python_wrappers[]                             # 普通 Python 函数
model_forward                                 # 必须存在
kernel_launch_sites[]                         # kernel[grid](args, BLOCK_SIZE=..., num_warps=...)
shape_variables                               # 推测的 seq_len / hidden_size 等绑定
output_allocations                            # torch.empty / torch.zeros 调用点
meta_parameters                               # BLOCK_SIZE / num_warps / num_stages
hardcoded_constants                           # 数字字面量及其上下文
```

**meta vs shape 常量区分启发式**

不强求完美区分，输出每个常量的类型推测（`block_meta` / `shape_param` / `numeric_const` / `unknown`）+ 置信度，让 agent 自己决定要不要改。

**验收标准**

```text
能用 ast 模块解析所有 MVP task 文件
能正确定位每个 @triton.jit 的起止行号
能完整还原 kernel_launch_sites 的 grid / BLOCK / num_warps
hardcoded_constants 列表对 MVP 任务召回率 ≥ 95%
```

---

### 5.3 LangGraph agent

Agent 根据 ONNX diff、旧代码摘要和上一步错误信息，决定下一步动作。

**允许动作**

```text
inspect_code_region        # 读取 candidate 文件特定 region 的当前内容
propose_patch              # 输出 patch ops 草稿（不落盘，便于自检）
apply_patch_ops            # 真正应用 patch
run_static_check
run_correctness_check
run_benchmark              # 仅 correctness 通过后才计分
finalize                   # 标记完成
```

**循环控制**

```text
max_steps                  # 默认 8，可配置
max_patch_attempts         # 默认 5，超出后强制 finalize
on_repeated_same_error     # 同一 error_signature 连续出现 2 次后强制改变策略
```

**错误回填策略**

每次 tool 失败，回填给 agent 的 observation 包含：

```text
error_class            # syntax / import / runtime / shape / dtype / numerical
error_message_tail     # traceback 最后 30 行
suggested_region       # static check 推断的相关 region
hint                   # 模板化提示，如 "shape mismatch: expected (B, 1024), got (B, 768)"
```

**验收标准**

```text
支持多步循环
支持失败后根据错误继续修复
每一步都能记录 observation / action / tool_result / reward
最多步数可配置且强制生效
重复错误能被识别并打破循环
```

---

### 5.4 Patch ops（细粒度版）

Agent 不直接覆盖文件，必须输出 patch ops。共 8 种类型：

**Patch 类型分级**

| 优先级 | 类型 | 适用场景 | 风险 |
|---|---|---|---|
| ★ | `update_constant` | 改 hidden_size / seq_len / eps 等单个数字常量 | 极低 |
| ★ | `update_kernel_meta` | 改 BLOCK_SIZE / num_warps / num_stages | 低 |
| ★ | `replace_function` | 替换整个 Python 函数（model_forward / wrapper） | 中 |
| ★ | `replace_kernel_body` | 替换 @triton.jit kernel 的函数体 | 中高 |
| ☆ | `replace_region` | 通用区域替换（其他工具兜底） | 中高 |
| ☆ | `insert_after_region` | 添加新 helper / wrapper | 低 |
| ☆ | `regex_replace` | 兜底文本替换，仅在 AST 无法定位时用 | 高 |
| ☆ | `full_file_replace` | 最后一档，惩罚最重 | 极高 |

**示例**

```json
{
  "operation": "update_constant",
  "path": "candidate_model_triton.py",
  "constant_name": "HIDDEN_SIZE",
  "old_value": 768,
  "new_value": 1024
}
```

```json
{
  "operation": "replace_kernel_body",
  "path": "candidate_model_triton.py",
  "kernel_name": "fused_ffn_kernel",
  "new_body": "    pid = tl.program_id(0)\n    ..."
}
```

**安全约束**

```text
patch path 必须是当前 task 的 candidate 文件
拒绝绝对路径 / .. / 任何任务目录外路径
patch 应用前先在内存中模拟，失败回滚
所有 patch ops 写入 trajectory，可审计
单次 rollout 内 patch ops 数量上限 20
```

**验收标准**

```text
八类 patch ops 都有单测
patch 失败的错误信息能反馈给 agent
patch 后 candidate 文件可被 ast 正确解析（即便语义错误）
所有 patch ops 可序列化为 patch_ops.jsonl
```

---

### 5.5 Static check

对 candidate 文件做静态检查。

**检查项**

```text
Python 语法正确（ast.parse）
import 都在白名单（torch / triton / triton.language / numpy / math / typing 等）
model_forward 存在且签名兼容
@triton.jit kernel 至少存在 1 个（如果原文件有）
kernel launch 语法合法（grid 是元组 / lambda）
无明显危险操作（os.system / subprocess / eval / exec / open(写) / requests）
candidate 文件大小 < 500KB
```

**返回**

```json
{
  "syntax_ok": true,
  "imports_ok": true,
  "model_forward_present": true,
  "kernels_present": ["fused_ffn_kernel"],
  "danger_findings": [],
  "warnings": ["unused import: math"]
}
```

**验收标准**

```text
所有语法错误能被捕获并定位行号
危险操作命中即拒绝执行 correctness check
检查耗时 < 200ms
```

---

### 5.6 Correctness check（subprocess 沙盒）

**强制要求**：必须在独立 subprocess 中运行，不允许直接 `import` candidate 文件到 agent 进程。

**沙盒约束**

```text
独立 Python subprocess
CPU/GPU 时间上限：30s（可配置）
内存上限：4GB（resource.setrlimit / cgroups）
禁用网络（unshare -n / 容器层屏蔽）
禁止读写 task 目录外文件
stdout / stderr 限制 1MB
```

**执行流程**

```text
1. 写 candidate 文件到 task_xxx/candidate_model_triton.py
2. 写 runner.py，加载 candidate + test_inputs，调用 model_forward
3. subprocess.run([python, runner.py], timeout=30)
4. 捕获输出 .pt 文件
5. 与 hidden_eval/target_outputs.pt 比对
6. 计算 max_abs_error / max_rel_error
7. 返回结构化结果
```

**比对容差**

```text
fp32: atol=1e-4, rtol=1e-4
fp16: atol=1e-2, rtol=1e-2
bf16: atol=1e-2, rtol=1e-2
```

**返回结构**

```json
{
  "executed": true,
  "shape_match": true,
  "dtype_match": true,
  "max_abs_error": 3.2e-5,
  "max_rel_error": 1.1e-5,
  "passed": true,
  "failure_class": null,
  "error_tail": null
}
```

**验收标准**

```text
能执行 candidate model_forward
能和 target_outputs.pt 对比
能输出 max_abs_error / max_rel_error
timeout 时返回 failure_class=timeout 而不是 hang
subprocess crash 时 agent 主进程不受影响
```

---

### 5.7 Benchmark（可选）

```text
correctness 通过后才执行
统计 wall time / kernel launches / torch fallback 次数
仅产出 metrics，不直接当 reward 主信号
允许第一版直接 skip
```

---

## 6. Reward 设计（含 anti-hacking 守卫）

### 6.1 主奖励（密集）

```text
syntax_pass:                        +0.10
import_pass:                        +0.10
model_forward_callable:             +0.20
output_shape_match:                 +0.30
output_dtype_match:                 +0.20
numerical_correctness_pass:         +1.00   # 主信号
benchmark_no_regression:            +0.20   # 仅 correctness 通过后才能拿
```

### 6.2 条件奖励（必须前置项满足）

```text
small_localized_patch:              +0.10
  前置：output_shape_match == True
  目的：防止"输出 no-op patch 拿小奖励"

semantic_label_addressed:           +0.20
  前置：semantic_change 标签对应的 region 确实被修改
  目的：鼓励"对症下药"
```

### 6.3 惩罚

```text
syntax_error:                       -0.40
runtime_error:                      -0.30
shape_mismatch:                     -0.50
numerical_mismatch:                 -0.70
unsafe_code:                        -1.00
oversized_patch:                    -0.10 ~ -0.50  # patch 行数 > 文件 30%
timeout:                            -1.00
patch_apply_error:                  -0.30
repeated_same_error:                -0.20  # 同一 error 连续两次
```

### 6.4 终局奖励（防 reward hacking 关键）

```text
final_all_or_nothing_bonus:         +0.50
  条件：correctness_pass == True AND patch_count <= 5 AND no_unsafe_code
  目的：防止模型在中间项上反复刷分而不去解决主问题
```

### 6.5 已删除的奖励项

```text
[DEL] no_old_hardcoded_shape: +0.2
  原因：BLOCK_SIZE 等本就是合理硬编码，难以与"残留旧 shape"区分，false positive 多
  替代：semantic_label_addressed 提供更强信号
```

### 6.6 验收标准

```text
reward 可重复（相同 candidate + 相同 task 多次跑得到相同 reward，± 数值误差忽略）
reward 来自工具验证，不来自 LLM judge
每步 step_reward 和 final_reward 都记录
各 reward 分量在 trajectory 中独立可见，便于诊断
```

---

## 7. Trajectory 格式

```json
{
  "task_id": "task_000001",
  "tier": "MVP-easy",
  "change_types": ["hidden_size_change"],
  "agent_model": "deepseek-chat",
  "steps": [
    {
      "step_idx": 0,
      "observation": {
        "onnx_diff": {...},
        "code_summary": {...},
        "last_error": null
      },
      "action": {
        "tool": "apply_patch_ops",
        "patch_ops": [
          {"operation": "update_constant", "constant_name": "HIDDEN_SIZE", "old_value": 768, "new_value": 1024}
        ]
      },
      "tool_result": {
        "static_check": "pass",
        "correctness": "fail",
        "failure_class": "shape_mismatch",
        "error_tail": "RuntimeError: shape '[1, 128, 768]' is invalid for input of size 131072"
      },
      "step_reward": -0.20,
      "reward_breakdown": {
        "syntax_pass": 0.10,
        "import_pass": 0.10,
        "shape_match": -0.50,
        "small_patch": 0.10
      },
      "done": false
    }
  ],
  "final_output_path": "task_000001/candidate_model_triton.py",
  "success": true,
  "final_reward": 1.85,
  "all_or_nothing_bonus": 0.50,
  "total_steps": 3,
  "total_patch_ops": 4
}
```

**验收标准**

```text
不记录 hidden chain-of-thought（只记录 action / tool_result）
每步都有 reward_breakdown
失败步必须填充 failure_class
能导出 SFT / DPO / RL 三种格式
单条 trajectory < 1MB
```

---

## 8. 阶段规划与验收

### Phase -1：DeepSeek Frontier Baseline（新增）

**目的**：在训练小模型之前，先用 DeepSeek 跑通整套 pipeline，建立性能 ceiling，验证 reward / evaluator / agent 框架本身没有 bug。

**用法**

```text
模型：deepseek-chat（DeepSeek V3，code 能力强且便宜）
API endpoint: https://api.deepseek.com/v1/chat/completions
环境变量：DEEPSEEK_API_KEY
本地开发临时 key：sk-7fd53cc113b341d2944a714b79349841
  （production 必须改用 .env / secrets manager，不写入仓库）
```

**要做**

```text
封装 DeepSeek client（OpenAI 兼容协议）
agent 首版直接以 deepseek-chat 作为 LLM 后端
跑完 20 个手写 task，记录 trajectory
统计 baseline 指标
```

**验收**

```text
20 个 MVP-easy + medium 任务，DeepSeek 端到端成功率 ≥ 50%
patch_ops_valid_rate ≥ 80%
平均步数 ≤ 5
所有失败案例都有可复现 trajectory
```

---

### Phase 0：任务格式与 Schema 固定

**要做**

```text
定义 task schema（meta.json / hidden_eval / oracle 目录）
定义 patch ops schema（8 种类型）
定义 trajectory schema（含 reward_breakdown / failure_class）
定义 evaluator 输入输出
搭仓库骨架 + CI（lint / unit test）
```

**验收**

```text
能手工创建 1 个完整 task（task_000001）
能加载 base.onnx / target.onnx / old_model_triton.py
schema 校验脚本通过
```

---

### Phase 1：单任务闭环（手写 patch）

**要做**

```text
ONNX diff analyzer (含 semantic labels)
code analyzer
patch applier (7 种 ops)
static check
sandboxed correctness check
reward 计算器
```

**验收**

```text
给定 task_000001，手写 patch_ops.json 能生成 new_model_triton.py
static check 通过
correctness check 通过
reward.json 输出 final_reward ≈ 2.4（满分附近）
subprocess timeout 测试通过
```

---

### Phase 2：LangGraph + DeepSeek Agent 闭环

**要做**

```text
LangGraph 状态图：observe -> propose -> apply -> check -> reward
工具调用接入（patch / static / correctness / benchmark）
错误回填策略
trajectory 记录
DeepSeek client（OpenAI 兼容）
```

**验收**

```text
DeepSeek agent 在 task_000001 上能多步完成修复
失败时能收到错误并二次修改
trajectory.jsonl 可生成
重复错误能被识别（hint 改变策略）
```

---

### Phase 3a：20 个手写精品 task

**要做**（由 AI 协助生成，人工审核）

```text
覆盖 MVP-easy（12 个）+ MVP-medium（8 个）
每个 task 都有完整 oracle + hidden_eval
所有 oracle 在本地真实跑通且数值对齐
所有 base/target ONNX 用 PyTorch -> ONNX export 可重现
脚本化生成（generate_tasks.py），可复现
```

**MVP-easy 类目（12）**

```text
seq_len_change       (×3：  128->256, 256->512, 512->1024)
hidden_size_change   (×3：  768->1024, 1024->768, 512->768)
intermediate_size    (×2：  3072->4096, 2048->3072)
ln_epsilon_change    (×2：  1e-5 -> 1e-6, 1e-12 -> 1e-5)
batch_dim_change     (×2：  static B=1 -> dynamic B)
```

**MVP-medium 类目（8）**

```text
dtype_fp32_to_fp16   (×2)
dtype_fp32_to_bf16   (×1)
GELU_to_BiasGELU     (×2)
GELU_to_ReLU         (×1)
LayerNorm_to_RMSNorm (×1)
combo: hidden + dtype (×1)
```

**验收**

```text
20 个 task 用手写 oracle 全部 correctness 通过
DeepSeek baseline 在这 20 个上 success rate ≥ 50%
每个 task 的 patch_ops.json oracle 长度 ≤ 8
generate_tasks.py 可重新跑出完全相同的 task 集
```

---

### Phase 3b：扩 100 个 task

**要做**（仅在 Phase 3a 全绿后启动）

```text
基于 Phase 3a 模板批量化生成 base/target ONNX 配对
用 Phase 3a oracle 模式作为生成函数
人工抽检 20%
```

**验收**

```text
100 个任务可被 evaluator 跑通
≥ 90% oracle 样本 correctness 通过
DeepSeek baseline 在 100 个上 success rate ≥ 40%
难度分布合理（easy:medium:hard ≈ 5:3:2）
```

---

### Phase 4：SFT

**要做**

```text
oracle patch_ops + DeepSeek 高分 trajectory 转成 SFT 数据
训练小模型 A（建议 7B 量级，code-base 模型）
SFT 数据格式：(observation -> action) pairs
```

**验收**

```text
patch_ops_valid_rate (JSON 合法率) 提升到 ≥ 90%
static check pass rate 提升 ≥ 30 pp
单步 patch 成功率高于 base model
小模型在 20 个手写 task 上 success rate ≥ 30%
```

---

### Phase 5：DPO

**要做**

```text
每个 task 用 SFT 模型采样 K=8 个候选 patch
用 evaluator 排序得到 chosen / rejected
DPO 训练
```

**验收**

```text
correctness pass rate 比 SFT 提升 ≥ 5 pp
oversized_patch_rate 下降
runtime_error 比例下降
```

---

### Phase 6：Online RL

**要做**

```text
模型 A 通过 LangGraph rollout
evaluator 给 reward
GRPO / RLOO / PPO（推荐 GRPO，便宜）
监控各 reward 分量均值漂移防 hacking
```

**前置条件**

```text
task 数 ≥ 500（不足时先扩数据）
单步 rollout cost < 30s（含 Triton 编译）
有稳定的 GPU 资源
```

**验收**

```text
end-to-end success rate 比 DPO 提升 ≥ 5 pp
平均修复步数下降
final_reward 分布右移
no reward hacking：各 reward 分量比例不出现极端漂移
```

---

## 9. MVP 定义（双档）

### MVP-easy：纯常量 / meta 替换

**模型结构**

```text
Input -> LayerNorm -> Linear -> GELU -> Linear -> Output
```

**支持变化**

```text
seq_len change
hidden_size change
intermediate_size change
LayerNorm epsilon change
batch dim static -> dynamic
```

**预期 patch ops**

```text
主要使用 update_constant
偶尔 update_kernel_meta
不需要 replace_kernel_body
```

**验收**

```text
12 个 task 全部跑通
DeepSeek agent 成功率 ≥ 70%
SFT 后小模型成功率 ≥ 50%
```

### MVP-medium：语义改动

**支持变化**

```text
fp32 -> fp16 / bf16
GELU -> BiasGELU / ReLU
LayerNorm -> RMSNorm
hidden_size + dtype 组合
```

**预期 patch ops**

```text
需要 replace_kernel_body 或 replace_function
可能伴随 update_constant
```

**验收**

```text
8 个 task 全部跑通
DeepSeek agent 成功率 ≥ 30%
SFT 后小模型成功率 ≥ 20%
```

---

## 10. 成功指标

### 核心指标

```text
patch_ops_valid_rate            # patch JSON 合法率
static_check_pass_rate
correctness_pass_rate           # 主指标
end_to_end_success_rate         # 多步内任意一次成功
avg_steps_to_success
unsafe_patch_rate
oversized_patch_rate
repeated_error_rate             # 反映回填策略效果
semantic_label_hit_rate         # 反映"对症下药"能力
```

### 各阶段目标

| 阶段 | end-to-end success | 备注 |
|---|---|---|
| Phase -1 DeepSeek baseline | ≥ 50% (20 tasks) | 验证框架 |
| Phase 3a | ≥ 50% (20 tasks) | 同上口径 |
| Phase 3b | ≥ 40% (100 tasks) | 难度更广 |
| Phase 4 SFT | ≥ 30% on 20 tasks (small model) | 7B 模型 |
| Phase 5 DPO | +5 pp over SFT | |
| Phase 6 RL | +5 pp over DPO | |

### Anti-hacking 监控

```text
reward 分量比例分布（每周快照）
oversized patch 比例不能上升
unsafe code 必须 0
```

---

## 11. 工程细节

### 11.1 目录结构

```text
triton_change/
  spec/
    onnx_triton_single_file_agent_spec.md   # 本文件
  schemas/
    task_schema.json
    patch_ops_schema.json
    trajectory_schema.json
  src/
    diff_analyzer/
    code_analyzer/
    patcher/
    sandbox/
    evaluator/
    agent/                # LangGraph
    llm_clients/
      deepseek_client.py
  tasks/
    task_000001/ ...
  trajectories/
  scripts/
    generate_tasks.py
    run_baseline.py
    run_agent.py
  tests/
  .env.example
```

### 11.2 DeepSeek 调用

```text
SDK：openai 库（DeepSeek 100% 兼容 OpenAI Chat Completions）
base_url: https://api.deepseek.com/v1
model: deepseek-chat (V3)，需要更强推理时切 deepseek-reasoner
温度：propose 阶段 0.7，重试阶段 0.3
max_tokens：4096
超时：60s，失败重试 3 次（指数退避）
所有 call 记录到 trajectory 的 llm_call_log（计费审计）
```

### 11.3 Triton 编译缓存

```text
环境变量 TRITON_CACHE_DIR 指向独立缓存目录
跨 task 共享，加速重复 kernel 编译
RL 阶段尤其关键，可省 50%+ 时间
```

### 11.4 安全

```text
API key 走环境变量 + .env（已加 .gitignore）
candidate 文件强制 subprocess 沙盒执行
patch 路径白名单校验
unsafe import 黑名单（os, sys, socket, subprocess from candidate）
```

### 11.5 可复现

```text
所有随机过程固定 seed
generate_tasks.py 输出可重现的 task 集
trajectory 记录 git_commit_sha + agent_model 版本
```

---

## 12. 路线图（time estimate）

| 阶段 | 工期 | 关键产出 |
|---|---|---|
| Phase 0  | 3 天 | schemas + 仓库骨架 |
| Phase 1  | 1 周 | 单任务闭环（手写 patch） |
| Phase 2  | 1 周 | DeepSeek + LangGraph agent |
| Phase 3a | 1.5 周 | 20 个手写精品 task + baseline 数据 |
| Phase 3b | 2 周 | 扩 100 task |
| Phase 4  | 1 周 | SFT |
| Phase 5  | 1.5 周 | DPO |
| Phase 6  | 3 周 | Online RL |
| **合计** | **~12 周** | 端到端 v1 |

---

## 13. 不变式（开发期间任何 PR 都不得违反）

```text
1. Agent 可见输入永远只有 base.onnx / target.onnx / old_model_triton.py
2. correctness 永远来自工具验证，不来自 LLM judge
3. candidate 文件永远在 subprocess 沙盒执行
4. trajectory 永远不记录 hidden chain-of-thought
5. patch 永远不能写到 task 目录之外
6. API key 永远不入仓库
7. 任何 reward 项的添加 / 修改 / 删除都必须有 anti-hacking 论证
```
