# Triton Change

这个项目演示一条面向 NVIDIA Triton 服务代码迁移的工作流：

```text
旧 ONNX + 新 ONNX + 旧 ONNX 对应的 Triton 代码
=> Python 解析两个 ONNX 的架构差异
=> 把差异 JSON 和旧 Triton 代码摘要交给 LangGraph 里的 LLM
=> LLM 只生成少量局部 patch operations
=> 本地工具应用 patch，生成新 ONNX 对应的新 Triton 代码
=> 写出日志、token 用量和 PATCH_SUMMARY.md
```

项目不会下载 CUDA，也不要求实际运行 GPU/Triton Server。当前样例使用 CPU 版 ONNX Runtime 来验证生成的 ONNX 和 Triton Python backend 逻辑。

## 主要能力

- 生成一个带 `Conv1d` 和 `TransformerEncoder` 的训练/导出样例模型。
- 导出两份 ONNX：
  - `hybrid_base.onnx`：旧模型。
  - `hybrid_modified.onnx`：新模型，包含输入序列长度变化、Conv1d 通道变化、Transformer FFN 维度变化和 Cast 精度变化。
- 生成一份类似 NVIDIA Triton Python backend 的旧服务代码。
- 比较两份 ONNX，生成结构化差异 JSON。
- 通过 LangGraph 调用 OpenAI-compatible LLM，例如 DeepSeek。
- 让 LLM 基于 ONNX diff 和旧 Triton 代码生成局部 patch ops，而不是输出完整文件。
- 通过本地工具应用 patch，生成新的 Triton model repository。
- 记录节点输入/输出、LLM 输入/输出、token 用量、工具调用快照。

## 目录结构

```text
triton_change/
  pyproject.toml
  README.md
  .env.example

  src/triton_change/
    models/
      hybrid_model.py          # Conv1d + Transformer 样例模型
      train_model.py           # 简单训练骨架
      export_pair.py           # 导出 base/new ONNX，并生成旧 Triton repo

    onnx_delta/
      analyzer.py              # 解析 ONNX graph、tensor、initializer、op 统计
      diff.py                  # 生成 ONNX 架构差异
      cli.py                   # ONNX diff CLI
      schema.py                # diff 数据结构

    triton/
      repository.py            # 生成 NVIDIA 风格 Triton Python backend 样例
      patch_ops.py             # patch op 工具：复制 ONNX、regex 替换、写报告
      patcher.py               # 确定性 patcher，保留作非 LangGraph 路径
      patch_cli.py             # 确定性 patcher CLI

    langgraph_app/
      graph.py                 # LangGraph 节点编排
      nodes.py                 # 节点实现
      llm.py                   # OpenAI-compatible LLM 调用
      logging.py               # 日志、快照、token 记录
      state.py                 # LangGraph state 定义
      cli.py                   # LangGraph CLI

  skills/triton-onnx-delta/
    SKILL.md                   # 独立 Codex skill 方案
    scripts/                   # skill wrapper 脚本
    references/                # patch 策略参考

  tests/
    test_shape_utils.py
```

运行后会生成这些目录，默认不会提交到 git：

```text
artifacts/                     # 默认样例产物
artifacts_nvidia_like/          # 可选实验产物
log/                           # LangGraph 运行日志和快照
```

## 安装

```powershell
cd D:\test\mygithub\triton_change
py -m pip install -e ".[torch,langgraph]"
```

如果只想跑 ONNX diff 和确定性 patch，可以不安装 `langgraph` extra；如果要重新导出样例 ONNX，需要安装 `torch` extra。

## 生成样例 ONNX 和旧 Triton 代码

```powershell
$env:PYTHONPATH="src"
py -m triton_change.models.export_pair --out-dir artifacts
```

生成结果：

```text
artifacts/
  onnx/
    hybrid_base.onnx
    hybrid_base.onnx.data
    hybrid_modified.onnx
    hybrid_modified.onnx.data

  triton_repo/
    hybrid_text_model/
      config.pbtxt
      model.onnx
      1/
        model.py
```

样例旧 Triton `1/model.py` 里故意包含生产服务中常见的硬编码约束，例如：

```python
INPUT_TENSOR_NAME = "input_ids"
OUTPUT_TENSOR_NAME = "logits"
EXPECTED_SEQUENCE_LENGTH = 32
EXPECTED_NUM_CLASSES = 5
```

当新 ONNX 的输入长度从 `32` 变成 `40` 时，LangGraph 需要让 LLM 生成对 `1/model.py` 的局部修改，而不只是替换 ONNX。

## LangGraph 使用方法

### 1. 配置 OpenAI-compatible API

不要把 API key 写进代码或 README。用环境变量配置：

```powershell
$env:OPENAI_BASE_URL="https://api.deepseek.com"
Set-Item Env:OPENAI_API_KEY "<your-api-key>"
$env:OPENAI_MODEL="deepseek-chat"
```

也支持其他 OpenAI-compatible 服务，只要设置对应的 `OPENAI_BASE_URL`、`OPENAI_API_KEY` 和 `OPENAI_MODEL`。

如果没有设置 key，LangGraph 会走 deterministic fallback，仍然会生成 patch ops 和日志，但 token 用量是本地估算值，不是服务商返回值。

### 2. 运行 LangGraph

```powershell
$env:PYTHONPATH="src"
py -m triton_change.langgraph_app.cli `
  artifacts\onnx\hybrid_base.onnx `
  artifacts\onnx\hybrid_modified.onnx `
  --triton-model-dir artifacts\triton_repo\hybrid_text_model `
  --out artifacts\langgraph_patch `
  --mapping-rules examples\improved_triton_mapping_rules.md `
  --log-dir log `
  --run-id demo_run
```

三个核心输入：

- `base_onnx`：旧 ONNX。
- `target_onnx`：新 ONNX。
- `--triton-model-dir`：旧 ONNX 对应的 Triton model repository。
- `--mapping-rules`：可选 Markdown 规则文件，用来描述自定义 ONNX 到 Triton 代码的映射关系。

主要输出：

```text
artifacts/langgraph_patch/
  hybrid_text_model/
    config.pbtxt
    model.onnx
    hybrid_modified.onnx.data
    delta_report.json
    PATCH_SUMMARY.md
    1/
      model.py
```

## LangGraph 工作流程

当前 LangGraph 节点顺序：

```text
analyze_graphs
  -> inspect_triton_code
  -> plan_patch_ops
  -> apply_patch
  -> write_summary
```

### `analyze_graphs`

用 Python 解析两份 ONNX，生成：

- 输入/输出 tensor 名称、dtype、shape。
- initializer 形状变化。
- op count 变化，例如 `Cast` 从 `0` 变成 `2`。
- 架构标签，例如 `conv1d`、`transformer_attention`、`precision_cast`。
- compact diff JSON，供 LLM 使用。

### `inspect_triton_code`

读取旧 Triton model repository，生成代码摘要：

- 文本文件的前若干行预览。
- 文件大小和行数。
- binary 文件只记录类型和大小，不把 ONNX 二进制内容塞进 LLM。

例如会把 `1/model.py` 里的 `EXPECTED_SEQUENCE_LENGTH = 32` 暴露给 LLM。

### `plan_patch_ops`

调用 LLM，输入是：

- ONNX diff JSON。
- 旧 Triton 代码摘要。
- 可选映射规则 Markdown，例如 `examples/improved_triton_mapping_rules.md`。
- 允许的 patch operation 类型。
- fallback patch ops。

LLM 只能返回结构化 JSON，例如：

```json
{
  "patch_ops": [
    {
      "operation": "copy_target_onnx",
      "path": "model.onnx"
    },
    {
      "operation": "regex_replace",
      "path": "config.pbtxt",
      "pattern": "(name:\\s*\"input_ids\"[\\s\\S]*?dims:\\s*\\[)[^\\]]*(\\])",
      "replacement": "\\1 -1, 40 \\2",
      "count": 1,
      "reason": "Update Triton dims for input_ids."
    },
    {
      "operation": "regex_replace",
      "path": "1/model.py",
      "pattern": "EXPECTED_SEQUENCE_LENGTH\\s*=\\s*\\d+",
      "replacement": "EXPECTED_SEQUENCE_LENGTH = 40",
      "count": 1,
      "reason": "Update hardcoded Triton Python backend sequence length guard."
    }
  ]
}
```

LLM 不负责写文件，也不输出完整 `model.py`。它只输出“改哪里、怎么改”的少量指令。

## 自定义映射规则 Markdown

有时你提供的 Python 服务代码不是标准 Triton 写法，而是改进后的内部框架。例如它可能不用 `EXPECTED_SEQUENCE_LENGTH`，而是使用：

```python
MODEL_SEQUENCE_TOKENS = 32
MODEL_NUM_CLASSES = 5
```

这时可以写一个 Markdown 映射规则文件，告诉 LLM：

- ONNX 的 `input_ids` 目标 shape 最后一维对应 `MODEL_SEQUENCE_TOKENS`。
- ONNX 的 `logits` 目标 shape 最后一维对应 `MODEL_NUM_CLASSES`。
- 只需要 patch 常量，不要重写完整函数。
- internal initializer 或 Cast 变化只需要替换 ONNX 和记录报告，不要随意改 Python dtype。

示例文件：

```text
examples/improved_triton_mapping_rules.md
```

运行时传入：

```powershell
py -m triton_change.langgraph_app.cli `
  artifacts\onnx\hybrid_base.onnx `
  artifacts\onnx\hybrid_modified.onnx `
  --triton-model-dir artifacts\improved_triton_repo\hybrid_text_model `
  --mapping-rules examples\improved_triton_mapping_rules.md `
  --out artifacts\improved_langgraph_patch
```

LangGraph 会把这个 Markdown 与 ONNX diff JSON、旧 Triton 代码摘要一起写入 `plan_patch_ops` 的 LLM 输入快照。LLM 必须基于这些规则输出局部 patch ops。

### `apply_patch`

本地工具执行 patch ops：

- `copy_target_onnx`：复制新 ONNX 到 Triton repo。
- 自动复制 ONNX external data sidecar，例如 `hybrid_modified.onnx.data`。
- `regex_replace`：对 `config.pbtxt` 或 `1/model.py` 做局部替换。
- `replace_text`：做精确文本替换。
- `write_delta_report`：把完整 ONNX diff 写成 `delta_report.json`。

工具会校验 patch 路径不能逃出目标 Triton model 目录。

### `write_summary`

生成：

```text
PATCH_SUMMARY.md
```

里面包含：

- 改了哪些文件。
- 每个 patch op 的执行结果。
- 使用了哪些 ONNX 差异。
- LLM token 用量。

## 日志内容

每次运行会写入：

```text
log/<run-id>/
```

典型文件：

```text
01_node_input_analyze_graphs.json
02_node_output_analyze_graphs.json
03_node_input_inspect_triton_code.json
04_node_output_inspect_triton_code.json
05_node_input_plan_patch_ops.json
06_llm_call_make_patch_ops.json
07_node_output_plan_patch_ops.json
08_node_input_apply_patch_node.json
09_tool_call_apply_patch_ops.json
10_node_output_apply_patch_node.json
11_node_input_write_summary.json
12_tool_call_write_patch_summary.json
13_node_output_write_summary.json
```

日志会包含：

- 每个 LangGraph 节点的输入 state。
- 每个 LangGraph 节点的输出 state。
- LLM 调用的输入摘要。
- 映射规则 Markdown 内容，如果传入了 `--mapping-rules`。
- LLM 返回的 patch ops。
- LLM token 用量：
  - `input_tokens`
  - `output_tokens`
  - `total_tokens`
  - `model`
  - `provider`
- 工具调用 payload。
- 每个工具 patch 的执行结果。

日志会做基本敏感信息清理，`sk-...` 形式的 key 会被替换为 `sk-***REDACTED***`。同时 `log/` 默认在 `.gitignore` 中，不会被提交。

## 终端进度

运行 LangGraph 时，终端会显示类似：

```text
[langgraph] run_id=demo_run
[langgraph] log_dir=log\demo_run
[langgraph] analyze_graphs: start
[langgraph] analyze_graphs: done
[langgraph] inspect_triton_code: start
[langgraph] inspect_triton_code: done
[langgraph] plan_patch_ops: start
[langgraph] make_patch_ops: tokens in=3560 out=239
[langgraph] plan_patch_ops: done
[langgraph] apply_patch_node: start
[langgraph] tool apply_patch_ops: applied_patch_ops
[langgraph] apply_patch_node: done
[langgraph] write_summary: start
[langgraph] tool write_patch_summary: wrote_markdown_summary
[langgraph] write_summary: done
```

## 确定性 ONNX Diff

不使用 LangGraph 时，可以只生成 ONNX 差异：

```powershell
$env:PYTHONPATH="src"
py -m triton_change.onnx_delta.cli `
  artifacts\onnx\hybrid_base.onnx `
  artifacts\onnx\hybrid_modified.onnx `
  --out artifacts\delta.json
```

默认终端输出 compact report，`--out` 会写完整报告。

## 确定性 Triton Patch

如果不想调用 LLM，可以使用确定性 patcher：

```powershell
$env:PYTHONPATH="src"
py -m triton_change.triton.patch_cli `
  artifacts\triton_repo\hybrid_text_model `
  artifacts\onnx\hybrid_base.onnx `
  artifacts\onnx\hybrid_modified.onnx `
  --out artifacts\triton_repo_patched
```

这条路径主要用于 baseline 对照。更推荐用 LangGraph 路径测试“LLM 生成局部修改指令，本地工具执行修改”的能力。

## 测试

```powershell
$env:PYTHONPATH="src"
py -m pytest -q
```

也可以验证 Python 文件能正常编译：

```powershell
$env:PYTHONPATH="src"
py -m compileall -q src tests
```

## 安全约定

- 不要把 API key 写进仓库。
- 用环境变量传入 `OPENAI_API_KEY`。
- `artifacts/`、`artifacts_*/`、`log/`、`.env` 默认不会提交。
- LLM 只输出 patch ops，不直接写文件。
- 文件修改由本地工具完成，并限制在目标 Triton model 目录内。
