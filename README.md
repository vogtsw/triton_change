# Triton Change

ONNX-guided Triton single-file kernel migration agent.

**Goal**: train a small model that, given `(base.onnx, target.onnx, old_model_triton.py)`,
produces `new_model_triton.py` whose `model_forward` matches the semantics of
`target.onnx` — through multi-step LangGraph patch ops with validator-based reward.

The full design lives in [`onnx_triton_single_file_agent_spec.md`](onnx_triton_single_file_agent_spec.md).

---

## Status

| Phase | Description | Status |
|---|---|---|
| Phase 0 | Schemas + repo skeleton | Done |
| Phase -1 | DeepSeek frontier baseline plumbing | Done (client + smoke) |
| Phase 1 | Patcher + static check + correctness check + reward | **Done** (60 tests, e2e demo: reward 2.10) |
| Phase 2 | LangGraph-style multi-step agent + DeepSeek policy | **Done** (28 tests, 5-task baseline run) |
| Phase 3a | 20 handcrafted tasks | **Done** (12 easy + 8 medium) |
| Phase 3b | Expand to 100 tasks (easy:medium:hard ≈ 5:3:2) | **Done** |
| Phase 4 | SFT export (observation→action) | **Done** (`scripts/export_sft.py`) |
| Phase 5 | DPO pairs (K=8 sampling + oracle judge) | **Done** (`scripts/build_dpo_pairs.py`) |
| Phase 6 | LangGraph rollout + GRPO/RLOO/PPO scaffolding | **Done** (task<500; no GPU trainer) |

### Phase 1 acceptance evidence

| Component | Verification |
|---|---|
| `patcher.py` (8 op types + path safety) | 21 unit tests + 5-task parameterized oracle round-trip |
| `static_check.py` (AST + danger scan) | 10 unit tests |
| `correctness.py` (subprocess sandbox) | 6 unit tests with synthetic torch-only candidates |
| `reward.py` (anti-hacking composition) | 11 unit tests; perfect-run scenario lands at 2.70 |
| `run_phase1.py` (e2e driver) | 5/5 tasks pass on Windows CPU via `cpu_demo_patch_ops.json` (reward 2.10) |

The `cpu_demo_*` files exist purely so the pipeline can be validated end-to-end
on machines without OpenAI Triton. Real Triton runs (the 2.70-reward path)
require Linux + CUDA + `triton>=2.1`.

### Phase 2 acceptance evidence

| Spec requirement | Verification |
|---|---|
| LangGraph-style state graph (observe -> propose -> apply -> check -> reward) | `agent/runner.py` — pure-Python loop; trivially wrappable in LangGraph |
| Tool calls: inspect / apply / static / correctness / benchmark / finalize | `agent/tools.py` (10 unit tests) |
| Error feedback: failure_class + error_tail + hint into next observation | `agent/observation.py::failure_hint` (9 unit tests) |
| Repeated-error breaker | `error_signature` + `max_repeated_errors` (test_repeated_same_error_terminates) |
| Trajectory.jsonl with reward_breakdown / failure_class | `agent/trajectory.py`; schema-validated |
| DeepSeek agent multi-step repair on task_000001 | `scripts/run_agent.py` real run: 3 steps, surgical patches, schema-valid |
| 5-task baseline run | `scripts/run_phase2_baseline.py --policy deepseek` — see `trajectories/baseline_deepseek.md` |

### Phase 3a acceptance evidence (20 handcrafted tasks)

| Category | Tasks | Patch style |
|---|---|---|
| hidden_size_change (×3) | 000001–000003 | update_constant |
| intermediate_size (×2) | 000004–000005 | update_constant |
| seq_len_change (×3) | 000006–000008 | update_constant SEQ_LEN |
| ln_epsilon_change (×2) | 000009–000010 | update_constant LN_EPS |
| batch_dim_change (×2) | 000011–000012 | replace_function + regex |
| dtype fp32→fp16 (×2) | 000013–000014 | update_constant DTYPE_NAME |
| dtype fp32→bf16 (×1) | 000015 | update_constant DTYPE_NAME |
| GELU→BiasGELU (×2) | 000016–000017 | regex + replace_kernel_body |
| GELU→ReLU (×1) | 000018 | replace_kernel_body |
| LayerNorm→RMSNorm (×1) | 000019 | replace_kernel_body |
| combo hidden+dtype (×1) | 000020 | update_constant ×3 |

All 20 tasks: schema-valid, oracle round-trip AST match, cpu-demo Phase 1/2 pass (reward 2.10).

Regenerate anytime:

```powershell
py scripts\generate_tasks.py                    # all 100 tasks
py scripts\generate_tasks.py --from 21 --to 100 # Phase 3b only
py scripts\eval_patch_oracle.py --from 1 --to 100
py scripts\run_phase2_baseline.py --policy oracle --cpu-demo
py scripts\export_sft.py trajectories\baseline_oracle.jsonl
py scripts\build_dpo_pairs.py
py scripts\run_rl_rollout.py --policy oracle --cpu-demo --from 1 --to 5
```

### Phase 3b–6 (100 tasks + data prep + RL scaffolding)

| Item | Detail |
|---|---|
| Task catalog | 100 tasks: 50 easy / 30 medium / 20 hard (`scripts/task_catalog.py`) |
| Oracle judge | AST match vs oracle — no DeepSeek / no GPU (`scripts/eval_patch_oracle.py`) |
| SFT export | trajectory JSONL → observation→action pairs (`scripts/export_sft.py`) |
| DPO pairs | K=8 perturbations ranked by oracle judge (`scripts/build_dpo_pairs.py`) |
| RL scaffolding | LangGraph wrapper + GRPO/RLOO/PPO math (`src/triton_change/training/`) |

Evaluation uses **oracle patch correctness** (apply + AST match), not live DeepSeek or GPU Triton runs.

---

## Layout

```text
schemas/                          JSON Schemas
  task_schema.json
  patch_ops_schema.json
  trajectory_schema.json

tasks/                            Handcrafted migration tasks
  task_000001/                    hidden 768->1024, intermediate 3072->4096   (2 ops)
  task_000002/                    hidden 768->512,  intermediate 3072->2048   (2 ops)
  task_000003/                    intermediate-only 3072->4096                 (1 op)
  task_000004/                    LayerNorm eps 1e-5 -> 1e-6                   (1 op)
  task_000005/                    combo: hidden+intermediate+eps               (3 ops)

    Each task contains:
      meta.json
      old_model_triton.py
      oracle/new_model_triton.py            gold-standard Triton output
      oracle/patch_ops.json                 surgical update_constant ops
      oracle/diff_summary.json
      oracle/cpu_demo_new_model_triton.py   torch-only (Windows demo)
      oracle/cpu_demo_patch_ops.json        full_file_replace using the above
      hidden_eval/input_specs.json
      hidden_eval/semantic_change_labels.json
      hidden_eval/reference_forward.py      PyTorch ground truth
      base.onnx, target.onnx                       (generated)
      hidden_eval/weights.pt, test_inputs.pt, target_outputs.pt   (generated)

src/triton_change/                Python package
  __init__.py
  patcher.py                      8 patch op types, path-safe
  static_check.py                 AST + import whitelist + danger scan
  correctness.py                  subprocess sandbox + tolerance comparison
  reward.py                       anti-hacking reward composition
  agent/
    observation.py                CodeSummary + Observation + hint templates
    tools.py                      Unified ToolResult wrappers
    policy.py                     PolicyBase + Mock / Oracle / DeepSeek policies
    runner.py                     observe -> act -> tool -> reward loop
    trajectory.py                 JSONL writer + schema validation
  llm_clients/
    deepseek_client.py            OpenAI-compatible DeepSeek client

scripts/
  validate_task.py                Schema-validate a task directory
  generate_tasks.py               20-task generator (Phase 3a catalog)
  task_profiles.py                Profile templates (shape/seq/batch/dtype/activation/norm)
  build_cpu_demo_patches.py       Emit cpu_demo_patch_ops.json for any task
  run_phase1.py                   End-to-end Phase 1 driver
  run_agent.py                    Phase 2 single-task agent driver
  run_phase2_baseline.py          Phase 2 multi-task baseline (oracle / deepseek)
  smoke_test_deepseek.py          DeepSeek API smoke test

tests/
  test_patcher.py                 21 tests + 20-task parameterized oracle round-trip
  test_static_check.py            10 tests — syntax/imports/danger scan
  test_correctness.py             6 tests — sandbox + numerical/shape/timeout failure classes
  test_reward.py                  11 tests — every reward branch
  test_schemas.py                 12 tests — JSON schema conformance
  test_agent_observation.py       9 tests — code summary + hint templates
  test_agent_tools.py             10 tests — tool wrappers
  test_agent_runner.py            9 tests — full loop (mock + oracle policies)
```

---

## Setup

```powershell
cd D:\test\mygithub\triton_change
py -m pip install -e ".[dev,torch,dotenv]"
copy .env.example .env
# edit .env, set DEEPSEEK_API_KEY=sk-...
```

Optional: install Triton (Linux + CUDA only; on Windows you can develop the
patcher and static check on CPU, but `correctness_check` requires Triton at runtime).

```bash
py -m pip install -e ".[triton-runtime]"
```

---

## Quick start

### 1. Generate all 5 tasks' dynamic data

```powershell
$env:PYTHONIOENCODING="utf-8"
py scripts\generate_tasks.py
```

This writes `base.onnx`, `target.onnx`, `weights.pt`, `test_inputs.pt`,
`target_outputs.pt` and refreshes the static .py / .json files for every task.

### 2. Validate all task directories

```powershell
foreach ($id in 1..5) { $tid = "task_{0:D6}" -f $id; py scripts\validate_task.py "tasks\$tid" }
```

### 3. Run the test suite

```powershell
$env:PYTHONPATH="src"
py -m pytest -v
```

Expected: 55+ tests pass.

### 4. Run Phase 1 end-to-end

On Linux + GPU + Triton (canonical path, expected reward ~ 2.70):

```bash
python scripts/run_phase1.py tasks/task_000001 --device cuda
```

On Windows / CPU (demo path, expected reward 2.10):

```powershell
py scripts\run_phase1.py tasks\task_000001 --patch-ops tasks\task_000001\oracle\cpu_demo_patch_ops.json
```

### 5. Run the Phase 2 multi-step agent

Oracle policy (sanity check, replays oracle/patch_ops.json):

```powershell
py scripts\run_agent.py tasks\task_000001 --policy oracle
```

DeepSeek policy (real LLM driving the loop):

```powershell
py scripts\run_agent.py tasks\task_000001 --policy deepseek --max-steps 5
```

5-task baseline (oracle or deepseek):

```powershell
py scripts\run_phase2_baseline.py --policy oracle --cpu-demo
py scripts\run_phase2_baseline.py --policy deepseek --max-steps 6
```

Outputs: `tasks/<task>/trajectory.json`, `trajectories/baseline_<policy>.{jsonl,md}`.

### 6. Smoke-test the DeepSeek client

```powershell
py scripts\smoke_test_deepseek.py
```

Expected: response `'pong'`, ~30 tokens, ~3 seconds.

---

## Design summary

The agent operates under three hard constraints:

1. **Agent-visible inputs are exactly three files** (`base.onnx`, `target.onnx`, `old_model_triton.py`). Nothing else.
2. **Correctness comes from validators, never from LLM judge** — `static_check`, sandboxed `correctness_check`, and benchmark.
3. **Patch ops, not full-file rewrites** — 8 op types ranked by surgical precision, anti-hacking reward design.

Reward components, semantic-change labels, sandbox requirements, trajectory
schema, and phased acceptance gates are spelled out in the spec document.

---

## Conventions

- API keys live only in `.env` (gitignored). Never commit them.
- Generated `.onnx` / `.pt` files are gitignored and reproduced by `scripts/generate_task_*.py`.
- Tasks numbered `task_NNNNNN` (six digits, zero-padded).
- Patch op `path` values are relative, always end with `.py`, never contain `..` or leading `/`.
- Trajectories include `failure_class` and `reward_breakdown` for debugging.
