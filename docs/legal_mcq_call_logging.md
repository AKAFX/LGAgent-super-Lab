# LegalMCQ 逐调用日志

本文说明 LegalMCQ 的逐调用观测、Solver/Verifier 结构化输出和受预算回退。
这些改动不重新运行或覆盖此前的 30 题结果。

## 1. 生成的文件

当 `tools/run_legal_mcq.py` 使用 `--execute --output output/run/results.jsonl` 时：

| 文件 | 内容 |
|---|---|
| `results.jsonl` | 每题结果或异常；成功和失败都包含 `diagnostics.trace` |
| `results.calls.jsonl` | 每次逻辑模型调用的开始和结束事件，逐条追加并 flush |
| `results.summary.json` | 数据集运行汇总，包括成功/失败、已观察 usage 和结束原因分布 |

单题 `--question` 同样保存结果和逐调用日志，但不生成数据集准确率汇总。
`--dry-run` 不创建日志，不初始化模型客户端。
如果结果、日志或汇总路径已经存在，程序拒绝覆盖，需换新输出路径。

## 2. 使用方式

先在独立配置中设置 `legal_mcq.enabled: true`、`lgagent_plus.enabled: false`。
默认只记录调用元数据：

```bash
.venv/bin/python tools/run_legal_mcq.py \
  --dry-run \
  --config output/legal_mcq/lexgenius-30-20260908-seed42/config.yaml \
  --dataset data/LexGenius.jsonl \
  --max-examples 1 \
  --output output/legal_mcq/logged-pilot/results.jsonl
```

将 `--dry-run` 改为 `--execute` 才会实际调用供应商。
诊断输出协议时可额外添加 `--log-model-output`，记录脱敏后的可见正文。
此选项不改变发送给模型的请求。

不要使用原 30 题的结果路径，也不要用新日志或新结果替换旧实验工件。

## 3. 事件关联

每个事件都包含：

- `trace_schema_version: 1`；
- `question_id` 和 `dataset_index`（单题时 index 为 null）；
- `run_id`：与结果的 `trace_id` 相同；
- `call_id`：一对开始/结束事件使用同一个 ID；
- `sequence`：题内事件序号；
- `agent`、`model`、`attempt`；
- `started_at`、`occurred_at`；
- 实际请求的 `max_tokens`，以及调用前预算。
- `budget_tokens`：本次 Prompt 估算量加最大输出量；
- `reserve_calls_after`、`reserve_tokens_after`：为后续必需阶段保留的额度；
- `timeout_seconds`：题级剩余时间与单次调用上限中的较小值。
- `response_format_type`：本次请求是否发送 `json_schema`；
- `response_schema_name`：例如 `legal_mcq_controller_v3`、
  `legal_mcq_solver_v3` 或 `legal_mcq_verifier_v2`。
- `reasoning_effort_requested`：实际请求的推理强度；未发送时为 null。

事件类型为 `model_call_started` 和 `model_call_finished`。
开始事件在调用模型之前写入，包括后续被预算拒绝的尝试。
结束事件在本次响应解析完成或抛错后立即写入，不等待整道题结束。

若进程被强制结束，最后一条可能只有 started，没有 finished。
这表示结果未知，不等于供应商没有执行，也不应自动重复请求。
可捕获的 KeyboardInterrupt / SystemExit 会记录 cancelled 后继续向上抛出。
突然断电或写入过程中被终止仍可能造成最后一行不完整；应保留此前完整行。

## 4. 结束记录

### 供应商原始元数据

| 字段 | 含义 |
|---|---|
| `response_model` | 供应商响应报告的模型名，不是对上游模型身份的独立认证 |
| `request_id` | 可取得的响应/请求标识，用于关联供应商记录 |
| `finish_reason` | 原样保留 stop、length、tool_calls、content_filter 等值；缺失时为 null |
| `content_state` | text、empty、null、missing、parts 或 unsupported |
| `content_chars` | 提取的可见正文字符数，不是 token 数 |
| `content_sha256` | 脱敏后可见文本的 SHA-256 |
| `output_text` | 默认 null；显式开启时保存脱敏后的可见正文 |
| `refusal_present` | 响应是否含 refusal 信息，不保存其正文 |
| `reasoning_present` | 响应是否包含非空 reasoning 字段，不保存其内容 |
| `usage` | 供应商返回的 prompt/completion/total token 计数 |
| `reasoning_tokens` | 供应商细分计数；未报告时为 null |
| `cached_prompt_tokens` | 供应商缓存命中计数；未报告时为 null |
| `usage_reported` | 是否收到 usage 对象；未收到响应时可能为 null |

`content_state: "null"` 是供应商明确返回空正文；
字段本身为 null 则意味着没有可用的正文状态元数据，例如网络异常。
`finish_reason=length` 是供应商结束原因，不直接等同于 JSON 无效：
需要结合 `schema_valid`、正文状态和错误信息分析。

`reasoning_tokens` 通常是 completion tokens 的子集，不应重复加到 total tokens。
没有 usage 时旧计数结构仍显示零，但 `usage_reported` 和汇总中的
`usage_unknown_calls` 明确标识未知，不能当作零费用。

### 执行状态

| outcome | 含义 |
|---|---|
| `success` | 本次响应通过 schema 解析，不表示最终答案正确 |
| `schema_error` | 收到响应，但结构校验失败 |
| `provider_error` | SDK/供应商调用失败，或响应缺少必需结构 |
| `budget_blocked` | 发起底层模型调用前被预算拒绝，`model_invoked=false` |
| `budget_exceeded_after_response` | 已收到响应，结算时发现超预算；保留结束原因和 usage |
| `deadline_exceeded` | 供应商调用执行中达到题级 deadline，工作流停止等待 |
| `cancelled` | 捕获到 KeyboardInterrupt / SystemExit |
| `error` | 其他执行异常 |

还记录 `schema_valid`、`model_invoked`、`error_type`、`error_message`、
`http_status`、`provider_error_code`、`provider_error_type`、
`seed_requested`、`provider_seed_guarantee`、`duration_ms` 和调用后预算。
未进行结构校验时 `schema_valid=null`，不会把网络错误误记为 schema 不通过。

## 5. 失败结果与汇总

Runner 会在异常的 `diagnostics` 中附带完整 Trace、最终预算和失败角色。
CLI 不再只读取 `error.details`，因此预算异常也能保留 `failed_agent`。
成功结果使用相同的 Trace 结构，便于统一分析。

数据集汇总新增：

- `logical_model_calls`：真正进入底层模型客户端的逻辑调用数；
- `budget_blocked_attempts`：调用前预算拒绝次数；
- `observed_usage`：包括失败调用在内，已经观察到的 token 用量；
- `observed_reasoning_tokens`：已报告的推理 token 计数；
- `observed_reasoning_token_rate`：reasoning 占 completion tokens 的比例；
- `reasoning_efforts`：各推理强度对应的逻辑调用数；
- `usage_unknown_calls`：未明确收到 usage 的已调用次数；
- `finish_reasons` 和 `call_outcomes`；
- `calls_output`：逐调用日志路径。

预算阻断不算模型调用；缺失的结束原因归入 `not_reported`，不假设为 stop。
总体准确率继续使用全部题目作为分母，失败题不删除、不替换。

预算编排按 Controller、Solver、Verifier、可选 Revision Solver、最终 Verifier
的剩余阶段进行保留。默认 7 次调用只在发生协议错误时使用额外额度；
正常题仍为 3 次，发生一次完整修订时为 5 次；Revision Solver 自身超时并
切换备用模型时最多 6 次。若完整修订和最终 Verifier 无法同时容纳，系统
保留当前审计结果并输出 `partial`，不会执行半套修订。

Solver 和 Revision Solver 使用紧凑 `solver-v3` 协议。历史 `solver-v1/v2`
结果仍可读取，但新请求不会再生成旧协议。配置项
`legal_mcq.solver_structured_output_mode` 有三种模式：

| 模式 | 行为 |
|---|---|
| `auto` | 先发送 strict JSON Schema；供应商明确拒绝该能力时，最多进行一次受预算的 prompt-only 重试 |
| `json_schema` | 强制发送 Schema；供应商不支持时直接失败，不静默降级 |
| `prompt_only` | 不发送 `response_format`，仅依赖当前 Prompt 和本地校验 |

`auto` 回退会写入 `legal-mcq-structured-output-fallback` 路由事件，并作为
新的逻辑调用占用调用数和 token 预算。普通网络错误、认证错误、空正文或
一般 Schema 校验失败不会被误判成“供应商不支持 JSON Schema”。
JSON Schema 本身的估算 token 也计入本次请求和后续阶段保留量。

Controller v3 使用 `legal_mcq_controller_v3` Schema，只生成事实、争点和
核验点；选项 Claim 由代码从原始选项确定性绑定。模式由
`controller_structured_output_mode` 单独控制，回退规则同上。
Verifier v2 使用 `legal_mcq_verifier_v2` Schema，模式由
`verifier_structured_output_mode` 控制。

主 Solver timeout 不在同一模型上重试。若配置备用 Solver，Trace 会写入：

- `legal-mcq-solver-timeout`：主模型超时；
- `legal-mcq-solver-fallback`：一次共享预算的备用调用；
- `legal-mcq-solver-circuit-open`：连续超时达到阈值，后续题绕过主模型；
- `legal-mcq-solver-fallback-skipped`：剩余墙钟不足以保留 Final Verifier。

使用备用 Solver 的题不再执行业务 Revision。确定性 Validator 已失败时，
`legal-mcq-verifier-skipped-deterministic` 表示第一轮 LLM Verifier 被跳过，
直接使用错误码进入 Revision。

v3 调度新增 `visible_output_tokens`、`reasoning_allowance_tokens`、
`completion_token_cap` 和 `extra_reasoning_reserve_tokens` 字段。前两项
表示规划的正文目标和共享输出额度中的推理余量，不表示供应商强制子预算。
辅助角色的额外预留用于覆盖可能计在正文上限之外的 reasoning usage。

`legal-mcq-length-retry` 记录一次 length 恢复的新旧上限；不复用截断正文。
`legal-mcq-length-retry-skipped` 表示已达模型硬上限或已尝试一次恢复，
`legal-mcq-retry-skipped-budget` 表示后续阶段额度不足。所有实际重试仍计入
全局调用、token 和墙钟预算。

## 6. 安全边界

逐调用日志不接收 Request 正文、API 请求头、配置中的凭据或 Evaluator Oracle。
它由模型调用层生成，只向下游日志文件输出，不参与 Prompt 构造。

所有已配置角色 key 的完整值会在日志文本中替换为 `[REDACTED]`，
结构化的 api_key、authorization 等字段也会脱敏。
日志使用独占创建方式，文件权限为 0600；多个题目线程通过锁串行写入，
避免一行 JSON 被不同线程穿插。

`--log-model-output` 只记录 provider 的可见文本，不序列化整条 SDK message，
避免 content=null 时把 reasoning 字段一起保存。
可见正文仍可能包含题目中的业务信息；脱敏不是通用个人信息清洗，
不应未经检查分享开启正文记录后的日志。

## 7. 本阶段的限制

- 记录单位是逻辑模型调用。LegalMCQ 创建的 OpenAI-compatible 客户端将
  SDK 自动重试设为 0，因此每次 HTTP 尝试均由可审计的逻辑调用触发；
  外部注入的自定义客户端仍需自行满足这一约束。
- 题级 hard deadline 通过请求 timeout 和守护等待共同执行。若自定义
  Provider 完全忽略取消，后台 daemon 调用可能短暂继续；日志会设置
  `provider_may_still_be_running`，其未知 usage 不会被伪报为零成本结论。
- null、missing、empty、refusal 和意外 tool calls 现在会保留响应元数据，
  并明确归类为 provider error，不再把整个 SDK message 当作答案 JSON。
- 极性、日期解析、预算调度、模型调用 deadline、紧凑 Solver v3 和
  JSON Schema 请求均已实现并通过离线测试。
- 2026-09-11 独立 dev 5 题验证中，CommandCode 未拒绝 9 次 Schema 请求，
  未触发 `auto` 回退；6 / 9 次调用直接通过 v2 解析，3 次因
  `finish_reason=length` 截断。详见
  [CommandCode JSON Schema 独立 dev 验证](legal_mcq_commandcode_schema_validation_20260911.md)。
- 角色级 `reasoning_effort=low` 已接入并记录。真实重放中辅助模型 reasoning
  明显下降，但 LongCat 将其作为软约束，不能保证固定 token 上限；详见
  [隐藏推理预算优化](legal_mcq_reasoning_budget_optimization_20260911.md)。
- 逐调用日志 flush 可以减少进程退出造成的丢失，不提供断电级持久性保证。
- 本次未付费重跑。旧 30 题没有保存的 finish_reason 和正文无法补回，
  不会伪造其历史日志。

## 8. 离线验证

```bash
.venv/bin/python -m pytest \
  tests/test_legal_mcq_observability.py \
  tests/test_legal_mcq_cli.py \
  tests/test_task22_legal_mcq.py \
  tests/test_task3_domain.py -q
```

测试使用真实 SDK 响应类型、httpx.MockTransport 和 Fake Model，
覆盖截断、空正文、缺失 usage、分段正文、拒答标记、网络超时、
认证失败、schema 重试、预算前阻断、响应后超预算、
中断、并发文件写入、凭据脱敏、JSON Schema 透传、显式回退、
强制 Schema 模式以及单题/批量 CLI 落盘。
