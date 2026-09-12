# LGAgent LegalMCQ 三角色超级优化技术说明

## 1. 结论

本次改造不创建替代工程，而是在 LGAgent 中增加独立的
`src/lgagent/legal_mcq/` 能力域。新链路默认关闭，原始 LGAgent 和
LGAgent++ 的行为不因代码存在而改变。

目标架构是：

```text
Deterministic Request Boundary
  -> Controller（经济模型，只做答案中立结构化）
  -> Strong Solver（先进模型，唯一主答题者）
  -> Deterministic Decision Validator
  -> Verifier（经济模型，只做规范和一致性审计）
  -> 最多一次 Strong Solver 修订
  -> Deterministic Finalizer
```

该方案在工程上可行，并已形成离线可执行通路。它能够保证角色边界、
评测答案隔离、逐项裁决、证据来源约束、调用预算和一键回滚，但尚不能
据此声称准确率已经提高。准确率结论必须来自冻结配置后的独立测试集。

## 2. 对原技术包的适配决策

| 技术包能力 | LGAgent 适配 | 决策 |
|---|---|---|
| 领域 Request / Oracle 分离 | `adapter.py` | 完整吸收 |
| 确定性题目解析 | `parser.py` | 完整吸收并支持全角、跨行、日期、反向题 |
| 三角色 Agent | `agent.py` | 改造成 Controller / Strong Solver / Verifier |
| Skill 注册与预加载 | `skills.py` + YAML | 采用本地版本化、确定性加载 |
| Tool 白名单 | Skill 声明 + 固定 Provider | 不开放任意 ReAct 工具调用 |
| Open-book | `EvidenceProvider` | 仅允许已抓取、已审计的正文证据 |
| 搜索与抓取 | 复用 OATH-RAG | 不复制一套无约束网页搜索 |
| 最终答案准备 | `DecisionValidator` + `LegalAnswer` | 改为确定性 Finalizer |
| 流式状态 | 复用 `RunTrace` 路由事件 | 当前不引入独立 SSE 状态机 |
| API / Redis | 暂不迁移 | 当前项目无 FastAPI 依赖，先保持核心纯 Python |
| 评测 | `EvaluationCase` / `evaluate_case` | Oracle 在 Agent 返回后才读取 |

未直接迁移 FastAPI、Redis、开放式 ReAct 和第二套持久化系统，原因是这些
能力不会直接提升选择题判断质量，却会扩大依赖、运行状态和回滚面。现有
`ChatModel`、`ExecutionBudget`、`RunTrace`、配置和评测设施足以承载本阶段。

## 3. 三角色职责

### 3.1 Controller

Controller 只输出：

- 关键事实；
- 法律争点；
- 核验点。

新请求使用紧凑 `controller-v3`。Controller 不再拆分或改写选项命题；
代码从可信的原始选项文本为每项生成唯一规范 Claim 和 A1、B1 等 ID。
这同时固定 Claim 文本、所属选项和 ID，消除模型跨选项误挂命题的风险。
Schema 按题目选项生成，旧 Controller v1/v2 协议仍可读取，但其中的
模型 Claims 会在进入 Solver 前被确定性绑定结果覆盖。

`ControllerPlan.from_text()` 会递归拒绝 `answer`、`selected_options`、
`prediction` 等答案字段。Controller 不能决定最终选项。

### 3.2 Strong Solver

Solver 是唯一主答题角色，应配置为可用的最强模型。它必须：

- 对全部选项建立 `OptionAssessment`；
- 对每个规范选项 Claim 建立 `ClaimAssessment`，并在 reason 中核验复合条件；
- 为每个 claim 输出 `supported`、`contradicted` 或 `uncertain`；
- 最后输出候选选项集合；
- Open-book 时只能引用输入中的 `evidence_id`。

Controller 和 Verifier 可以使用更经济的模型，降低成本并限制其权力。

新请求固定生成紧凑的 `solver-v3`：

```json
{
  "protocol_version": "solver-v3",
  "selected_options": ["C"],
  "options": [
    {
      "label": "A",
      "claims": [
        {
          "claim_id": "A1",
          "verdict": "contradicted",
          "reason": "决定性理由",
          "evidence_ids": []
        }
      ],
      "verdict": "contradicted"
    }
  ],
  "confidence": 0.9
}
```

v3 复用 Controller 的 `claim_id`，并在 v2 基础上删除重复的 issues 和
顶层 rationale；最终理由由代码从已选项 claim reasons 组合。Parser 将
v3 归一化为既有领域对象，因此 Validator、Verifier、Finalizer 和 Oracle
隔离边界不变；历史 solver-v1/v2 结果保持只读兼容。

### 3.3 Verifier

Verifier 不直接覆盖答案，只检查：

- 题干是正向还是反向；
- 复合选项是否漏拆；
- 主体、行为和责任对象是否混淆；
- 构成要件、例外和时效是否遗漏；
- 引用是否真实存在并支持结论；
- 选项是否完整覆盖。

Verifier 拒绝后最多触发一次 Solver 全量修订。第二次仍冲突时输出
`partial` 和 `needs_review=true`，不会进入无限对话。

## 4. 确定性安全边界

以下规则不交给模型：

1. Benchmark 数据先拆成 `LegalQuestionRequest` 和 `EvaluationOracle`。
2. Agent、Prompt、检索 Query 和 Trace 只能接收 Request。
3. Controller 输出中任何嵌套答案字段都会被拒绝。
4. 单选必须只选一项；多选必须等于目标 verdict 的完整集合。
5. 反向题选择 `contradicted`，正向题选择 `supported`。
6. 每个题目选项必须有且只有一个 Option Assessment。
7. `supported` 选项的必要 claim 必须全部 supported。
8. `contradicted` 选项必须至少存在一个 contradicted claim。
9. Closed-book 禁止外部 Evidence ID。
10. Open-book Evidence ID 必须存在，来源必须达到权威阈值并覆盖题目日期。
11. 搜索摘要类型不能作为决定性证据。
12. 重复 Evidence ID、未知引用和版本错误均会阻断自动接受。
13. Verifier 引用未知选项或同时“接受并报错”会被代码判为合约冲突。
14. 所有模型调用共享题级调用数、token 和墙钟预算。

这些边界使模型负责推理，代码负责权限、数据流和可验证不变量。

## 5. Skill Runtime

生产 Skill 位于 `src/lgagent/legal_mcq/skill_specs/`：

- `legal-mcq-core@1.0.0`：Closed-book 和 Open-book 均加载；
- `legal-research@1.0.0`：仅 Open-book 加载；
- `legal-evaluation@1.0.0`：隐藏，仅供评测端说明，不进入生产 Prompt。

Registry 校验文件名、名称、版本、引用文本和 Tool 白名单。实际加载的
Skill 名称与版本写入 Trace 和结果诊断，便于复现实验。YAML 已声明为
setuptools package data，安装 wheel 后仍会保留。

## 6. Open-book 证据链

Open-book 不接受搜索结果摘要直接作为法源。受控链路为：

```text
Controller option claims
  -> OATH-RAG retrieval
  -> Evidence Auditor exact_span
  -> AuditedEvidenceMatrix
  -> OathAuditedEvidenceProvider
  -> LegalEvidence
  -> Solver
```

`OathAuditedEvidenceProvider` 只转换 SUPPORT、REFUTE、EXCEPTION 三类已审计
正文片段，过滤 irrelevant 和 temporally invalid 项。转换时要求 HTTP(S)
来源 URI，并保留 law、article、有效期、authority level、exact span 和
SHA-256 内容哈希。

该 Provider 是可选注入项。默认 `closed_book` 不进行检索；显式
`open_book` 但未配置 Provider 时会 fail closed，不会悄悄退回参数知识答案。

## 7. 模型客户端和预算

`build_openai_role_clients()` 按每个角色的 `base_url` 和 `api_key` 创建
OpenAI-compatible 客户端：

- endpoint 和凭据相同时复用 transport；
- endpoint 或凭据不同时创建独立 client；
- 缺少任一必需凭据时，在调用前失败；
- 异常和输出记录会对全部角色 key 做脱敏。

正常链路调用 3 次：

1. Controller；
2. Solver；
3. Verifier。

发生一次修订时调用 5 次。JSON 结构修复也消耗同一个全局预算，因此恶劣
响应无法绕过 `max_model_calls` 或 `max_total_tokens`。

推荐默认值：

```yaml
legal_mcq:
  enabled: false
  mode: closed_book
  seed: 42
  max_attempts: 2
  max_revision_rounds: 1
  max_model_calls: 7
  max_total_tokens: 65536
  max_wall_time_seconds: 360
  model_call_timeout_seconds: 60
  solver_call_timeout_seconds: 120
  solver_fallback_timeout_seconds: 90
  verifier_call_timeout_seconds: 45
  controller_structured_output_mode: auto
  solver_structured_output_mode: auto
  solver_visible_output_tokens: 2048
  solver_reasoning_allowance_tokens: 6144
  auxiliary_reasoning_reserve_tokens: 2048
  solver_model:
    model_name: gpt-5.6-sol
    max_tokens: 12288
    reasoning_effort: high
  solver_fallback_model:
    model_name: Qwen/Qwen3.8-Max-0902
    max_tokens: 8192
    reasoning_effort: medium
```

正常链路仍只调用 3 次，主 Solver timeout 加备用为 4 次，完整修订为 5 次，
Revision Solver 再切备用时为 6 次。7 次上限保留一个协议修复调用，同时
通过阶段保留量保证 Solver 修订后仍能执行最终 Verifier。
每次调用同时受角色级上限和题级剩余时间约束；当前前沿模型配置为
Controller 30 秒、主 Solver 120 秒、备用 Solver 90 秒、Verifier 45 秒。
OpenAI-compatible 客户端禁用 SDK 隐式重试，所有逻辑调用都进入统一
Trace 和预算。

Controller、Solver 和 Verifier 分别通过对应的
`*_structured_output_mode` 支持三种模式：

| 模式 | 行为 |
|---|---|
| `auto` | 先发送 strict `response_format=json_schema`；供应商明确拒绝时，最多进行一次受预算、可追踪的 prompt-only 回退 |
| `json_schema` | 强制 Schema，不进行静默回退 |
| `prompt_only` | 不发送 `response_format`，保留纯 Prompt 兼容路径 |

Schema 仅使用跨供应商公共关键字 `type`、`properties`、`required`、
`additionalProperties`、`enum` 和 `items`。标签范围、非空集合、
置信度区间和选项覆盖仍由本地 Parser/Validator 强制检查。Schema 文本
计入请求 token 估算；`auto` 回退是新的逻辑调用，也计入统一预算和 Trace。

CommandCode 的部分推理模型会把隐藏 reasoning tokens 计入总 usage，因此
使用上述 token 和墙钟预算，避免合法响应在 Final Verifier 前被提前截断。
2026-09-11 的独立 dev 5 题验证中，CommandCode 未拒绝 9 次 Schema 请求，
也未触发 prompt-only fallback；调用级 v2 解析成功率为 6 / 9，主 Solver
在两次尝试内为 5 / 5。剩余失败来自 3 次 `length` 截断及 Verifier 的
520/超时，详见
[CommandCode JSON Schema 独立 dev 验证](legal_mcq_commandcode_schema_validation_20260911.md)。

各角色模型可选配置 `reasoning_effort`。未配置或使用 `provider_default`
时不发送该参数；当前前沿配置对 GPT 主 Solver 使用 `high`、Qwen3.8 Max
fallback 使用 `medium`，辅助 Qwen 使用 `low`，并在 Trace 中记录
`reasoning_effort_requested`。历史工程重放显示辅助模型 reasoning 降幅明显，
LongCat 则只把 low 当作软约束。实现与对照数据见
[隐藏推理预算优化](legal_mcq_reasoning_budget_optimization_20260911.md)。

在后续健康检查仍发生截断后，v3 调度改为正文目标 2048 加推理余量 2048，
正常 Solver 请求共 4096；模型配置的 6144 是恢复硬上限。仅遇到 length
且预算足够时扩额一次，按已观察推理量规划正文空间，丢弃截断 JSON 后重生成。
这不是供应商强制的子预算，仍可能截断；题级 32768 tokens、7 calls、
180 秒限制不变。详情见 [输入输出修复](legal_mcq_io_repair_20260911.md)。

真实同题重放显示 Solver 截断下降但 5/7 调用转为超时。v4 因此取消同模型
timeout 重试，主 Solver 使用 70 秒，超时后用共享题级预算调用一次 45 秒
备用 Solver；连续两次主 Solver 超时后，当前 Runner 余下题目直接走备用。
使用备用 Solver 的题不再进行业务 Revision。Verifier 改为严格 verifier-v2
Schema，正文目标 384，length 恢复上限 512；确定性校验已失败时直接进入
Revision，不先调用 LLM Verifier。详见
[超时与 Verifier 优化](legal_mcq_timeout_verifier_optimization_20260912.md)。
前沿模型选择、真实兼容性探针和预算调整见
[前沿模型角色升级](legal_mcq_frontier_model_upgrade_20260912.md)。

## 8. 可回滚性

开启新链路：

```yaml
legal_mcq:
  enabled: true
lgagent_plus:
  enabled: false
```

立即回滚：

```yaml
legal_mcq:
  enabled: false
```

`legal_mcq.enabled` 与 `lgagent_plus.enabled` 被配置层强制互斥。关闭后
`LGAgentPlusRunner.run()` 继续走原 `corrected_baseline` 或 LGAgent++
分支，新角色模型不会被调用。无需删除代码、迁移数据或恢复 Git 提交。

## 9. 运行入口

先做无费用校验：

```bash
.venv/bin/python tools/run_legal_mcq.py \
  --dry-run \
  --config examples/parameter/legal2_rag_parameter.yaml \
  --question $'题干\nA. 选项甲\nB. 选项乙'
```

配置中 `legal_mcq.enabled` 必须显式设为 `true`。真实执行还必须显式使用
`--execute`：

```bash
.venv/bin/python tools/run_legal_mcq.py \
  --execute \
  --config examples/parameter/legal2_rag_parameter.yaml \
  --dataset data/LexGenius.jsonl \
  --max-examples 100 \
  --concurrency 10 \
  --output output/legal_mcq/lexgenius-100.jsonl
```

数据集入口通过 `BenchmarkRecordAdapter` 隔离 Oracle，结果按原顺序写入
JSONL，并生成同名 `.summary.json`。执行失败会记录稳定错误类型，API key
会被脱敏。

逐调用日志扩展另见 [LegalMCQ 逐调用日志](legal_mcq_call_logging.md)。
新版本还会逐条写入 `.calls.jsonl`，成功和失败都保留 Trace、供应商结束原因、
正文状态及已观察到的 token；默认不保存模型正文。

## 10. 可行性分析

### 已验证可行

- 不改变旧链路即可接入三角色 Agent；
- 不同角色可使用不同模型、endpoint 和 key；
- 结构错误可有限重试；
- 全局预算可在下一次调用前阻断；
- 单选、多选、反向题和跨行选项可确定性处理；
- Closed-book 不生成伪引用；
- Open-book 可复用现有 OATH 审计结果；
- Agent 与 benchmark 标签之间存在明确隔离边界；
- 关闭开关后不会调用任何 LegalMCQ 角色模型。

### 条件可行

- Open-book 的质量取决于法律语料覆盖率、版本元数据和 Evidence Auditor；
- 第三方 OpenAI-compatible 服务是否严格执行 seed 不能由客户端保证；
- 不同先进模型对严格 JSON 的遵循程度不同，仍需保留结构修复预算；
- 并发上限还受供应商限流和 key 配额约束。

### 当前不应宣称

- 不能宣称三角色链路已经提升 LexGenius 准确率；
- 不能把 Verifier 接受率当作正确率；
- 不能用测试集观察结果反向调整 Prompt、阈值或模型组合；
- 不能把搜索摘要、题库答案页或模型记忆当作权威法律证据。

## 11. 正式效果验证

建议按以下顺序冻结并实验：

1. 在至少 200 道独立 dev 题上选择 Controller、Solver、Verifier 模型。
2. 冻结 Prompt 版本、Skill 版本、温度、token、修订次数和预算。
3. 固定随机种子 `42`、`43`、`44`。
4. 在独立 test split 上比较 Original 与 LegalMCQ。
5. 同时报告 Exact Match、全选项覆盖率、Verifier 接受率、Review 率、
   平均调用数、token 和延迟。
6. 对配对结果执行 bootstrap 置信区间或 McNemar 检验。
7. 单独报告 Original 改对和改错的题数，防止平均值掩盖负迁移。

只有冻结后的独立测试显示稳定正增益，才能在论文中声称方法有效。

## 12. 代码地图

| 文件 | 责任 |
|---|---|
| `legal_mcq/models.py` | 领域模型、严格 JSON 协议、结果格式 |
| `legal_mcq/parser.py` | 确定性题目、极性和日期解析 |
| `legal_mcq/adapter.py` | Request / Oracle 隔离和泄漏检测 |
| `legal_mcq/skills.py` | Skill 加载、版本、目录和 Tool 白名单 |
| `legal_mcq/agent.py` | 三角色编排、一次修订和 Trace |
| `legal_mcq/decision.py` | 选项、claim、证据和极性不变量 |
| `legal_mcq/evidence.py` | OATH 审计证据转换 |
| `legal_mcq/clients.py` | 独立角色 endpoint/key 客户端 |
| `legal_mcq/evaluation.py` | Oracle 后读评测 |
| `tools/run_legal_mcq.py` | 显式 dry-run/execute 单题和数据集入口 |

## 13. 剩余风险

- 已完成 CommandCode Schema、fallback、熔断和 Controller v3 的真实小样本
  验证，但样本不足以代表长期供应商 SLA；
- 当前未跑冻结后的 LegalMCQ v5 正式准确率对照；
- Open-book Provider 已实现接口与 OATH 适配，但默认 CLI 不自动启用检索；
- 当前复用 `RunTrace`，未提供 HTTP SSE 和 Redis 跨进程任务恢复；
- OATH 当前 corpus schema 没有独立 publisher 字段，适配器使用来源域名作为
  publisher 展示值，真正上线前可扩展 corpus 元数据；
- 模型的法律结论仍可能错误，`needs_review=false` 只表示流程校验通过。
