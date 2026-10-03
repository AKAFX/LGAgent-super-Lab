# LGAgent-super-Lab 端到端运行展示

## 1. 本次运行概览

本次使用冻结的 LexGenius dev 样本，运行前固定三个此前没有 LegalMCQ 结果的
案例。用途是展示真实输入、各阶段输出和失败处理，不是正式准确率实验。

| ID | 领域 | 最终状态 | 预测 | 标签 | 耗时 |
|---|---|---|---|---|---:|
| 510 | Judicial Practice | completed | B | B | 48.72s |
| 472 | Legal Ethics | failed | 无答案 | B | 30.03s |
| 273 | Legal Understanding | completed | A | A | 41.23s |

总体结果：

- 3 个输入；
- 2 个完整答案，均与数据标签一致；
- 1 个 Controller 硬超时，未生成答案；
- 7 次逻辑模型调用；
- 6 次 `stop + Schema valid`；
- 0 次 Schema error；
- 0 次 length；
- 0 次 fallback；
- 0 次 Revision；
- 已观察 11,764 tokens，其中 reasoning tokens 为 4,388；
- 总墙钟时间 119.99 秒。

## 2. 整体状态机

```text
Benchmark JSONL
  |
  +-- BenchmarkRecordAdapter
  |     +-- LegalQuestionRequest ----------> Agent 可见
  |     `-- EvaluationOracle(golden answer) -> Agent 完成后才读取
  |
  +-- Deterministic Parser
  |     +-- 题干与 A-D 选项
  |     +-- 单选/多选
  |     +-- 正向题/反向题
  |     +-- 法域
  |     `-- 适用日期
  |
  +-- Skill Router
  |     `-- closed_book -> legal-mcq-core
  |
  +-- Controller v3 / Qwen3.7 Flash
  |     +-- facts
  |     +-- issues
  |     `-- checks
  |
  +-- Deterministic Claim Binder
  |     `-- 原始选项文本 -> A1/B1/C1/D1
  |
  +-- Strong Solver v3 / GPT-5.6 Sol
  |     +-- 每个选项 verdict
  |     +-- 每个 Claim 的决定性理由
  |     +-- selected_options
  |     `-- confidence
  |
  +-- Deterministic Validator
  |     +-- 选项覆盖和基数
  |     +-- Claim ID 归属
  |     +-- 极性与 verdict 一致性
  |     `-- 引用和证据约束
  |
  +-- Verifier v2 / Qwen3.7 Flash
  |     `-- accepted / stable error codes
  |
  +-- optional one Revision
  |
  `-- Deterministic Finalizer
        `-- LegalAnswer
```

当前角色配置：

| 角色 | 模型 | 权限 |
|---|---|---|
| Controller | Qwen/Qwen3.7-Flash | 只能整理事实、争点和检查项 |
| Strong Solver | gpt-5.6-sol | 唯一主答题模型 |
| Solver fallback | Qwen/Qwen3.8-Max-0902 | 仅在主 Solver timeout 后调用一次 |
| Verifier | Qwen/Qwen3.7-Flash | 只能接受或报告错误，不能直接改答案 |

## 3. Agent 真正接收什么

原始数据行包含：

```json
{
  "id": "510",
  "question": "题干和选项",
  "domain": "Judicial Practice",
  "golden_answers": ["B"]
}
```

适配器在调用 Agent 前拆分为：

```text
LegalQuestionRequest:
  question_id, stem, options, jurisdiction, date, type, mode

EvaluationOracle:
  golden_answers
```

`golden_answers` 不进入 Controller、Solver 或 Verifier Prompt。Evaluator
只在最终答案产生后读取 Oracle 并计算 Exact Match。

## 4. 案例 510：完整成功链路

### 4.1 原始输入

案件发生于 2014 年。朱某某因精神分裂症实施放火行为，经鉴定无刑事责任
能力。家属主张其病情稳定并愿意承担监护责任，反对强制医疗。问题要求判断
法院最可能依据哪项规定处理。

选项核心：

- A：直接适用《精神卫生法》住院治疗；
- B：依据《刑事诉讼法》评估继续危害社会可能性；
- C：优先家属监护并驳回强制医疗；
- D：变更监护人替代强制医疗。

### 4.2 确定性解析

```json
{
  "question_id": "510",
  "question_type": "single_choice",
  "asks_for_incorrect_option": false,
  "jurisdiction": "CN",
  "as_of_date": "2014-01-06",
  "date_source": "case_explicit",
  "mode": "closed_book",
  "parse_warnings": []
}
```

这一阶段没有模型参与。系统确定：

- 这是单选正向题；
- 必须且只能选择一个 `supported` 选项；
- 适用时点是 2014-01-06；
- 不允许 Solver 伪造外部证据。

### 4.3 Controller v3 输出

```json
{
  "protocol_version": "controller-v3",
  "facts": [
    "朱某某因妄想点燃仓房烧毁房屋，经鉴定作案时无刑事责任能力。",
    "检察机关以有继续危害社会可能为由申请强制医疗。",
    "法院采纳意见决定强制医疗。",
    "家属主张病情稳定且愿承担监护责任，反对强制医疗。"
  ],
  "issues": [
    {
      "description": "法院处理家属反对强制医疗主张的法律依据",
      "options": ["A", "B", "C", "D"]
    }
  ],
  "checks": [
    "是否应直接适用《精神卫生法》第三十条实施住院治疗",
    "是否应综合评估社会危害性后决定是否强制医疗",
    "是否应优先考虑家属监护意愿并驳回强制医疗申请",
    "是否应裁定变更监护人以替代强制医疗"
  ]
}
```

Controller 没有输出答案，也没有生成 Claim ID。

### 4.4 代码绑定 Claim

```json
{
  "A": ["A1 <A项完整原文>"],
  "B": ["B1 <B项完整原文>"],
  "C": ["C1 <C项完整原文>"],
  "D": ["D1 <D项完整原文>"]
}
```

Claim 来自可信原始选项，模型不能把 B 的子命题挂到 C 或 D。

### 4.5 Strong Solver v3 输出

```json
{
  "protocol_version": "solver-v3",
  "selected_options": ["B"],
  "options": [
    {
      "label": "A",
      "claims": [{
        "claim_id": "A1",
        "verdict": "contradicted",
        "reason": "本案属于刑事强制医疗程序，不能以一般住院治疗规定直接替代刑事诉讼中的法定条件审查。",
        "evidence_ids": []
      }],
      "verdict": "contradicted"
    },
    {
      "label": "B",
      "claims": [{
        "claim_id": "B1",
        "verdict": "supported",
        "reason": "朱某某实施危害公共安全的放火行为且经鉴定不负刑事责任，法院应重点审查其是否仍有继续危害社会的可能。",
        "evidence_ids": []
      }],
      "verdict": "supported"
    },
    {
      "label": "C",
      "claims": [{
        "claim_id": "C1",
        "verdict": "contradicted",
        "reason": "家属愿意监护并不当然排除继续危害社会的可能，法院无须因此驳回强制医疗申请。",
        "evidence_ids": []
      }],
      "verdict": "contradicted"
    },
    {
      "label": "D",
      "claims": [{
        "claim_id": "D1",
        "verdict": "contradicted",
        "reason": "变更监护人不能替代刑事强制医疗审查，且题示时点不能适用尚未施行的《民法典》。",
        "evidence_ids": []
      }],
      "verdict": "contradicted"
    }
  ],
  "confidence": 0.99
}
```

### 4.6 确定性校验

通过的约束包括：

- 四个选项全部覆盖；
- A1/B1/C1/D1 均属于正确选项；
- 单选题只选择 B；
- 正向题选择项 B 的 verdict 为 `supported`；
- 未选项均为 `contradicted`；
- closed-book 下没有伪造 Evidence ID；
- 无 Revision 必要。

### 4.7 Verifier v2 输出

```json
{
  "protocol_version": "verifier-v2",
  "accepted": true,
  "error_codes": [],
  "challenged_options": [],
  "suggested_selected_options": [],
  "note": ""
}
```

### 4.8 最终答案

```json
{
  "question_id": "510",
  "selected_options": ["B"],
  "concise_answer": "B",
  "rationale": "朱某某实施危害公共安全的放火行为且经鉴定不负刑事责任，法院应重点审查其是否仍有继续危害社会的可能。",
  "confidence": 0.99,
  "status": "completed",
  "needs_review": false,
  "warnings": []
}
```

Oracle 在此后读取，标签为 B，Exact Match 为 true。

### 4.9 调用与预算

| 调用 | 模型 | 耗时 | Tokens | Reasoning | Schema |
|---|---|---:|---:|---:|---|
| Controller | Qwen3.7 Flash | 22.19s | 2,049 | 1,024 | valid |
| Solver | GPT-5.6 Sol | 6.58s | 2,113 | 177 | valid |
| Verifier | Qwen3.7 Flash | 19.93s | 2,396 | 1,024 | valid |

最终使用 3/7 calls、6,558/65,536 tokens、48.72/360 秒。

## 5. 案例 472：受控失败链路

### 5.1 原始输入

题目要求法官在累犯从重、公共安全危害、认罪态度和家庭困难之间进行量刑平衡。
数据标签为 B。

### 5.2 确定性解析

解析器成功识别：

```json
{
  "question_id": "472",
  "question_type": "single_choice",
  "asks_for_incorrect_option": false,
  "jurisdiction": "CN",
  "date_source": "case_explicit",
  "mode": "closed_book"
}
```

### 5.3 Controller timeout

```json
{
  "agent": "legal_mcq_controller",
  "model": "Qwen/Qwen3.7-Flash",
  "outcome": "deadline_exceeded",
  "duration_seconds": 30.0145,
  "timeout_seconds": 30,
  "response_schema_name": "legal_mcq_controller_v3",
  "schema_valid": null,
  "provider_may_still_be_running": true
}
```

### 5.4 为什么没有答案

Controller 没有在角色级 30 秒 deadline 内返回可信 JSON，因此：

- 不从未知或残缺正文猜测事实；
- 不进入 GPT Solver；
- 不消耗 fallback；
- 不读取 Oracle；
- 不生成伪 completed 答案；
- 失败记录保留在分母中。

最终结果：

```json
{
  "question_id": "472",
  "error_type": "BudgetExceededError",
  "message": "execution budget exhausted: max_seconds",
  "failed_agent": "legal_mcq_controller",
  "prediction": null,
  "model_calls": 1,
  "elapsed_seconds": 30.03
}
```

这暴露了当时的真实瓶颈：Qwen Controller 的 30 秒上限在供应商延迟波动下
偏紧。它属于链路可靠性问题，不是 GPT 法律推理问题。

### 5.5 Controller 60 秒单变量重放

随后将 Controller timeout 从 30 秒统一调整到 60 秒，题目、模型、Prompt、
Token 预算和其他角色 timeout 均保持不变，再次运行 ID 472。

对照结果：

| 指标 | 30 秒 | 60 秒 |
|---|---:|---:|
| Controller | deadline exceeded | stop + Schema valid |
| Controller 耗时 | 30.01s 后中断 | 20.82s |
| Solver | 未调用 | GPT-5.6 Sol，6.80s |
| Verifier | 未调用 | Qwen3.7 Flash，15.85s |
| 最终状态 | failed | completed |
| 预测 | null | B |
| 标签 | B | B |
| Exact Match | 按失败计错 | true |
| 总逻辑调用 | 1 | 3 |
| 已观察 tokens | 未知 | 6,328 |
| 总耗时 | 30.03s | 43.47s |

重放中的 Controller 输出：

```json
{
  "protocol_version": "controller-v3",
  "facts": [
    "被告人梁大香系刑满释放人员，重新犯罪后被捕，构成累犯。",
    "多次盗剪铁路行车调度线和通讯线，造成通讯中断和经济损失。",
    "被告认罪态度较好，但犯罪危及公共安全。",
    "家属主张家庭极度贫困且需赡养老人。"
  ],
  "issues": [{
    "description": "量刑时如何平衡累犯从重与家庭困难等从轻情节",
    "options": ["A", "B", "C", "D"]
  }],
  "checks": [
    "是否违反累犯从重规定",
    "是否在法定量刑幅度内综合考量",
    "是否必须顶格量刑",
    "是否可以违法撤诉并转为社区矫正"
  ]
}
```

GPT Solver 的核心输出：

```json
{
  "protocol_version": "solver-v3",
  "selected_options": ["B"],
  "options": [
    {
      "label": "A",
      "verdict": "contradicted",
      "reason": "家庭困难不能成为突破法定量刑规范、违反累犯从重要求的依据。"
    },
    {
      "label": "B",
      "verdict": "supported",
      "reason": "在法定幅度内兼顾犯罪危害、累犯从重要求及认罪、家庭困难等情节，符合罪责刑相适应原则。"
    },
    {
      "label": "C",
      "verdict": "contradicted",
      "reason": "累犯从重不等于必须顶格量刑，认罪态度等相关情节仍应纳入综合裁量。"
    },
    {
      "label": "D",
      "verdict": "contradicted",
      "reason": "法院不得以人文关怀为由要求违法撤诉，严重犯罪也不宜以社区矫正替代依法判刑。"
    }
  ],
  "confidence": 0.98
}
```

Verifier 返回 `accepted=true`，最终答案为 B，`status=completed`。这次重放
证明 60 秒配置可以恢复已观察到的 Controller timeout，但单题成功不能证明
供应商未来所有请求都不会超过 60 秒。

## 6. 案例 273：完整成功链路

### 6.1 原始输入

```text
我国宪法规定，国家为了公共利益需要，可以依照法律规定对公民私有财产
实行征收或征用并给予补偿。问财产权限制界限的正确表述。

A. 征收或征用构成财产权的外部限制
B. 只需行政法规具体授权
C. 只需满足合目的性原则
D. 限制范围可由立法机关自由裁量
```

### 6.2 确定性解析

```json
{
  "question_id": "273",
  "question_type": "single_choice",
  "asks_for_incorrect_option": false,
  "jurisdiction": "CN",
  "as_of_date": "2026-09-24",
  "date_source": "runtime_default",
  "mode": "closed_book",
  "parse_warnings": []
}
```

### 6.3 Controller v3 输出

```json
{
  "protocol_version": "controller-v3",
  "facts": [
    "我国宪法规定国家为了公共利益的需要，可以依照法律规定对公民的私有财产实行征收或者征用并给予补偿。"
  ],
  "issues": [{
    "description": "判断关于公民财产权限制界限的正确表述",
    "options": ["A", "B", "C", "D"]
  }],
  "checks": [
    "选项A是否准确描述了征收或征用作为外部限制的性质",
    "选项B中仅遵循行政法规授权是否符合宪法规定的法律依据要求",
    "选项C中合目的性原则是否足以单独构成限制条件而无需其他要件",
    "选项D中立法机关自由裁量范围是否受宪法正当性约束"
  ]
}
```

### 6.4 确定性 Claim

```json
{
  "A": ["A1 对公民私有财产的征收或征用构成对公民财产权的外部限制"],
  "B": ["B1 对公民私有财产的征收或征用应遵循行政法规的具体授权即可"],
  "C": ["C1 只要满足合目的性原则即可对公民的财产权进行限制，无需额外条件"],
  "D": ["D1 对公民财产权的限制虽应具有宪法上的正当性，但可由立法机关自由裁量其范围"]
}
```

### 6.5 Strong Solver v3 输出

```json
{
  "protocol_version": "solver-v3",
  "selected_options": ["A"],
  "options": [
    {
      "label": "A",
      "claims": [{
        "claim_id": "A1",
        "verdict": "supported",
        "reason": "征收、征用是国家基于公共利益对既有私有财产权施加的外部限制。",
        "evidence_ids": []
      }],
      "verdict": "supported"
    },
    {
      "label": "B",
      "claims": [{
        "claim_id": "B1",
        "verdict": "contradicted",
        "reason": "征收、征用须依照法律规定，行政法规的授权不能替代法律依据。",
        "evidence_ids": []
      }],
      "verdict": "contradicted"
    },
    {
      "label": "C",
      "claims": [{
        "claim_id": "C1",
        "verdict": "contradicted",
        "reason": "限制财产权除目的正当外，还须满足法律依据、必要性、比例性及相应保障等要求。",
        "evidence_ids": []
      }],
      "verdict": "contradicted"
    },
    {
      "label": "D",
      "claims": [{
        "claim_id": "D1",
        "verdict": "contradicted",
        "reason": "立法机关确定限制范围仍受财产权保障、比例原则等宪法要求约束，并非自由裁量。",
        "evidence_ids": []
      }],
      "verdict": "contradicted"
    }
  ],
  "confidence": 0.98
}
```

### 6.6 Verifier v2 输出

```json
{
  "protocol_version": "verifier-v2",
  "accepted": true,
  "error_codes": [],
  "challenged_options": [],
  "suggested_selected_options": [],
  "note": ""
}
```

### 6.7 最终答案

```json
{
  "question_id": "273",
  "selected_options": ["A"],
  "concise_answer": "A",
  "rationale": "征收、征用是国家基于公共利益对既有私有财产权施加的外部限制。",
  "confidence": 0.98,
  "status": "completed",
  "needs_review": false,
  "warnings": []
}
```

Oracle 标签为 A，Exact Match 为 true。

### 6.8 调用与预算

| 调用 | 模型 | 耗时 | Tokens | Reasoning | Schema |
|---|---|---:|---:|---:|---|
| Controller | Qwen3.7 Flash | 18.62s | 1,770 | 1,024 | valid |
| Solver | GPT-5.6 Sol | 5.44s | 1,441 | 115 | valid |
| Verifier | Qwen3.7 Flash | 17.15s | 1,995 | 1,024 | valid |

最终使用 3/7 calls、5,206/65,536 tokens、41.23/360 秒。

## 7. 代码中每层负责什么

| 层 | 主要文件 | 责任 |
|---|---|---|
| 数据适配 | `legal_mcq/adapter.py` | Request 与 Oracle 隔离 |
| 题目解析 | `legal_mcq/parser.py` | 选项、极性、日期、题型 |
| 协议模型 | `legal_mcq/models.py` | Controller/Solver/Verifier Schema |
| Agent 编排 | `legal_mcq/agent.py` | 状态转移、fallback、Revision |
| 确定性校验 | `legal_mcq/decision.py` | Claim、基数、极性、证据不变量 |
| 客户端 | `legal_mcq/clients.py` | 多角色模型和凭据继承 |
| 预算 | `model.py`、`runner.py` | calls、tokens、deadline、熔断 |
| 日志 | `legal_mcq/observability.py` | 逐调用脱敏记录 |
| CLI | `tools/run_legal_mcq.py` | 数据集运行和评估 |

## 8. 这次运行说明了什么

已经证明：

1. GPT-5.6 Sol 在两个案例中均快速返回合法 Solver v3；
2. 成功题完整覆盖所有选项，不是只解释最终答案；
3. Controller 没有答案权，Claim 身份由代码绑定；
4. Verifier 不能直接改写答案；
5. Oracle 没有进入模型上下文；
6. 角色 timeout 会停止链路，而不是输出不可信答案；
7. 逐调用日志足以复盘模型、Schema、token、耗时和失败阶段。

同时暴露：

1. 原 30 秒 Controller timeout 在供应商延迟波动下会造成整题失败，现已
   调整为 60 秒并通过 ID 472 重放；
2. 本次 2/2 成功题答对只是案例展示，不能推断总体准确率；
3. hidden reasoning 正文不记录，只记录 token 数；
4. closed-book 结果依赖模型参数知识，没有外部法源引用。

## 9. 运行命令与工件

运行命令：

```bash
uv run python tools/run_legal_mcq.py \
  --execute \
  --dataset output/legal_mcq/e2e-showcase-3-20260924/sample.jsonl \
  --config output/legal_mcq/e2e-showcase-3-20260924/config.yaml \
  --output output/legal_mcq/e2e-showcase-3-20260924/results.jsonl \
  --concurrency 1 \
  --log-model-output
```

工件：

- `output/legal_mcq/e2e-showcase-3-20260924/sample.jsonl`
- `output/legal_mcq/e2e-showcase-3-20260924/config.yaml`
- `output/legal_mcq/e2e-showcase-3-20260924/manifest.json`
- `output/legal_mcq/e2e-showcase-3-20260924/results.jsonl`
- `output/legal_mcq/e2e-showcase-3-20260924/results.calls.jsonl`
- `output/legal_mcq/e2e-showcase-3-20260924/results.summary.json`
- `output/legal_mcq/e2e-showcase-3-20260924/analysis.summary.json`
- `output/legal_mcq/e2e-showcase-q472-controller60-20260924/results.jsonl`
- `output/legal_mcq/e2e-showcase-q472-controller60-20260924/results.calls.jsonl`
- `output/legal_mcq/e2e-showcase-q472-controller60-20260924/comparison.summary.json`

模型正文日志已脱敏。独立检查未发现 API key。
