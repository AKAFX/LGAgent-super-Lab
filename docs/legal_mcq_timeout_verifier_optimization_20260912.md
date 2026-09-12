# LegalMCQ Solver 超时与 Verifier 截断优化

## 目标

v3 同题重放中，Solver 的可见截断由 8/8 降到 0/7，但 5/7 调用在 60 秒
内没有返回；Verifier 3 次调用有 1 次 length。v4 针对这两个替代瓶颈，
不增加题级 7 calls、32768 tokens、180 秒上限。

## Solver 超时策略

### 上下文去重

Solver 不再同时接收完整 `question` 和重复的 `question_metadata`。新输入为：

- 一份题干及选项；
- 极性、题型、法域、日期等最小 constraints；
- Controller 的紧凑 plan；
- 可选证据与版本化 skill reference。

Oracle、golden answer 和 annotation 仍不会进入请求。

### 一次主调用和一次异构备用

主 Solver timeout 为 70 秒，`solver_timeout_retries` 固定为 0。超时后不会
再次等待同一 LongCat，而是在同一个 `ExecutionBudget` 中调用一次备用模型：

```yaml
solver_call_timeout_seconds: 70
solver_fallback_timeout_seconds: 45
solver_timeout_retries: 0
solver_timeout_circuit_breaker: 2
solver_fallback_model:
  model_name: Qwen/Qwen3.7-Flash
  max_tokens: 4096
```

备用调用开始前动态扣除 Final Verifier 的 30 秒墙钟；不能保留该时间时直接
跳过。备用调用使用同一 Trace、调用计数、token 预算和 seed 派生规则，
SDK 自动重试仍为 0。使用备用 Solver 后不再启动业务 Revision，避免在
降级路径上继续扩大延迟。

### 批次熔断

`SolverTimeoutCircuitBreaker` 位于 Runner 生命周期内并用锁保护。连续两次
主 Solver timeout 后打开，当前 Runner 后续题直接使用备用模型。主 Solver
成功会在熔断前清零连续 timeout；熔断打开后不在批次内自动恢复。

该机制对 `tools/run_legal_mcq.py` 的共享数据集 Runner 生效。独立创建的
Runner 各自拥有熔断状态，不跨进程持久化。

## Controller v3 确定性 Claim 绑定

真实 fallback 验证发现 Controller v2 可能把一个选项的子命题挂到另一个
选项。v5 链路不再让 Controller 输出 `option_claims`。模型只生成事实、争点
和核验点；代码随后从可信的原始选项文本生成：

```json
{
  "A": ["A1 <完整A项原文>"],
  "B": ["B1 <完整B项原文>"]
}
```

Solver 对每个完整选项只返回对应的唯一 Claim ID，并在 reason 中处理复合
条件。Controller v1/v2 仍可读取，但模型 Claims 在下游使用前会被上述规范
绑定覆盖。

## Solver v3

新输出移除 v2 的 `issues` 和顶层 `rationale`：

```json
{
  "protocol_version": "solver-v3",
  "selected_options": ["C"],
  "options": [
    {
      "label": "C",
      "claims": [
        {
          "claim_id": "C1",
          "verdict": "supported",
          "reason": "决定性适用理由",
          "evidence_ids": []
        }
      ],
      "verdict": "supported"
    }
  ],
  "confidence": 0.9
}
```

最终 rationale 由代码按已选项 claim reasons 组合。v3 的 issue 覆盖不再
单独校验，但 Controller claim ID 必须精确覆盖；v1/v2 继续只读兼容。

## Verifier v2

Verifier 发送 strict JSON Schema：

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

- accepted 时数组和 note 必须为空；
- rejected 时至少一个稳定错误码；
- note 最多 160 字；
- 不允许逐项复述答案。

Verifier 输入只保留一份题目、最小 constraints、Controller issues/claims、
Solver decision、确定性错误码和 open-book 证据，不再发送 skill context
和重复题目 metadata。

可见输出目标为 384 tokens；明确 length 时最多一次 512-token 重生成。
Verifier timeout 为 30 秒。Schema 明确不支持时，auto 模式允许一次受预算
prompt-only 回退；普通格式错误不会触发能力降级。

## 确定性直达修订

若 `DecisionValidator` 已发现选项覆盖、claim ID、极性或证据错误，第一轮
LLM Verifier 不再重复确认。代码直接产生确定性 rejection 并进入唯一一次
Revision；修订后仍不合法则直接输出 partial，不浪费 Final Verifier。

## 配置与回滚

完整示例位于 `examples/parameter/legal2_rag_parameter.yaml`。新增字段：

- 四个角色级 timeout；
- `solver_fallback_model`；
- `solver_timeout_retries` 和 `solver_timeout_circuit_breaker`；
- `verifier_structured_output_mode`；
- `verifier_visible_output_tokens` 和 `verifier_length_retry_tokens`；
- `skip_verifier_on_deterministic_errors`。

删除 `solver_fallback_model` 即关闭备用调用和熔断。设置
`skip_verifier_on_deterministic_errors: false` 可恢复旧核验顺序。
`legal_mcq.enabled: false` 仍是完整回滚开关。

## 验证边界

离线测试覆盖：

- timeout 后不重试主模型，只调用一次备用模型；
- 主/备用共享调用、token、deadline；
- 两次连续 timeout 后第三题绕过主模型；
- 使用备用模型后不进行 Revision；
- Verifier v2 Schema、384→512 length 恢复和显式能力回退；
- 确定性错误跳过第一轮 Verifier；
- Solver v1/v2 读取和 v3 生成；
- OpenAI SDK MockTransport 请求字段和 Oracle 隔离。

这些测试证明编排行为，不证明真实供应商成功率或准确率。下一轮真实验证应
使用新 dev 小样本，分别报告 primary timeout、fallback success、circuit
bypass、Verifier length、端到端 completed 和全部分母准确率。

前沿模型升级后的最终离线回归为 `321 passed`，仅一条第三方 Authlib
弃用警告；compileall 与 `git diff --check` 通过。新冻结 ID 为
`1931b3df9f45caf8f1c3ddc8`，旧 ID `64161b2c27f126a3ffe6d73c`
已归档。v4 初始实现未发起供应商请求；后续真实 fallback 与题 259 重放见
`legal_mcq_fallback_provider_validation_20260912.md`。
