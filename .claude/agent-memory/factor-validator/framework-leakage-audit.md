---
name: framework-leakage-audit
description: 模板前瞻偏差审计结论（2026-08-01）：框架切片安全；唯一泄露是 CompositeDailyFactor 全序列 zscore，已修复；新增"标准化只能截面"规则
metadata:
  type: project
---

# 模板框架前瞻偏差审计（2026-08-01）

审计对象：`rdagent/components/coder/factor_coder/factor.py` 中 `FactorFBWorkspace` 的 5 个模板。

## 结论

- **框架切片边界全部安全**：`searchsorted(side='right') + iloc[start:pos]` 语义正确（数值验证过），T 日因子只用 ≤T 数据；停牌日处理安全；涨停剔除用当日收盘已知信息；滑动窗口 `_uniq_dates[start:i+1]` 含当日不含之后。
- **唯一真实泄露**：`CompositeDailyFactor`（20260730/idea__0）在向量化 `calc_factor_series` 里用**全序列 mean/std 做时序 zscore**，T 日值受未来数据影响。
- **根因**：因子描述只说"四维度 zscore 标准化后等权相加"，没规定做法；LLM 选了**唯一会泄露的"全历史时序 zscore"**。量化惯例：zscore 一定是**截面**（当日全市场），绝不做时序。

## 已做修复

1. **CompositeDailyFactor**：`.code.py` 的 `_zscore_series` 改为 expanding 因果版本（数值验证：追加未来观测后历史值 diff=0）。已 deploy 到全量目录 `文献因子_全量/20260730/idea__0/`。**注意**：这仍是时序标准化（因果安全但不符合量化惯例），若按惯例应重写为 cross_section 截面因子——用户暂未要求，待定。
2. **DEEP_LEARNING 模板 off-by-one**：训练切片 `side='right'`（含当年第一个交易日）→ 新增 `_stock_positions_train` 用 `side='left'`，训练严格早于当年第一天。预测切片不变（仍 ≤T）。
3. **规则新增**（防未来再犯）：
   - `prompts.yaml` `evolving_strategy_factor_implementation_v1_system` 规则18：标准化只能是截面 zscore，绝不能是时序标准化。
   - `.claude/skills/factor/SKILL.md` 编码硬约束 14：同规则。

## 已决定不做

- **静态扫描**：用户明确不需要（审计只发现一处错误，用规则预防即可）。正则版有 19% 误报（`np.std`/`groupby().mean()`/日内切片等因果写法会被误拦），已从 `claude_factor_helper.py` 完全移除。

## 易泄露场景（若将来做防护的参考）

1. 向量化 `calc_factor_series`：shift(-1)、bfill、center=True、expanding/全序列 mean/std/zscore。
2. 截面/分钟截面：用户绕过传入的 `ad[s]`/`grp`，直接调 `load_stock`/`get_jq_data` 拿全历史。
3. DL：`predict_batch` 内部状态化更新用了 T 之后信息。
