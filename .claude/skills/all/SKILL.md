---
name: all
description: 全量运行指定日期下所有因子（一次扫描跑完，逐个计算+评估+绘图）
---

# /all — 全量运行某日期下的所有因子

启动 `run_all.py`，**一次扫描**该日期目录下所有因子，逐个跑（全量计算 + 评估 +
绘图 + Barra），跑完汇总结束。**不做自动重扫**——要补跑失败的因子直接再执行一次
（已有 parquet 的会自动跳过）。

## 用法

```bash
python3 scripts/run_all.py                 # 最近日期目录
python3 scripts/run_all.py 2026-10-04      # 指定日期
python3 scripts/run_all.py --report 研报名  # 只跑匹配的研报
python3 scripts/run_all.py --force         # 强制重跑（无视已有 parquet）
python3 scripts/run_all.py --dry-run       # 只看计划
```

## 执行（关键：让 harness 管住进程，跑完能唤醒你）

**不要**用 `nohup ... &` / `setsid ... &` / 末尾加 `&`。那样进程会脱离 harness，
跑完**不会有任何通知**，你会一声不吭地漏报。

**正确做法**：用 Bash 工具的 `run_in_background=true` 跑，命令本身**不加**任何
后台符号（不要 `&`、`nohup`、`setsid`）：

```bash
python3 -u scripts/run_all.py {args}
```

这样 harness 把进程当受管后台任务，**它真正结束时唤醒你一次**，你就能汇报。
（`-u` 保证无缓冲，输出及时落盘到 harness 的任务输出文件。）

**然后必须再排一次定时自醒**做中途汇报——否则只有结束时才汇报一次：

```
ScheduleWakeup(delaySeconds=1200, prompt="/all {args} 继续", noop=false)
```

每被唤醒一次就：读 `/tmp/run_all_progress.json` 汇报一行进度 → 若未完再排下一次
`ScheduleWakeup`；若 `progress` 已到 `N/N`（进程已退出）则汇报最终结果、
**不再排下一次**（结束）。

### 进度与失败怎么看

- 每完成一个因子打印一行到日志：`✅ [3/18] 报告/因子  (累计 成功3 失败0 跳过0, 12.3min)`。
- **实时进度**：`cat /tmp/run_all_progress.json`（进度/成功/失败/跳过/失败清单/耗时）。
- **失败原因**：进度文件 `failures[]`，或该因子的 `/tmp/{因子}.run.log` 末尾。
- **慢 ≠ 卡死**：有些因子本来要跑几分钟到几十分钟（实测有个分钟因子跑了近 2 小时），
  别因为慢就杀它。判卡死看进度文件 `updated_at` 是否长期不变。

## 说明

- 扫描 `因子产出/全量/{DATE}/` 下每个「有 .code.py」的因子子目录。
- 状态判定：无 parquet → 全量计算；有 parquet 但日期落后 → 增量补算；已最新 → 跳过。
- 单个因子失败**不中断队列**，记入失败清单后继续下一个。**天然断点续跑**：
  成功的因子会跳过，所以修复失败因子后直接重跑同一天即可。
- `run_factor_full.py`：单因子版本（`python scripts/run_factor_full.py <code.py>`），
  用于修完某个因子后快速验证非空率，不必等全量跑完。

## ⚠️ 失败因子处理

跑完后逐因子检查产出：parquet 存在、非空率正常（minute/daily >60%、cross_section >50%）、
有 decile.png。失败因子 → 派 agent 分析根因 → 修核心函数 → `test-and-export` +
`deploy-to-full` → 重跑该因子（重跑 `run_all.py` 或单跑 `run_factor_full.py`）。

常见失败模式：
- **数据 NaN**（如非行情列空）→ 检查 `非行情数据/` 是否有效
- **模板 bug**（如 cross_section 索引列）→ 修模板后清 `_template_cache`
- **因子算法**（全量规模退化、日期对齐、O(N³) 过慢）→ 改核心函数
