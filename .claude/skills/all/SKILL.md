---
name: all
description: 全量运行指定日期下所有因子（逐个计算+评估+绘图）
---

# /all — 全量运行某日期下所有因子

跑 `run_all.py`：扫描该日期目录下每个有 `.code.py` 的因子，逐个算全量 →
评估 → 绘图 → Barra，跑完汇总。状态（无 parquet 跑全量 / 有 parquet 看能否
更新 / 已最新跳过）由 run_all 自己判，**agent 不要手工复算、不要读别的日期目录**。

## 执行

**严格按用户给的日期跑，参数是什么就执行什么**——不要去别的日期目录核对
「跑过没有」，也不要替用户判断该跑哪天。结果是什么就报什么（哪怕是「无待处理」）。

两步缺一不可：

**1. 受管后台启动**（Bash `run_in_background=true`，命令不加 `&`/`nohup`/`setsid`）：

```bash
python3 -u scripts/run_all.py {date}
```

加后台符号会让进程脱离 harness，跑完无通知 → 漏报。

**2. 立即排定时自醒**（无条件，不管因子多少）：

```
ScheduleWakeup(delaySeconds=1200, noop=false,
  prompt="/all {date} 中途汇报：读 /tmp/run_all_progress.json 汇报进度")
```

每次唤醒：读 `/tmp/run_all_progress.json` 用一两行汇报（完成 N/总数、成功/失败、
失败清单）→ 未到 `n/n` 就再排一次 → 到 `n/n` 或进程已退出就报最终结果并停止排期。

## 进度与失败在哪看

- `/tmp/run_all_progress.json`：进度、成功/失败计数、`failures[]`、耗时。
- `/tmp/{因子}.run.log`：单个因子的完整日志（失败原因看末尾）。
- **慢 ≠ 卡死**：分钟因子可能跑很久（实测近 2 小时）。看 progress 的 `updated_at`
  长期不变才算卡死。

## 跑完之后

逐因子查产出：parquet 存在、非空率正常（daily/minute >60%、cross_section >50%）、
有 decile.png。失败因子 → 派 agent 查根因 → 修核心函数 → `test-and-export` +
`deploy-to-full` → 单跑 `python scripts/run_factor_full.py <code.py>` 验证。
