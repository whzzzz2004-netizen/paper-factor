# /factor — 研报/文章因子端到端处理（三层派发架构）

> **数据根目录**：先运行 `python3 scripts/claude_factor_helper.py data-root` 获取实际路径（本机 D 盘 `/mnt/d/paper-factor-data`；另一台电脑 G 盘自动识别）。下文所有 `/mnt/d/paper-factor-data` 均以该输出为准。

## 用法

- `/factor` — 扫描数据根下的 `papers/inbox/`（默认 D 盘；G 盘自动识别），处理所有未处理项
- `/factor <数据根>/papers/inbox/某篇.pdf` — 处理单个 PDF

### 研报目录：inbox（处理区） / done（归档区）

| 目录 | 用途 |
|---|---|
| `papers/inbox/` | **投放 + 处理区**。放进去就会被处理 |
| `papers/done/{DATE}/` | **归档区**。处理完由 `/factor` 自动移过去 |

**inbox 里的文件一律重新处理**——不看历史、不做去重。投放前先确认 inbox 已清空。

**`/factor` 结束时自动归档**：
```bash
python scripts/claude_factor_helper.py archive-inbox --date {DATE}
```

**只做 `/factor`**（定义+测试+部署代码）。全量数值（parquet/IC/图表）需另跑 `python3 scripts/run_all.py {DATE}`。

## 主 agent 规则

0. **必须用 `claude_factor_helper.py` 的命令**，不得自己写爬虫、装库、手动处理数据
1. **全自动决策**，不问用户
2. **`{DATE}` 在开始前取一次**（`datetime.now().strftime("%Y-%m-%d")`），**全程复用同一个值**。
   不用研报原始日期。**不要在每篇重新计算** —— 跨午夜会算成第二天，产物分裂到两个目录。
3. **每次全量处理**：直接扫 inbox 里所有文件，全部处理。
4. **禁止 Read 图片 / PDF / 大文件（>50KB）**。一次读入就是 24k~154k token，且此后
   **每一轮都重发一遍**。要看图/看长文 → 派子 agent 去看，只要一行结论。
5. **跑批会话只跑批**。开发、调试、看图、读架构文档一律另开会话 ——
   同一会话里混着开发，主上下文会被撑到 200k+ 并在随后几百轮里反复重发
   （实测一次会话主 agent 因此烧掉 123M token，占全会话 82%）。

## 三层派发架构

```
主 agent（只派批，不派篇）
 │  Step 0  扫 inbox + 数据预检 + 预提取 PDF
 │  Step 1  按 10 篇一批切分，**逐个派发**批调度 agent（批之间串行）
 │  Step 2  收统计 → 兜底补报告 → 归档 inbox → 汇报
 │
 └─ 批调度 agent × ⌈N/10⌉   ← 不读研报、不写代码，只派论文 agent
      │  维持 2 篇在飞，滚动补位
      └─ 论文 agent          ← 自己不读原文、不写代码
           │  1. 定义 agent   ← 读原文 → 挖定义 → 写 extracted.json → 返回
           │  2. 因子 worker × 6 ← 谁先完成就补谁
           └─ 写复现报告
```

### 切批规则

```
papers = 扫 inbox 得到的列表
BATCH = 10
batches = [papers[i:i+BATCH] for i in range(0, len(papers), BATCH)]
```

- **不足 10 篇不加批调度层**：只有 1 批时，主 agent **直接派论文 agent**（跳过批调度层），
  自己维持 2 篇在飞、滚动补位（见「单批直连」）。
- 例：24 篇 → 3 批（10 / 10 / 4）。

### 并发参数（本机 32 核 / 31GB 内存）

| 参数 | 默认值 | 说明 |
|---|---|---|
| 批在飞数 | **1** | 批之间**串行**：一个批返回后才派下一个 |
| 批内论文在飞数 | **2** | 批调度内部同时 2 篇 |
| 每篇因子并发 | **6** | 论文 agent 内部 6 个因子 worker |

```bash
export FACTOR_N_WORKERS=2      # 峰值 1×2×6 = 12 job × 2 进程 = 24 进程 / 32 核
```

> ⚠️ `批在飞数 × 批内论文在飞数 × 每篇因子并发 × FACTOR_N_WORKERS` 过大时 CPU 超载、
> 分钟因子可能 OOM。上表是实测可用的配置。

**所有层的阻塞方式：派完立刻结束回合，等系统唤醒。禁 `sleep`、禁 `TaskOutput`。**

---

### Step 0: 扫描 inbox + 数据预检

**先打开探索拦截**（只在本次运行期间生效；平时改代码不受影响）：
```bash
python3 .claude/hooks/factor_gate.py on
rm -f /tmp/.factor_probe_*     # 清上一轮的探测计数（每个因子超 6 次会被直接判失败）
date +%s > /tmp/.factor_run_start   # 本次运行起点：收尾只统计 mtime ≥ 此刻的产物
```

> ⚠️ **同一天重跑时，上一次留下的 `code.py` / `missing.json` / `复现报告.md` 不算本次结果**。
> 运行起点写入 `/tmp/.factor_run_start`，Step 3 与各层子 agent 的磁盘核对都必须按它过滤 mtime。

```bash
DR=$(python3 scripts/claude_factor_helper.py data-root)
ls "$DR"/papers/inbox/*.pdf "$DR"/papers/inbox/*.md 2>/dev/null
```

> ⚠️ **必须同时包含 `.md`**：只扫 `*.pdf` 会静默漏掉 Markdown 研报。

**立刻把本批论文清单快照到 `/tmp/.factor_run_papers`**（Step 3 只认这份清单，
不认归档后的 `done/{DATE}/` —— 那里可能混有同日更早批次的文件）：

```bash
python3 - <<'EOF'
from pathlib import Path
from scripts.data_paths import DATA_ROOT
inbox = DATA_ROOT / "papers" / "inbox"
names = sorted(p.stem for p in inbox.iterdir() if p.suffix.lower() in (".pdf", ".md"))
Path("/tmp/.factor_run_papers").write_text("\n".join(names), encoding="utf-8")
print(len(names), "papers")
EOF
```

**队列排序**：含 "深度学习/GRU/TCN/LSTM/deep_learning" 的排末尾（模型训练类最慢）。

**数据完整性预检（跳过会导致后续跑全量而非测试数据）：**
```bash
python3 - <<'EOF'
import json
from scripts.data_paths import DATA_ROOT
for label, sub in [
    ("日线测试", "数据仓库/行情数据/日线/测试/stock_data/daily"),
    ("分钟测试", "数据仓库/行情数据/分钟线/测试/stock_data"),
]:
    base = DATA_ROOT / sub
    sl = json.load(open(base / "stock_list.json"))
    td = json.load(open(base / "trade_dates.json"))
    print(f"{label}: {len(sl)}只×{len(td)}天")
EOF
```
日线/分钟都必须是 300只×300天，否则**先修复数据再继续**。

**全市场数据（如全市场分钟收益率）需预计算市场代理文件**，检查两个目录：
```bash
DR=$(python3 scripts/claude_factor_helper.py data-root)
ls "$DR"/数据仓库/行情数据/分钟线/测试/stock_data/minute_by_date/market_minute_return.parquet
ls "$DR"/数据仓库/行情数据/分钟线/全量/stock_data/minute_by_date/market_minute_return.parquet
```
缺失就跑（测试 + 全量都要）：
```bash
DR=$(python3 scripts/claude_factor_helper.py data-root)
python scripts/precompute_market_proxy.py --data-dir "$DR/数据仓库/行情数据/分钟线/测试"
python scripts/precompute_market_proxy.py --data-dir "$DR/数据仓库/行情数据/分钟线/全量"
```

---

### Step 1: 预提取原文 + 切批

**主 agent 一次性预提取所有 PDF**（不交给子 agent）：

```bash
rm -rf /tmp/factor_pdf
python scripts/claude_factor_helper.py extract-pdf {全部真实 paper 路径，空格分隔} --outdir /tmp/factor_pdf
# 输出 {"<源路径>": "/tmp/factor_pdf/NN_xxx.txt", ...}
```

> ⚠️ **必须逐个文件路径传参**，不要传目录 —— 传目录会把所有 PDF 合并成**一个** txt。
> 路径含特殊字符时用 `find ... -print0 | xargs -0`。
>
> 抽完**检查每个 txt 非空**（扫描版 PDF 会抽出空文本）。

**然后按 10 篇一批切分**（见「切批规则」）。

---

### Step 2: 逐个派发批调度 agent（批之间串行）

**一次只派 1 个批**，该批返回后再派下一个批。

**用 Agent 工具，`subagent_type: "factor-batch"`**，prompt 只有这几行：

```
你是批次调度 agent，自己不读研报、不写代码。
Read `.claude/skills/factor/batch_worker.md` 并严格照做，不做文件之外的事。

### 本批参数
- 批号: {批号}
- DATE: {DATE}
- 本批研报（报告名 → 预提取的原文 txt 路径）：
  {报告名1} → {txt_path1}
  {报告名2} → {txt_path2}
  ...

完成后**只返回这一行 JSON，前后不加任何文字**：
{"batch": "{批号}", "results": [{"report": "...", "total": 0, "ok": 0, "missing": 0, "error": null}, ...]}
```

**派发循环**：

```
all_results = []
for batch in batches:
    派发 batch（run_in_background=true）
    **立即结束回合**（本回合不得再调用任何工具）
    被唤醒 → 记下这个批的 results，继续下一个批
```

> ⚠️ **派发后立刻结束回合，本回合不得再调用任何工具** —— 连一次 `ls` 都不要，
> 不要"顺便看一眼进度"。每一次工具调用都要把你（主 agent）整段上下文重发一遍，
> 是全场最贵的操作。核对留到被系统唤醒后的下一回合。
>
> ⚠️ 系统会在**每个**后台子 agent 完成时唤醒你一次，机制是通的，不需要自己盯着。
> **不要 `sleep N` 轮询**；**不要用 `TaskOutput`**（子 agent 层没有这个工具）。
>
> ⚠️ **某批超时（>10 分钟无响应）→ 记该批为失败，继续下一个批。绝不重派**。
>
> ⚠️ 批调度返回值格式不对（解析不出 `results`）时，**不要追问、不要重派**，
> 该批按失败记，最后靠 Step 3 兜底。

#### 单批直连（只有 1 批时）

**只有 1 批就不派批调度 agent**，但**仍然派论文 agent**（`subagent_type: "factor-paper"`）——
主 agent 只维持 2 篇在飞、收到完成就补下一篇，**自己不跑论文级的核对循环**。

> ⚠️ 别让主 agent 亲自当论文 agent：那会把每个因子 worker 的完成、每次磁盘核对
> 都灌进主上下文（实测一次 52 个子 agent 的单批直连跑，主 agent 自己被唤醒 536 轮、
> 烧 123M token）。论文 agent 的上下文是干净的，把它放在中间层最省。

论文 agent 的 prompt（**只有这几行**）：

```
你是单篇研报处理 agent，按 /factor 流程处理这一篇。
Read `.claude/skills/factor/paper_worker.md` 并严格照做，不做文件之外的事。

### 本篇参数
- 报告名: {报告名}
- 原文 txt 路径: {txt_path}
- DATE: {DATE}

完成后**只返回这一行 JSON，前后不加任何文字**：
{"report": "{报告名}", "total": 0, "ok": 0, "missing": 0, "error": null}
```

---

### Step 3: 收尾（**必须从磁盘核对，不只信 agent 返回值**）

agent 可能返回格式不对、超时、或静默失败。**所有产物都落盘了，所以以磁盘为准**：

```bash
# 1) 磁盘核对：以 Step 0 扫到的 inbox 清单为基准逐篇对照
#    （必须在归档之前跑；若已归档，改用 done/{DATE}/ 当基准）
#    ⚠️ 只统计本次运行的产物（mtime ≥ /tmp/.factor_run_start）；同日重跑的旧产物一律不算
python3 - <<'EOF'
import os, json
from pathlib import Path
from scripts.data_paths import DATA_ROOT
D = "{DATE}"
START = float(Path("/tmp/.factor_run_start").read_text().strip())
snap = Path("/tmp/.factor_run_papers")
if snap.exists() and snap.read_text().strip():
    papers = [x for x in snap.read_text().splitlines() if x.strip()]
else:
    root = DATA_ROOT / "papers"
    inbox = root / "inbox"
    base = inbox if (inbox.is_dir() and any(inbox.iterdir())) else root / "done" / D
    papers = sorted(p.stem for p in base.iterdir()
                    if p.suffix.lower() in (".pdf", ".md")) if base.is_dir() else []
ex = DATA_ROOT / "数据仓库" / "因子产出" / "extracted_reports" / D
ts = DATA_ROOT / "数据仓库" / "因子产出" / "测试" / D

def fresh(p):
    return p.exists() and p.stat().st_mtime >= START

defined = enc = mis = nov = repro = 0
fails = []
for r in papers:
    ej = ex / f"{r}.extracted.json"
    if not fresh(ej):
        fails.append((r, "定义失败")); continue
    defined += 1
    try:
        factors = [x["name"] for x in json.load(open(ej)).get("factors", []) if x.get("name")]
    except Exception:
        fails.append((r, "定义文件损坏")); continue
    d = ts / r
    for n in factors:
        if fresh(d / n / f"{n}.code.py"): enc += 1
        elif fresh(d / n / f"{n}.missing.json"): mis += 1
        else: nov += 1
    if fresh(d / "复现报告.md"): repro += 1
    else: fails.append((r, "缺复现报告"))
print(f"论文={len(papers)} 定义={defined} 编码={enc} 缺字段={mis} 无产出={nov} 复现报告={repro}")
print("失败清单: 无" if not fails else "失败清单: " + "; ".join(f"{w}={r}" for r, w in fails))
EOF

# 2) 兜底：补生成缺失或过期的复现报告
python3 - <<'EOF'
import os, subprocess
from pathlib import Path
from scripts.data_paths import DATA_ROOT
D = "{DATE}"
START = float(Path("/tmp/.factor_run_start").read_text().strip())
ts = DATA_ROOT / "数据仓库" / "因子产出" / "测试" / D
for r in sorted(os.listdir(ts)) if ts.is_dir() else []:
    d = ts / r
    rep = d / "复现报告.md"
    if d.is_dir() and not (rep.exists() and rep.stat().st_mtime >= START):
        print("补生成复现报告:", r)
        subprocess.run(["python", "scripts/claude_factor_helper.py",
                        "write-repro-report", "--date", D, "--report", r])
EOF

# 3) 归档 inbox
python scripts/claude_factor_helper.py archive-inbox --date {DATE}

# 4) 关闭探索拦截（必须执行）
python3 .claude/hooks/factor_gate.py off
```

**汇报：只输出下面这个格式，不加任何其他文字**（不解释、不总结、不列举因子名/formulation/代码）。

```
批次：{成功批数}/{总批数}
论文：{N} 篇（定义 {a} / 编码 {b} / 缺字段 {c} / 无产出 {d}）
复现报告：{e}/{N}
失败清单：{无 | 逐行 "原因: 报告名"}
耗时：{x} 分钟
```

**磁盘核对与 agent 报数不一致时，以磁盘为准**，在该行末尾加 `*`，并在最后补一行
`* 磁盘核对与 agent 报数不一致：{一句话}`。除此之外不加任何内容。

> ⚠️ **只报告本次运行的结果**：`定义/编码/缺字段/无产出/复现报告` 全部按
> `mtime ≥ /tmp/.factor_run_start` 统计。同日重跑时，上一次的产物既不算编码，也不算缺字段。

> ⚠️ **跨午夜时先查 `{DATE+1}` 目录**。有就按 no-clobber + md5 合并回 `{DATE}`
> （同名同 hash 去重，不同 hash 保留 `{DATE}` 版），再删空的 `{DATE+1}` 目录。

---

### 并行全量计算（自动重扫）

想在 `/factor` 生成因子的同时让另一个 dialog 自动跑全量：`/all 2026-08-16`。
两边互不打断。
