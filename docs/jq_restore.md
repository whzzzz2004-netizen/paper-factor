# 聚宽（JQData）文档恢复手册

> **本文件不是给 Phase 2 因子 agent 看的。** 放在 `docs/` 下，不在
> `.claude/skills/factor/knowledge/` 里，所以 `/factor` 流程不会读到它。
>
> **背景（2026-09-27）**：`get_jq_data` 实测不可用 —— 聚宽凭据被服务端拒绝
> （`dataapi.joinquant.com/v2/apis` 返回 `用户不存在或密码错误`，与故意传错口令的
> 报错文案完全一致）。为避免误导，已把所有面向 agent 的 jq 说明撤下，只保留
> `INDUSTRY_DICT`。模板源码里的 `def get_jq_data` **未改动**，代码能力还在。
>
> **恢复时机**：确认聚宽账号可正常认证之后（见文末「恢复前自检」）。

## 一键恢复 / 撤回

已经写好脚本，**不用手工改文件**：

```bash
python docs/jq_restore.py --check    # 先看当前处于哪个状态
python docs/jq_restore.py            # 恢复：把 get_jq_data 说明加回所有文件
python docs/jq_restore.py --revert   # 撤回：再删掉，回到只有 INDUSTRY_DICT 的状态
```

脚本覆盖 7 个文件的 9 处改动（helper 输出、4 个 knowledge md、phase2_check.md ×2、SKILL.md）。
特性：

- **幂等**：已是目标状态就跳过，重复执行安全
- **可逆且逐字节精确**：`--revert` 后与原文件完全一致（往返 3 次已实测）
- **状态感知**：`--check` 会报告每个文件是「恢复态 / 撤下态 / 混合态」

改完务必复验：

```bash
python scripts/claude_factor_helper.py show-columns --type daily_single
python scripts/claude_factor_helper.py show-columns --type minute
```

---

## 一、权威内容（三处已提交来源，随时可取回原文）

撤下的文档文字**从未提交进 git**，无法用 `git checkout` 找回。但内容本身有三个
已提交的权威来源，恢复时照抄即可：

| # | 来源 | 含什么 | 取法 |
|---|---|---|---|
| ① | `rdagent/app/qlib_rd_loop/paper_factor_knowledge_graph.yaml` | `index_data` + `jqdata_fallback` 两个完整节点 | 见下方「1.1」 |
| ② | `rdagent/components/coder/factor_coder/prompts.yaml` | 最全的一份，含完整指数代码表 | 见下方「1.2」 |
| ③ | `factor.py` 的 4 个模板 | `def get_jq_data` docstring + 用法示例 | `git show HEAD:rdagent/components/coder/factor_coder/factor.py` |

### 1.1 knowledge_graph.yaml 原文（190-251 行）

```yaml
  index_data:
    title: 指数数据
    aliases:
      - 基准指数
      - 指数收益
      - 市场收益
      - 超额收益
      - beta
      - 相对强弱
      - 中证500
      - 沪深300
      - 上证指数
      - ZZ500
      - CSI300
      - benchmark
      - index return
      - market return
    requires:
      - local_data_discipline
    content: |-
      指数数据通过聚宽（JoinQuant）接口按需下载，首次调用后自动缓存为 parquet 文件。
      框架提供 get_jq_data(symbol, data_type) 函数，获取指数行情用 get_jq_data(symbol, 'price')。
      symbol 使用聚宽格式：'000905.XSHG'=中证500, '000300.XSHG'=沪深300, '000001.XSHG'=上证指数, '399006.XSHE'=创业板指。
      返回 DataFrame，列：open, close, high, low, volume, money。
      用途：市场收益、超额收益、Beta、相对强弱、CAPM 模型等需要基准指数的因子。
      注意：首次调用需要联网下载，后续调用直接读缓存，无网络开销。

  jqdata_fallback:
    title: 聚宽通用数据获取
    aliases:
      - 成分股
      - 指数成分
      - 中证500成分
      - 沪深300成分
      - 创业板成分
      - 行业成分
      - 板块成分
      - 指数列表
      - 股票列表
      - 聚宽
      - JoinQuant
      - jqdatasdk
      - 聚宽数据
      - 在线数据
      - 联网获取
      - 下载数据
      - 数据缺失
      - 本地没有
      - 本地不存在
      - additional data
      - index constituents
      - index members
      - stock list
    requires:
      - local_data_discipline
    content: |-
      当因子需要本地数据中不存在的字段或标的时，可使用 get_jq_data(symbol, data_type) 通过聚宽在线获取并自动缓存。
      - get_jq_data('000905.XSHG', 'index_components') 返回中证500成分股列表
      - get_jq_data('600519.XSHG', 'price') 返回个股行情（本地无此标的时使用）
      - 优先检查本地缓存，已有则跳过下载。
      - 需要 JQ_USER / JQ_PASS 环境变量（已注入 Docker 容器）。
```

> ⚠️ 末句「已注入 Docker 容器」是旧环境的说法，本项目**不用 Docker**，恢复时应改写。
> 另外 `index_data` 节点缺 `'000016.XSHG'`(上证50) 等代码，完整表见 1.2。

### 1.2 prompts.yaml 原文（377-383 行，指数代码表最全）

```
  1. **行业分类**：`INDUSTRY_DICT` 字典，key 为股票代码，value 为申万一级行业名（如 "银行I"、"食品饮料I"）。用法：`industry = INDUSTRY_DICT.get(stock, "未知")`
  2. **聚宽通用数据获取**：`get_jq_data(symbol, data_type)` 函数，自动缓存，首次调用联网下载。
     - `get_jq_data('000300.XSHG', 'price')` 获取指数行情
     - `get_jq_data('000905.XSHG', 'index_components')` 获取指数成分股列表
     - 优先读本地缓存，没有再通过聚宽在线下载。需要 JQ_USER / JQ_PASS 环境变量。
     - **常用指数代码**：000300.XSHG(沪深300)、000905.XSHG(中证500)、000016.XSHG(上证50)、
       000688.XSHG(科创50)、000001.XSHG(上证指数)、399001.XSHE(深证成指)、399006.XSHE(创业板指)、
       000852.XSHG(中证1000)、000906.XSHG(中证800)、000015.XSHG(上证红利)、932056.XSHG(中证2000)
```

`prompts.yaml` 另有两处规则条目（415-417 行附近）：

```
  16. **行业数据**：`INDUSTRY_DICT` 可用于行业中性化、行业分组、行业内排名等。用 `INDUSTRY_DICT.get(stock, "未知")` 获取行业。
  17. **指数数据**：`get_jq_data(symbol, 'price')` 可用于计算超额收益、Beta、市场收益等。首次调用自动下载，后续读缓存。
```

> ⚠️ **`prompts.yaml` 我只在撤下时没动它** —— 也就是说，如果走的是
> `evolving_strategy_factor_implementation_v1` 这条老的 LLM 路径，
> jq 说明**现在仍然在 prompt 里**。恢复时要一并检查（见「2.6」）。

---

## 二、恢复步骤（逐处，含精确锚点）

恢复时**改写**：所有 `需要 JQ_USER / JQ_PASS 环境变量（已配置）` →
`需要有效的 JQ_USER / JQ_PASS 环境变量`。

### 2.1 `scripts/claude_factor_helper.py` — `show-columns` 输出

当前内容（第 79-87 行）：

```python
EXTRA_DATA_TEXT = """\
额外可用数据（框架已注入，函数内可直接使用，不属于 parquet 列）：
  INDUSTRY_DICT                   申万一级行业字典，INDUSTRY_DICT[股票代码] = 行业名（如 "银行I"）。
                                  数据文件 industry.json，31 个申万一级行业，覆盖本地全部股票。
                                  用途：行业中性化、行业分组统计、行业内排名、行业动量/轮动。
                                  用法：industry = INDUSTRY_DICT.get(stock, "未知")

注：需要指数行情 / 指数成分股等其他在线数据时，本地不可用；按缺字段处理即可。
"""
```

**恢复动作**：删除末行「注：…」，在 `INDUSTRY_DICT` 四行之后插入：

```
  get_jq_data(symbol, data_type)  聚宽在线数据，首次调用下载后自动缓存为 parquet，之后读缓存。
                                  data_type='price'            → 指数/个股行情 DataFrame，列 open/close/high/low/volume/money
                                  data_type='index_components' → 指数成分股列表 DataFrame，列 stock
                                  常用指数代码：000300.XSHG(沪深300)、000905.XSHG(中证500)、
                                    000016.XSHG(上证50)、000852.XSHG(中证1000)、000906.XSHG(中证800)、
                                    000001.XSHG(上证指数)、399001.XSHE(深证成指)、399006.XSHE(创业板指)
                                  用途：市场收益、超额收益、Beta、相对强弱、CAPM、指数成分股筛选。
                                  需要有效的 JQ_USER / JQ_PASS 环境变量。
```

### 2.2 `knowledge/daily.md` — 「额外可用数据」段

在 `INDUSTRY_DICT` 小节之后、「指数行情、指数成分股等其他在线数据本地不可用…」
那句之前，插入 `### get_jq_data(symbol, data_type) — 聚宽数据（指数行情 / 成分股）`
小节，含：缓存说明表（data_type → 返回 → 说明）、两个调用示例、常用指数代码表
（5 列 × 2 行的表格）、用途、环境变量、以及前瞻约束提示：

```
- ⚠️ **前瞻约束照旧**：拿到的指数行情要按 T 日截断（`idx[idx.index <= trade_date]`），
  不得把 T 日之后的行情用于 T 日因子值
```

最后把「指数行情、指数成分股等其他在线数据本地不可用…」整句**删除**。

### 2.3 `knowledge/cross_section.md`

同 2.2。另需在缺列段落补一行「指数行情 / 指数成分股 → 用 `get_jq_data(...)`，
不写进 `--cols`」。

### 2.4 `knowledge/minute.md`

同 2.2，额外两条：

```
- ⚠️ 返回的是**日频**数据，与分钟因子的时间粒度不同，需自行按日对齐
```

并把特殊约束第 5 条改回：
`**禁止读日线 parquet 数据列**（分钟模板只提供分钟列）；但上面的 \`INDUSTRY_DICT\` /
\`get_jq_data\` 是框架注入的额外数据，可以用`

### 2.5 `knowledge/deep_learning.md`

同 2.2，用途写作「市场收益、超额收益、Beta、相对强弱等作为模型输入特征」，
前瞻约束改为「训练集只能用到截止日为止的数据」。

### 2.6 `phase2_check.md` + `SKILL.md`

> 注：Phase 2 作业指导原先在单个 `phase2_prompt.md`，2026-09-27 拆成
> `phase2_check.md`（字段核对）+ `phase2_code.md`（写码测试）以省 token。
> jq 相关文字在 **`phase2_check.md`** 里。

- `phase2_check.md` 缺列判定处：在「额外可用数据」提示块补回 `get_jq_data` 一条；
  「字段齐全」判定处补回「指数行情 / 指数成分股 → 用 `get_jq_data(...)`，不写进 `--cols`」。
- `SKILL.md` 规则 19 补回 `get_jq_data` 条目，并把「其他在线数据本地不可用」那半句删掉。
- **检查 `rdagent/components/coder/factor_coder/prompts.yaml`**（规则 16/17 与
  「额外可用数据」段）—— 撤下时**未改动**，确认是否与恢复后的文档口径一致。

### 2.7 改完必做

```bash
python scripts/claude_factor_helper.py show-columns --type daily_single
python scripts/claude_factor_helper.py show-columns --type minute
```

确认「额外可用数据」段同时含 `INDUSTRY_DICT` 和 `get_jq_data`，且**无**「本地不可用」字样。

---

## 三、恢复前自检（先确认凭据真的能用）

```bash
timeout 120 python3 - <<'PY'
import os
import jqdatasdk as jq
jq.auth(os.environ.get("JQ_USER", ""), os.environ.get("JQ_PASS", ""))
print("AUTH OK, quota:", jq.get_query_count())
print(jq.get_index_stocks("000905.XSHG")[:3])
PY
```

**通过标准**：打印 `AUTH OK` 并返回成分股前 3 个。

如果仍然报 `用户不存在或密码错误`：

1. 对照**故意传错口令**的报错文案 —— 若两种文案完全相同，说明是账号/密码本身的问题，
   不是权限未开通。
2. 凭据位置一共四处，必须同步更新：
   - `.env` 的 `JQDATA_USERNAME` / `JQDATA_PASSWORD`
   - shell 环境变量 `JQ_USER` / `JQ_PASS`（模板读的是这两个）
   - 另有旧副本：`~/RD-Agent/.env`、`~/paper_factor_project/.env`
3. 网络本身是通的（`dataapi.joinquant.com:443` 可连、`/v2/apis` 有响应），
   且 `jqdatasdk 1.9.8` 走的正是 v2 接口，**无需降级或换 SDK**。

另可先用一条最小调用验证写入路径可用（缓存会落到
`{日线数据目录}/stock_data/daily/jq_*.parquet`）：

```bash
timeout 300 python3 - <<'PY'
import subprocess, sys, os, tempfile
from pathlib import Path
sys.path.insert(0, "/home/dministrator/paper-factor")
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace as F
code = "def calc_factor_series(df, stock):\n    return df['close']\n"
full = F._build_factor_code(F.DAILY_FRAMEWORK_TEMPLATE, code, 20, ["close"])
d = Path(tempfile.mkdtemp()); p = d / "m.py"
p.write_text(full.split("# <<<USER_CODE_START>>>")[0], encoding="utf-8")
env = os.environ.copy()
env["FACTOR_DATA_DIR"] = "/mnt/d/paper-factor-data/数据仓库/行情数据/日线/测试"
r = subprocess.run([sys.executable, "-c",
    f"import runpy; g=runpy.run_path({str(p)!r}); "
    "idx=g['get_jq_data']('000300.XSHG','price'); print(idx.shape, list(idx.columns))"],
    capture_output=True, text=True, env=env, timeout=280)
print(r.stdout or r.stderr[-500:])
PY
```

---

## 四、技术背景（恢复时无需改动，仅供排查）

- **模板里的 `def get_jq_data` 共 4 处**（`factor.py` 的 DAILY / MINUTE /
  CROSS_SECTION / DEEP_LEARNING 模板），**本次一行未改**。缓存键为
  `jq_{data_type}_{md5(symbol)[:8]}.parquet`，写在各自模板的
  `STOCK_DATA_DIR`（日线/截面/DL）或分钟模板解析出的 `_DAILY_DATA_DIR` 下。
- **失败方式**：凭据无效时**抛异常**，不会静默返回空 DataFrame —— 不会让因子悄悄
  算出全 NaN。已实测确认。
- **零存量依赖**：AST 扫过全部 480 个已部署 `.code.py`，**真正调用 `get_jq_data`
  的因子数为 0**。表面"命中"（339 个）全部是模板自带的 `def` 和它 docstring 里的
  示例行。因此撤下文档不会让任何存量因子失效。
- **分钟模板的行业分类是本次修的另一处 bug**：`INDUSTRY_DICT` 原先恒为空
  （分钟线目录下没有 `industry.json`），已改为回退到同级「日线」目录，实测
  测试集 292 只 / 全量 5207 只。这部分与 jq 无关，**恢复 jq 时不要动**。
