# 定义 agent 作业指导

读原文 → 挖出因子定义 → 写 `extracted.json` → 返回。因子编码不由你做。

---

## 输入

主 agent 的 prompt 给了你：`报告名` / `原文 txt 路径` / `DATE`。

---

## 步骤

### 1. 读原文

用 Read 工具读 `{txt_path}`（已由主进程预提取，无需自己跑命令）。
文本为空 → 直接跳到「返回」并带 error。

### 2. 定义因子

**最多 15 个**。原文没写"因子"二字也要提取：择时阈值→截面排序因子、
行业轮动→行业偏离度、选股逻辑→多维度因子。每条：

- `name`: 英文驼峰
- `description`: 中文
- `formulation`: 完整数学表达式，**从原始数据字段出发，禁止 `f(·)` 占位符**
- `type`: `daily` / `minute` / `cross_section` / `deep_learning`
  - 分钟频 → **一律 `minute`**（没有任何例外；分钟数据上的行业分位/排名也写 `minute`，
    编码时用 minute 模板的可选截面钩子实现）
  - 其他 per-stock → `daily` 或 `minute`

  **判定方法：看 `formulation` 里有没有「跨股票」的量。** 只要出现下面任何一项，
  就必须是 `cross_section`（`daily` 模板只喂**单只股票**的数据，看不到任何同行）：
  - 对**同行/全市场**取统计量：行业均值、行业中位数、行业标准差、市场均值
  - 在**同行/全市场**内做排名、分位数、分层、分组（行业分位、行业内排名、市值分组）
  - 「相对行业」「行业调整」「行业偏离」「行业轮动」——**无论叫不叫分位**，只要减/除/比较的对象是行业聚合值

  ⚠️ **这不是关键词游戏，是公式问题**：`formulation` 里写了 `mean(... | industry)`、
  `rank(... within industry)`、`split by industry` 就是 `cross_section`；
  哪怕因子名叫 `XxxDev`、描述里没出现「行业」二字，也一样。
  反例：`close.pct_change()`、`volume.rolling(20).mean()` 只用到**这一只股票自己**的序列 → `daily`。

  判错代价很大：`daily` 标签 + 行业截面定义 = 因子 worker 拿着单股签名做不出来，
  只能去翻模板源码找答案（实测单因子烧 12M token、24 分钟）。**拿不准就选 `cross_section`。**
- `lookback`: 天数（1月≈20，1季≈60，6月≈120，1年≈250）。
  **只算核心计算需要多少天**；论文末尾的「截面标准化/std20/取波动率」是后处理，不算。
  只用当天数据 → `lookback=1`。
- `source_excerpt`: 从原文直接复制

**不要填 `cols`。**
**不要因为字段可能缺失就丢掉因子** —— 全部照常定义，缺不缺由因子 worker 判。

### 3. 保存

用 Write 工具写 `/tmp/{DATE}_{报告名}_factors.json`（严格照抄 schema，不要去翻 helper 源码）：

```json
{
  "report_name": "报告名",
  "date": "{DATE}",
  "factors": [
    {
      "name": "VolumeRatioConfirm",
      "description": "中文描述",
      "formulation": "完整数学表达式",
      "type": "daily",
      "lookback": 20,
      "source_excerpt": "原文原句"
    }
  ]
}
```

每条**必须**含这 6 个键；`type` 只能是上述四值之一；`lookback` 是整数；**不要**加 `cols`。

```bash
python scripts/claude_factor_helper.py save-extracted --name "{报告名}" --date {DATE} < /tmp/{DATE}_{报告名}_factors.json
# 期望输出：OK: /mnt/d/paper-factor-data/数据仓库/因子产出/extracted_reports/{DATE}/{报告名}.extracted.json
```

---

## 返回（只有一行）

```
{"report": "{报告名}", "total": 12, "error": null}
```

失败（原文读不出、save-extracted 失败等）：`{"report": "...", "total": 0, "error": "一句话原因"}`

**只返回这一行，前后不加任何文字。**
禁止附加：完成说明、因子清单、formulation、代码片段、原文摘录。

---

## 硬约束

1. **不要写因子代码、不要跑 test-and-export、不要派任何子 agent** —— 那是后续 agent 的事
2. **不要 `ls`/`grep` 探索目录**，不要读别的报告
3. **读不到原文 / save-extracted 失败 → 直接带 error 返回，不要重试**
4. 写完定义立刻返回，**不要在返回前做任何额外核对**
