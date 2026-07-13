#!/usr/bin/env python3
"""
用LLM评价因子代码的逻辑正确性。

输入:
  - 原始研报信息 (factor_description + factor_formulation)
  - 因子代码 (.code.py)
  - 因子元数据

输出:
  - llm_review.json: LLM的评价结果
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


LLM_REVIEW_SYSTEM_PROMPT = """你是一个量化因子审查专家。审查因子的代码实现是否与研报定义一致。

审查依据的优先级：
1. 如果提供了研报原文（source_excerpt），以原文为准 — 原文是研报的直接表述，比二次概括的 description/formulation 更可靠
2. 如果未提供 source_excerpt，使用 description + formulation

判断原则：
- 研报原文通常只有公式和文字描述，不涉及具体数据列名、具体代码实现细节
- 只要因子的核心计算逻辑（公式结构、窗口期、信号方向）与研报一致，就判定为"正确"
- 数据列名不同、具体代码风格差异、中间变量命名不同等均不影响正确性判断
- 因子效果差（IC低、Sharpe低）不是代码实现错误的证据 — 研报本身的因子可能在某些时间段效果就不好

倾向于判定为"正确"，只在逻辑完全不一致时才判定"部分正确"。

只需输出 concise JSON verdict：
{
  "verdict": "正确/部分正确",
  "summary": "<一句话说明代码是否按研报公式正确实现了因子计算，简洁扼要>"
}"""


def build_review_prompt(report_title: str, factor_description: str, factor_formulation: str,
                        variables: dict, code: str, factor_name: str,
                        source_excerpt: str = "") -> str:
    sections = [f"## 因子信息\n- 因子名称: {factor_name}\n- 研报标题: {report_title}\n"]

    if source_excerpt:
        sections.append(f"""## 研报原文（最高优先级）
{source_excerpt}""")
    else:
        sections.append(f"""## 因子描述
{factor_description}

## 因子公式
{factor_formulation}

## 变量说明
{json.dumps(variables, ensure_ascii=False, indent=2) if variables else "无"}""")

    sections.append(f"""## 因子代码
```python
{code}
```

请仔细阅读上述信息，对照研报原文（如有）审查代码实现的逻辑正确性。""")

    return "\n\n".join(sections)


def review_factor_with_llm(report_title: str, factor_description: str, factor_formulation: str,
                           variables: dict, code: str, factor_name: str,
                           source_excerpt: str = "") -> dict:
    """调用LLM对因子进行审查"""
    try:
        from rdagent.oai.llm_utils import APIBackend
    except ImportError:
        print("⚠️ 无法导入rdagent的LLM模块，使用模拟审查")
        return {
            "verdict": "未评估",
            "summary": "LLM审查模块不可用，跳过自动审查",
        }

    try:
        api_backend = APIBackend()
        prompt = build_review_prompt(report_title, factor_description, factor_formulation,
                                     variables, code, factor_name, source_excerpt=source_excerpt)

        response = api_backend.build_messages_and_create_chat_completion(
            user_prompt=prompt,
            system_prompt=LLM_REVIEW_SYSTEM_PROMPT,
            json_mode=True,
            json_target_type=dict,
        )
        return json.loads(response)
    except Exception as e:
        print(f"⚠️ LLM审查失败: {e}")
        return {
            "verdict": "未评估",
            "summary": f"审查过程出错: {str(e)}",
        }


def main():
    parser = argparse.ArgumentParser(description="LLM因子逻辑正确性审查")
    parser.add_argument("meta_path", help="meta.json路径")
    parser.add_argument("code_path", help="code.py路径")
    args = parser.parse_args()

    meta_path = Path(args.meta_path)
    code_path = Path(args.code_path)

    if not meta_path.exists():
        print(f"❌ meta.json不存在: {meta_path}")
        return 1
    if not code_path.exists():
        print(f"❌ code.py不存在: {code_path}")
        return 1

    meta = json.loads(meta_path.read_text())
    code = code_path.read_text()

    result = review_factor_with_llm(
        report_title=meta.get("source_report_title", ""),
        factor_description=meta.get("factor_description", ""),
        factor_formulation=meta.get("factor_formulation", ""),
        variables=meta.get("variables", {}),
        code=code,
        factor_name=meta.get("factor_name", code_path.stem),
        source_excerpt=meta.get("source_excerpt", ""),
    )

    # 输出结果
    print(f"\n{'='*50}")
    print(f"LLM审查结果: {meta.get('factor_name', '')}")
    print(f"{'='*50}")
    print(f"判定: {result.get('verdict', 'N/A')}")
    print(f"总结: {result.get('summary', 'N/A')}")

    # 保存到meta.json
    meta["llm_review"] = result
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"已更新: {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
