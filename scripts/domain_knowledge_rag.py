"""
领域知识 RAG 模块
==================
基于 TF-IDF + jieba 中文分词的知识检索。
- Index: .claude/skills/factor/knowledge/*.md 中的 A 股市场规则
- Retrieval: cosine similarity on TF-IDF vectors
- 零向量数据库依赖，无需 GPU

用法:
    from domain_knowledge_rag import DomainKnowledgeRAG
    rag = DomainKnowledgeRAG()
    rag.build_index()          # 加载知识库 + 建索引
    results = rag.retrieve("涨停怎么判断")  # 检索
    print(results[0]["text"])
"""

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _chunk_markdown(text: str, source: str) -> list[dict[str, str]]:
    """按 ## 标题切分 markdown 文档为语义块"""
    lines = text.split("\n")
    chunks: list[dict[str, str]] = []
    current_heading = "(前言)"
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("## "):
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    chunks.append({"text": body, "source": source, "heading": current_heading})
            current_heading = line.lstrip("# ")  # e.g., "涨停规则"
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            chunks.append({"text": body, "source": source, "heading": current_heading})

    return chunks


def _chinese_tokenizer(text: str) -> list[str]:
    """jieba 中文分词，保留数字和英文 token"""
    words = jieba.lcut(text)
    # 过滤纯空格，保留中文词、英文、数字
    return [w.strip() for w in words if w.strip()]


class DomainKnowledgeRAG:
    """

    领域知识 RAG。
    用法:
        rag.
        knowledge_rag = DomainKnowledgeRAG()
        rag.load_chunks_and_build()      # 一步加载+建索引
        results = rag.retrieve("科创板涨跌停限制是多少", top_k=3)
    """

    def __init__(self, knowledge_dir: str | None = None, index_cache: str | None = None):
        self.knowledge_dir = Path(knowledge_dir or __file__).resolve().parent.parent / ".claude" / "skills" / "factor" / "knowledge"
        self.index_cache = Path(index_cache or Path(__file__).resolve().parent / ".rag_index.pkl")
        self.chunks: list[dict[str, str]] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._vectors: np.ndarray | None = None  # (n_chunks, n_features)
        self._chunk_texts: list[str] | None = None

    # ── 加载 + 切分 ──

    def load_and_chunk(self) -> list[dict[str, str]]:
        """加载所有 .md 知识文件并按 ## 标题切分"""
        if not self.knowledge_dir.exists():
            raise FileNotFoundError(f"知识库目录不存在: {self.knowledge_dir}")

        chunks: list[dict[str, str]] = []
        for fpath in sorted(self.knowledge_dir.glob("*.md")):
            text = fpath.read_text(encoding="utf-8")
            file_chunks = _chunk_markdown(text, source=fpath.name)
            chunks.extend(file_chunks)

        self.chunks = chunks
        return chunks

    # ── 建索引 ──

    def build_index(self) -> None:
        """加载知识 → jieba 分词 → TF-IDF 向量化"""
        if not self.chunks:
            self.load_and_chunk()

        self._chunk_texts = [c["text"] for c in self.chunks]

        self._vectorizer = TfidfVectorizer(
            tokenizer=_chinese_tokenizer,
            max_features=5000,
        )
        self._vectors = self._vectorizer.fit_transform(self._chunk_texts).toarray()

    # ── 检索 ──

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[dict[str, Any]]:
        """检索与 query 最相关的 top_k 个知识块"""
        if self._vectorizer is None or self._vectors is None:
            raise RuntimeError("请先调用 build_index()")

        q_vec = self._vectorizer.transform([query]).toarray()
        sims = cosine_similarity(q_vec, self._vectors).flatten()
        top_idx = sims.argsort()[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for idx in top_idx:
            score = float(sims[idx])
            if score < min_score:
                continue
            results.append({**self.chunks[idx], "score": round(score, 4)})
        return results

    # ── 持久化 ──

    def save_index(self, path: str | None = None) -> str:
        """持久化索引到磁盘"""
        dst = Path(path or self.index_cache)
        dst.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "chunks": self.chunks,
            "vectorizer": self._vectorizer,
            "vectors": self._vectors,
            "chunk_texts": self._chunk_texts,
        }
        dst.write_bytes(pickle.dumps(data))
        return str(dst)

    def load_index(self, path: str | None = None) -> bool:
        """从磁盘加载持久化索引。返回是否成功"""
        src = Path(path or self.index_cache)
        if not src.exists():
            return False

        try:
            data = pickle.loads(src.read_bytes())
        except (pickle.UnpicklingError, AttributeError, ImportError, ModuleNotFoundError, TypeError):
            # pickle 跨模块命名空间不匹配（如 __main__ vs domain_knowledge_rag）
            return False
        self.chunks = data["chunks"]
        self._vectorizer = data["vectorizer"]
        self._vectors = data["vectors"]
        self._chunk_texts = data["chunk_texts"]
        return True

    # ── 快捷方法 ──

    def load_chunks_and_build(self, force_rebuild: bool = False) -> None:
        """加载+建索引（有缓存读缓存）"""
        if not force_rebuild and self.load_index():
            return
        self.build_index()
        self.save_index()

    def info(self) -> dict:
        """打印知识库统计"""
        if not self.chunks:
            self.load_and_chunk()
        sources = {}
        for c in self.chunks:
            sources[c["source"]] = sources.get(c["source"], 0) + 1
        return {
            "total_chunks": len(self.chunks),
            "sources": sources,
            "index_cache": str(self.index_cache),
            "knowledge_dir": str(self.knowledge_dir),
        }


# ── CLI ──

def cli():
    """命令入口: python -m domain_knowledge_rag <command> [args]"""
    rag = DomainKnowledgeRAG()

    if len(sys.argv) < 2:
        print("用法:")
        print("  python scripts/domain_knowledge_rag.py build          # 建索引")
        print("  python scripts/domain_knowledge_rag.py info           # 知识库统计")
        print("  python scripts/domain_knowledge_rag.py query <text>   # 检索")
        return

    cmd = sys.argv[1]

    if cmd == "build":
        rag.load_chunks_and_build(force_rebuild=True)
        info = rag.info()
        print(f"✅ 索引已构建: {info['total_chunks']} 个知识块")
        for src, n in info["sources"].items():
            print(f"   {src}: {n} 块")

    elif cmd == "info":
        rag.load_and_chunk()
        info = rag.info()
        print(f"知识库: {info['knowledge_dir']}")
        print(f"总块数: {info['total_chunks']}")
        print(f"来源:")
        for src, n in info["sources"].items():
            print(f"   {src}: {n} 块")

    elif cmd == "query":
        if len(sys.argv) < 3:
            print("用法: python scripts/domain_knowledge_rag.py query <检索词>")
            return
        query = " ".join(sys.argv[2:])
        rag.load_chunks_and_build()
        results = rag.retrieve(query, top_k=5)
        print(f"\n检索: 「{query}」\n")
        for r in results:
            print(f"[{r['score']:.3f}] {r['source']} > {r['heading']}")
            print(f"   {r['text'][:150]}...")
            print()

    else:
        print(f"未知命令: {cmd}")


if __name__ == "__main__":
    cli()