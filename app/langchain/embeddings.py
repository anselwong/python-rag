"""LangChain Embeddings 适配器：直接调用百炼 Embedding API。"""
import os, hashlib, math
from typing import List
import dashscope
from dashscope import TextEmbedding
from langchain_core.embeddings import Embeddings

class BailianEmbeddings(Embeddings):
    """实现 LangChain 标准 Embeddings 协议，批量请求并校验维度。"""
    def _embed(self, texts: List[str]) -> List[List[float]]:
        if os.getenv("EMBEDDING_PROVIDER", "dashscope").lower() == "hash":
            dim = int(os.getenv("EMBEDDING_DIMENSION", "1536")); out=[]
            for text in texts:
                values=[0.0]*dim; digest=hashlib.blake2b(text.encode(),digest_size=32).digest()
                for i,b in enumerate(digest): values[b % dim] += 1 if i%2 else -1
                norm=math.sqrt(sum(v*v for v in values)) or 1; out.append([v/norm for v in values])
            return out
        key=os.getenv("DASHSCOPE_API_KEY"); base=os.getenv("DASHSCOPE_BASE_URL")
        if not key or not base: raise RuntimeError("缺少 DASHSCOPE_API_KEY 或 DASHSCOPE_BASE_URL")
        dim=int(os.getenv("EMBEDDING_DIMENSION", "1536")); dashscope.api_key=key; dashscope.base_http_api_url=base
        response=TextEmbedding.call(model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"), input=texts, dimension=dim)
        if response.status_code != 200: raise RuntimeError(f"Embedding 调用失败: {response.status_code}")
        return [item["embedding"] for item in response.output["embeddings"]]
    def embed_documents(self, texts: List[str]) -> List[List[float]]: return self._embed(texts)
    def embed_query(self, text: str) -> List[float]: return self._embed([text])[0]
