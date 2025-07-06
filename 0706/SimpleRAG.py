import numpy as np
import json
from typing import List, Union
from openai import OpenAI

# 文本分块：将长文本分割为指定大小和重叠的块
def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return chunks

# 文本嵌入：将文本转换为向量嵌入
def embed_func(text: Union[str, List[str]], model_name: str, api_key: str,
               base_url: str, batch_size: int = 32) -> List[List[float]]:
    client = OpenAI(api_key=api_key, base_url=base_url)
    if isinstance(text, str):
        text = [text]
    embeddings = []
    for i in range(0, len(text), batch_size):
        batch = text[i:i + batch_size]
        response = client.embeddings.create(
            model=model_name,
            input=batch
        )
        batch_embeddings = [data.embedding for data in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings

# 向量存储：存储向量数据到文件
def store_vec(data: List[dict], path: str) -> None:
    # 存储向量数据到JSON文件
    # 数据结构: [{"content": "文本", "embedding": [0.1, 0.2, ...]}, ...]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 向量加载：从文件加载向量数据
def load_vec(path: str) -> List[dict]:
    # 从JSON文件加载向量数据
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 检索：使用余弦相似度检索最相关的top_k个文档索引
def retrieve(query: List[float], vec_db: List[List[float]], top_k: int) -> List[int]:
    # numpy/sklearn实现计算余弦相似度: query(1×d) vs vec_db(n×d) → sim(1×n)
    # 使用np.argsort获取top_k索引
    query_vec = np.array(query).reshape(1, -1)
    db_matrix = np.array(vec_db)
    dot_product = np.dot(db_matrix, query_vec.T).flatten()  # 点积
    query_norm = np.linalg.norm(query_vec)
    db_norms = np.linalg.norm(db_matrix, axis=1)  # 模长
    cosine_sim = dot_product / (db_norms * query_norm)
    # 获取top_k索引
    top_indices = np.argsort(cosine_sim)[::-1][:top_k]
    return top_indices.tolist()

# 生成：使用语言模型生成文本
def gen_func(prompt: str, model_name: str, api_key: str, base_url: str) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

class SimpleRAG:
    def __init__(self, embed_model: str, gen_model: str, api_key: str, base_url: str):
        self.embed_model = embed_model
        self.gen_model = gen_model
        self.api_key = api_key
        self.base_url = base_url
        self.vector_db = []

    def index_documents(self, file_path: str, chunk_size=500, overlap=50):
        # 构建知识库索引
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = split_text(text, chunk_size, overlap)
        embeddings = embed_func(
            chunks, self.embed_model, self.api_key, self.base_url
        )
        self.vector_db = [
            {"content": chunk, "embedding": emb}
            for chunk, emb in zip(chunks, embeddings)
        ]

    def save_index(self, path: str):
        store_vec(self.vector_db, path)

    def load_index(self, path: str):
        self.vector_db = load_vec(path)

    def query(self, question: str, top_k=5) -> str:
        question_embed = embed_func(
            question, self.embed_model, self.api_key, self.base_url
        )[0]
        all_embeddings = [item["embedding"] for item in self.vector_db]
        top_indices = retrieve(question_embed, all_embeddings, top_k)
        context = "\n\n".join([
            f"[文档 {i + 1}]: {self.vector_db[idx]['content']}"
            for i, idx in enumerate(top_indices)
        ])
        prompt = f"""基于以下上下文回答问题：
{context}
问题：{question}
答案："""
        return gen_func(prompt, self.gen_model, self.api_key, self.base_url)


if __name__ == "__main__":
    # API_KEY = "your_api_key"
    API_KEY = "sk-"
    # BASE_URL = "https://api.openai.com/v1"
    BASE_URL = "https://www.chataiapi.com/v1"
    EMBED_MODEL = "text-embedding-3-small"
    GEN_MODEL = "gpt-3.5-turbo"
    DATA_FILE = "wd.txt"
    INDEX_FILE = "vector_db1.json"

    # 初始化RAG系统
    rag = SimpleRAG(EMBED_MODEL, GEN_MODEL, API_KEY, BASE_URL)

    # 构建索引（首次运行）
    rag.index_documents(DATA_FILE)
    rag.save_index(INDEX_FILE)

    # 加载索引（后续使用）
    # rag.load_index(INDEX_FILE)

    # 执行查询
    question = "在构造散列函数时需要注意什么？"
    answer = rag.query(question)
    print(f"问题: {question}\n答案: {answer}")
