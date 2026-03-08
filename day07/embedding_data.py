import pandas as pd
from langchain_ollama import OllamaEmbeddings

df = pd.read_json('FailureSensorIQ/failuresensoriq_standard/all.jsonl',lines=True)
# 定义拼接函数
def concatenate_row(row):
    return ' '.join([str(value) for value in row])
texts = df.apply(concatenate_row, axis=1).to_list()


embeddings = OllamaEmbeddings(model="qwen3-embedding")

batch_size = 100  # 根据模型和硬件调整
vectors = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    batch_vectors = embeddings.embed_documents(batch)
    vectors.extend(batch_vectors)
print(vectors[10])  # 打印第11个向量