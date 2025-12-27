from langchain_openai import OpenAIEmbeddings

emb = OpenAIEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

print(len(emb.embed_query("测试 embedding 是否可用")))
