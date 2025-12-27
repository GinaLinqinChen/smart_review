import os, json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

OUT_DIR = "outputs/law_faiss"
QUERY = "更改名称 条款 条件 程序 申请 审批 抄报 备案 公告"
K = 5

def main():
    emb = OllamaEmbeddings(
        model=os.environ["EMBED_MODEL"].replace(":latest",""),  # bge-m3
        base_url="http://127.0.0.1:11434"
    )
    vs = FAISS.load_local(OUT_DIR, emb, allow_dangerous_deserialization=True)
    docs = vs.similarity_search(QUERY, k=K)

    out = []
    for d in docs:
        out.append({"article_no": d.metadata.get("article_no"), "text": d.page_content})
    with open("outputs/retrieved_articles.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved outputs/retrieved_articles.json")

if __name__ == "__main__":
    main()
