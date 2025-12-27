import os, re
from dotenv import load_dotenv
from docx import Document as Docx
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

LAW_PATH = "data/国家级自然保护区范围调整和功能区调整及更改名称管理规定.docx"
OUT_DIR = "outputs/law_faiss"

def read_docx(path: str) -> str:
    d = Docx(path)
    return "\n".join(p.text.strip() for p in d.paragraphs if p.text.strip())

def split_by_article(text: str):
    # 按“第X条”切分（保守写法）
    pattern = r"(第[一二三四五六七八九十百千0-9]+条)"
    parts = re.split(pattern, text)
    # parts: [前言, 第X条, 内容, 第Y条, 内容...]
    chunks = []
    for i in range(1, len(parts), 2):
        article = parts[i]
        body = parts[i+1].strip() if i+1 < len(parts) else ""
        if body:
            chunks.append((article, f"{article}\n{body}"))
    return chunks

def main():
    text = read_docx(LAW_PATH)
    chunks = split_by_article(text)

    docs = []
    for article_no, content in chunks:
        docs.append(Document(
            page_content=content,
            metadata={"law_name": "国家级自然保护区范围调整和功能区调整及更改名称管理规定", "article_no": article_no}
        ))


    emb = OllamaEmbeddings(
        model=os.environ["EMBED_MODEL"].replace(":latest",""),  # bge-m3
        base_url="http://127.0.0.1:11434"
    )

    print("Embedding test:", emb.embed_query("hello")[:5])

    vs = FAISS.from_documents(docs, emb)
    vs.save_local(OUT_DIR)
    print(f"Saved FAISS to {OUT_DIR}, docs={len(docs)}")

if __name__ == "__main__":
    main()
