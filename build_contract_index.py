import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


from docx import Document as Docx
from pypdf import PdfReader

load_dotenv()

IN_FILE = os.getenv("CONTRACT_PATH", "清新回兰明霞洞自然保护区更名申报书_样例.docx")  # docx/pdf
OUT_DIR = os.getenv("CONTRACT_INDEX_DIR", "outputs/contract_faiss")


def read_docx_paras(path: str) -> list[str]:
    d = Docx(path)
    return [p.text.strip() for p in d.paragraphs if p.text and p.text.strip()]


def read_pdf_blocks(path: str) -> list[str]:
    reader = PdfReader(path)
    blocks: list[str] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        parts = [x.strip() for x in text.split("\n\n") if x.strip()]
        blocks.extend(parts)
    return blocks


def build_ollama_embeddings() -> OllamaEmbeddings:
    model = os.getenv("EMBED_MODEL", "bge-m3").replace(":latest", "") # 兼容 bge-m3:latest
    base_url = os.getenv("EMBED_BASE_URL", "http://127.0.0.1:11434")  # 原生端点不带 /v1
    return OllamaEmbeddings(model=model, base_url=base_url)


def main():
    if not os.path.exists(IN_FILE):
        raise FileNotFoundError(f"CONTRACT file not found: {IN_FILE}")

    if IN_FILE.lower().endswith(".docx"):
        parts = read_docx_paras(IN_FILE)
    elif IN_FILE.lower().endswith(".pdf"):
        parts = read_pdf_blocks(IN_FILE)
    else:
        raise ValueError("Only .docx or .pdf supported for CONTRACT_PATH")

    docs: list[Document] = []
    fname = os.path.basename(IN_FILE)

    for idx, text in enumerate(parts):
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source_file": fname,
                    "chunk_id": f"c{idx:04d}",
                    "approx_loc": f"段落#{idx+1}",
                },
            )
        )

    emb = build_ollama_embeddings()

    # quick sanity check
    _ = emb.embed_query("hello")[:5]
    print("Ollama embedding OK")

    vs = FAISS.from_documents(docs, emb)
    vs.save_local(OUT_DIR)
    print(f"Saved contract FAISS to {OUT_DIR}, chunks={len(docs)}")


if __name__ == "__main__":
    main()
