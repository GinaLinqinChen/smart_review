from dotenv import load_dotenv
from docx import Document
from langchain_core.documents import Document as LCDocument
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

print("All imports OK")
