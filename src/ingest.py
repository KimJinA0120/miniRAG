# 인덱싱 파이프라인: 로딩 → 청킹 → 임베딩 → Chroma 저장
# 실행: python src/ingest.py
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # config.py를 import하기 위해
import config

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# [1. 로딩] PDF를 페이지 단위 Document 리스트로 읽는다.
# 각 Document의 metadata["page"]에 0부터 시작하는 페이지 번호가 들어 있다.
def load_pdf(path):
    pages = PyPDFLoader(str(path)).load()
    print(f"[로딩] {len(pages)}페이지")
    return pages


# [2. 청킹] 페이지 텍스트를 chunk_size 글자 이하 조각으로 자른다.
# 문단(\n\n) → 줄(\n) → 공백 순으로 자를 곳을 찾아서 문장이 덜 끊기게 한다.
# 페이지별로 자르므로 각 청크는 원래 페이지 번호(metadata)를 그대로 물려받는다.
def split_pages(pages, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"[청킹] {len(chunks)}개 청크 (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# [3. 임베딩 + 4. 저장] 청크를 bge-m3로 벡터화해서 Chroma 컬렉션에 저장한다.
# 같은 이름의 컬렉션이 있으면 지우고 새로 만든다 (중복 저장 방지).
def save_to_chroma(chunks, name):
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL)
    db = Chroma(
        collection_name=name,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )
    db.delete_collection()
    db = Chroma(
        collection_name=name,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )
    start = time.time()
    db.add_documents(chunks)  # 이 안에서 임베딩 호출 + 저장이 같이 일어난다
    print(f"[임베딩/저장] 컬렉션 '{name}'에 저장, {time.time() - start:.1f}초")


if __name__ == "__main__":
    pages = load_pdf(config.PDF_PATH)
    chunks = split_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    save_to_chroma(chunks, config.collection_name())
