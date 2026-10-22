# 질의 파이프라인: 질문 → 검색 → 답변 생성 (출처 페이지 포함)
# 실행: python src/query.py "질문 내용"
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # config.py를 import하기 위해
import config

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


# [검색] 질문을 같은 임베딩 모델로 벡터화하고, 가장 가까운 청크 top-k개를 가져온다.
# name: 어떤 청킹 설정의 컬렉션에서 찾을지 (실험 때 바꿔 끼운다)
def retrieve(question, k=config.TOP_K, name=None):
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL)
    db = Chroma(
        collection_name=name or config.collection_name(),
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )
    # 점수는 거리(distance)라서 작을수록 질문과 가깝다
    return db.similarity_search_with_score(question, k=k)


# [생성] 검색된 청크를 프롬프트에 넣고 LLM에게 답하게 한다.
# 문서에 없는 내용은 지어내지 말라고 지시해서 환각을 줄인다.
def generate(question, docs):
    context = "\n\n".join(
        f"[p.{doc.metadata['page'] + 1}]\n{doc.page_content}" for doc in docs
    )
    prompt = f"""아래 문서 내용만 근거로 질문에 한국어로 답하세요.
문서에 답이 없으면 "문서에서 찾을 수 없습니다"라고 답하세요.
답변 끝에 근거가 된 페이지를 [p.번호] 형식으로 적으세요.

### 문서
{context}

### 질문
{question}"""
    llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_URL, temperature=0)
    return llm.invoke(prompt).content


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else input("질문: ")

    start = time.time()
    results = retrieve(question)
    print(f"\n[검색] {time.time() - start:.1f}초, top-{len(results)}")
    for doc, score in results:
        page = doc.metadata["page"] + 1  # 사람이 보는 페이지 번호(1부터)
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  p.{page} (거리 {score:.3f}) {preview}...")

    start = time.time()
    answer = generate(question, [doc for doc, _ in results])
    print(f"\n[생성] {time.time() - start:.1f}초\n{answer}")
