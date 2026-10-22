# 실험용 설정값 모음. 실험할 때는 이 파일만 바꾼다.
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# [로딩] 원본 PDF 경로
PDF_PATH = BASE_DIR / "data" / "doc.pdf"

# [청킹] 청크 크기(글자 수)와 겹치는 길이
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# [임베딩/생성] Ollama 모델 이름
EMBED_MODEL = "bge-m3"
LLM_MODEL = "qwen3:4b-instruct"
OLLAMA_URL = "http://localhost:11434"

# [저장] Chroma 저장 폴더. 컬렉션 이름에 청킹 설정을 넣어 설정별로 따로 저장한다.
CHROMA_DIR = BASE_DIR / "chroma_db"


def collection_name(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    return f"chunk{chunk_size}_overlap{chunk_overlap}"


# [검색] 몇 개의 청크를 가져올지
TOP_K = 3

# [실험] run_eval.py가 비교할 청킹 설정 (크기 × overlap 조합을 모두 돈다)
EXP_CHUNK_SIZES = [300, 600, 1000]
EXP_OVERLAPS = [0, 100]
QUESTIONS_PATH = BASE_DIR / "experiments" / "questions.csv"
RESULTS_PATH = BASE_DIR / "experiments" / "results.csv"
