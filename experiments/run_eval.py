# 청킹 실험: 설정별로 [검색]만 돌려서 Hit@k를 잰다 (LLM 생성은 호출하지 않음)
# 실행: python experiments/run_eval.py            → config의 모든 조합
#       python experiments/run_eval.py 600 100    → 한 설정만
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))
import config
from ingest import load_pdf, split_pages, save_to_chroma
from query import retrieve

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# [저장] 이 설정의 컬렉션이 이미 있는지 확인 (있으면 인덱싱을 건너뛰고 재사용)
def collection_exists(name):
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL)
    db = Chroma(collection_name=name, embedding_function=embeddings,
                persist_directory=str(config.CHROMA_DIR))
    return len(db.get(limit=1)["ids"]) > 0


# 질문 파일 읽기. pages는 "19;38" → {19, 38} (비어 있으면 문서에 답이 없는 질문)
def load_questions():
    with open(config.QUESTIONS_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["gold"] = {int(p) for p in row["pages"].split(";") if p}
    return rows


# [검색 평가] 질문마다 top-k 청크의 페이지를 보고, 정답 페이지가 몇 위에 처음 나오는지(rank) 기록
# rank가 있으면 Hit@k, rank == 1이면 Hit@1. 순위까지 보는 이유: 문서가 작아 Hit@3는 모든 설정이 만점
def evaluate(name, questions):
    results = []
    for q in questions:
        docs = retrieve(q["question"], k=config.TOP_K, name=name)
        top_pages = [doc.metadata["page"] + 1 for doc, _ in docs]  # 1부터 시작하는 페이지
        ranks = [i + 1 for i, p in enumerate(top_pages) if p in q["gold"]]
        rank = ranks[0] if ranks else 0  # 0 = top-k 안에 정답 페이지 없음
        hit = "" if not q["gold"] else int(rank > 0)
        results.append({
            "id": q["id"], "type": q["type"],
            "gold_pages": q["pages"],
            "top_pages": ";".join(map(str, top_pages)),
            "distances": ";".join(f"{score:.3f}" for _, score in docs),
            "hit": hit,
            "rank": "" if not q["gold"] else rank,
        })
    return results


if __name__ == "__main__":
    if len(sys.argv) == 3:
        settings = [(int(sys.argv[1]), int(sys.argv[2]))]
    else:
        settings = [(s, o) for s in config.EXP_CHUNK_SIZES for o in config.EXP_OVERLAPS]

    questions = load_questions()
    pages = None
    all_rows, summary = [], []

    for size, overlap in settings:
        name = config.collection_name(size, overlap)

        # [로딩 → 청킹 → 임베딩 → 저장] 컬렉션이 없을 때만 새로 만든다
        if not collection_exists(name):
            pages = pages or load_pdf(config.PDF_PATH)
            save_to_chroma(split_pages(pages, size, overlap), name)

        start = time.time()
        rows = evaluate(name, questions)
        scored = [r for r in rows if r["hit"] != ""]  # 답이 없는 질문은 Hit@k 계산에서 제외
        hits = sum(r["hit"] for r in scored)
        hit1 = sum(r["rank"] == 1 for r in scored)
        # MRR: 정답 순위의 역수 평균 (1위=1, 2위=0.5, 3위=0.33, 못 찾음=0)
        mrr = sum(1 / r["rank"] for r in scored if r["rank"]) / len(scored)
        print(f"\n== {name}: Hit@{config.TOP_K} = {hits}/{len(scored)}, Hit@1 = {hit1}/{len(scored)}, "
              f"MRR = {mrr:.2f} ({time.time() - start:.1f}초)")
        for r in rows:
            mark = {1: "O", 0: "X", "": "-"}[r["hit"]]
            print(f"  [{mark}] Q{r['id']:>2} 정답 p.{r['gold_pages'] or '-':<8} 검색 p.{r['top_pages']:<10} {r['rank'] or ''}위")

        all_rows += [{"chunk_size": size, "overlap": overlap, **r} for r in rows]
        summary.append((name, hits, hit1, mrr, len(scored)))

    # 결과 기록: 설정 × 질문 한 줄씩
    with open(config.RESULTS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n== 요약")
    for name, hits, hit1, mrr, total in summary:
        print(f"  {name:<22} Hit@{config.TOP_K} {hits}/{total}  Hit@1 {hit1}/{total}  MRR {mrr:.2f}")
    print(f"결과 저장: {config.RESULTS_PATH}")
