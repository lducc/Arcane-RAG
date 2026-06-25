import json
import pickle
import re
import zipfile
from pathlib import Path

INDEX_DIR = Path("data/processed/faiss_index")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
TARGET_ARTICLES = 5


def is_valid_article(article):
    return "|Điều " in article


def strip_soluong(name):
    return re.sub(r'\s+số$', '', name)


def main():
    results = json.loads(Path("data/results.json").read_text())

    with open(INDEX_DIR / "offsets.pkl", "rb") as file:
        offsets = pickle.load(file)

    needed_articles = set()
    fallback_articles = set()
    for result in results:
        for article in result.get("predicted_articles", []):
            if article:
                if is_valid_article(article):
                    needed_articles.add(article)
                else:
                    fallback_articles.add(article)

    article_texts = {}
    all_needed = needed_articles | fallback_articles
    with CHUNKS_PATH.open() as file:
        for idx in range(len(offsets)):
            file.seek(offsets[idx])
            item = json.loads(file.readline())
            article = item.get("relevant_article", "")
            if article in all_needed:
                current_text = article_texts.get(article, "")
                if len(current_text) < 4000:
                    separator = "\n" if current_text else ""
                    article_texts[article] = current_text + separator + item.get("text", "")

    submissions = []
    for result in results:
        all_articles = result.get("predicted_articles", [])
        raw_valid = [a for a in all_articles if a and is_valid_article(a)][:TARGET_ARTICLES]
        if not raw_valid:
            raw_valid = [a for a in all_articles if a][:TARGET_ARTICLES]
        valid_articles = []
        for a in raw_valid:
            parts = a.split("|")
            if len(parts) >= 3:
                valid_articles.append(f"{parts[0]}|{strip_soluong(parts[1])}|{parts[2]}")
            else:
                valid_articles.append(a)

        documents = {}
        for article in valid_articles:
            parts = article.split("|")
            if len(parts) >= 2:
                doc_key = f"{parts[0]}|{strip_soluong(parts[1])}"
                documents[doc_key] = True

        answer_parts = []
        total_len = 0
        for article in raw_valid:
            text = article_texts.get(article, "")
            if text and total_len < 4000:
                answer_parts.append(text)
                total_len += len(text)

        submissions.append({
            "id": result["id"],
            "question": result["question"],
            "answer": "\n".join(answer_parts)[:4000],
            "relevant_docs": list(documents),
            "relevant_articles": valid_articles,
        })

    Path("results.json").write_text(
        json.dumps(submissions, ensure_ascii=False)
    )

    with zipfile.ZipFile("submission.zip", "w") as zip_file:
        zip_file.write("results.json")

    valid_count = sum(1 for s in submissions if any(is_valid_article(a) for a in s["relevant_articles"]))
    print(f"Done — {len(submissions)} entries ({valid_count} with valid articles) -> submission.zip")


if __name__ == "__main__":
    main()
