import json
import re
from pathlib import Path

INPUT = Path("data/processed/documents.jsonl")
OUTPUT = Path("data/processed/chunks.jsonl")
ARTICLE_RE = re.compile(r"Điều\s+([0-9]+[a-zA-ZđĐ]?)\s*[\.:]", re.I)
PART_RE = re.compile(r"(?=\s(?:[0-9]+\.|[a-zđ]\)))", re.I)
MAX_WORDS = 700
CHUNK_WORDS = 450
OVERLAP = 80

def get_field(value):
    #Get a string from a field.  EX: ["80/2021/NĐ-CP"] -> "80/2021/NĐ-CP"
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    return str(value).strip() if value else ""

def split_tokens(text):
    return text.split()

def word_chunks(text):
    #Split text into overlapping word windows
    ws = split_tokens(text)
    step = CHUNK_WORDS - OVERLAP
    for start in range(0, len(ws), step):
        yield " ".join(ws[start:start + CHUNK_WORDS])

def articles(text):
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        yield "full_text", text.strip()
        return
    if matches[0].start() > 0:
        yield "preamble", text[:matches[0].start()].strip()
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield f"Điều {match.group(1)}", text[start:end].strip()

def split_long_article(text):
    if len(split_tokens(text)) <= MAX_WORDS:
        yield text
        return

    parts = [x.strip() for x in PART_RE.split(text) if x.strip()]
    if len(parts) <= 1:
        yield from word_chunks(text)
        return

    chunk = ""
    for part in parts:
        part_words = split_tokens(part)
        if len(part_words) > CHUNK_WORDS:
            if chunk:
                yield chunk.strip()
                chunk = ""
            yield from word_chunks(part)
            continue

        candidate = f"{chunk} {part}".strip()
        if len(split_tokens(candidate)) > CHUNK_WORDS and chunk:
            yield chunk.strip()
            chunk = part
        else:
            chunk = candidate

    if chunk:
        yield chunk

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with INPUT.open() as src, OUTPUT.open("w") as dst:
        for row in src:
            doc = json.loads(row)
            text = get_field(doc.get("markdown"))
            law_id = get_field(doc.get("doc_number"))
            title = get_field(doc.get("title"))
            legal_type = get_field(doc.get("legal_type"))
            if not text or not law_id or not title:
                continue

            doc_name = " ".join(x for x in [legal_type, law_id, title] if x)
            for article, article_text in articles(text):
                for i, chunk in enumerate(split_long_article(article_text), 1):
                    out = {
                        "chunk_id": f"{law_id}|{article}|{i}",
                        "law_id": law_id,
                        "doc_name": doc_name,
                        "article": article,
                        "text": chunk,
                        "source_url": get_field(doc.get("source_url")),
                        "relevant_doc": f"{law_id}|{doc_name}",
                        "relevant_article": f"{law_id}|{doc_name}|{article}",
                    }
                    dst.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
