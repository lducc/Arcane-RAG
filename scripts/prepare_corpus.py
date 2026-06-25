import json, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv

OUTPUT = Path("data/processed/documents.jsonl")
load_dotenv()
token = os.getenv("HF_TOKEN")

def main():

    rows = load_dataset("tmquan/vbpl-vn", "documents", split="train", streaming = True, token=token)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with OUTPUT.open("w") as f:
        for row in rows:
            n += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # print(n)

if __name__ == "__main__":
    main()
