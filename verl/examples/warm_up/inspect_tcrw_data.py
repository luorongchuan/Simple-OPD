#!/usr/bin/env python3
"""Scan local parquet files for TCRW/Simple-OPD warm-up compatibility.

The script only reads parquet metadata and a tiny sample. It reports whether
prompt, response, and is_correct columns are present, so the user can select
an existing teacher-CoT warm-up file without guessing.
"""

import argparse
from pathlib import Path

import pandas as pd


def inspect(path: Path):
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return {"path": str(path), "error": repr(exc)}

    cols = list(df.columns)
    required = {"prompt", "response"}
    has_required = required.issubset(cols)
    result = {
        "path": str(path),
        "rows": len(df),
        "columns": cols,
        "has_prompt_response": has_required,
        "has_is_correct": "is_correct" in cols,
    }
    if has_required and len(df) > 0:
        sample = df.iloc[0]
        result["prompt_preview"] = str(sample["prompt"])[:100].replace("\n", " ")
        result["response_preview"] = str(sample["response"])[:140].replace("\n", " ")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/luorongchuan/workspace_134/datasets")
    parser.add_argument("--max-files", type=int, default=100)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"dataset root does not exist: {root}")

    files = sorted(root.rglob("*.parquet"))[: args.max_files]
    if not files:
        raise SystemExit(f"no parquet files found under {root}")

    print(f"Scanning {len(files)} parquet files under {root}\n")
    compatible = []
    for path in files:
        info = inspect(path)
        if "error" in info:
            print(f"[ERROR] {path}: {info['error']}")
            continue
        marker = "CANDIDATE" if info["has_prompt_response"] else "skip"
        correct = "+is_correct" if info["has_is_correct"] else "no-is_correct"
        print(f"[{marker:9s}] rows={info['rows']:7d} {correct:13s} {info['path']}")
        print(f"             columns={info['columns']}")
        if info["has_prompt_response"]:
            print(f"             prompt  ={info.get('prompt_preview', '')}")
            print(f"             response={info.get('response_preview', '')}")
            compatible.append(info)

    print("\n============================================================")
    if not compatible:
        print("No parquet with both `prompt` and `response` was found.")
        print("You need to generate Teacher CoT rollouts with the Simple-OPD data pipeline first.")
    else:
        print("TCRW-compatible candidates (prompt + response):")
        for item in compatible:
            suffix = " (supports correct/wrong ablation)" if item["has_is_correct"] else ""
            print(f"  {item['path']}{suffix}")
        print("\nUse one as:")
        print("  SFT_PARQUET=/absolute/path/file.parquet bash run_tcrw_warmup.sh")


if __name__ == "__main__":
    main()
