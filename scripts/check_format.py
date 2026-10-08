"""Check the saved single-text ChatML dataset without changing it."""

import argparse
import re

ASSISTANT_MARKER = "<|im_start|>assistant\n"


def extract_completion(row):
    text = row.get("text", "")
    if not isinstance(text, str) or ASSISTANT_MARKER not in text:
        raise ValueError("missing assistant segment in text")
    completion, end_marker, trailing = text.split(ASSISTANT_MARKER, 1)[1].partition("<|im_end|>")
    if not end_marker or trailing.strip():
        raise ValueError("invalid assistant end marker")
    return completion.strip()


def main():
    from datasets import load_from_disk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="./local_luogu_dataset")
    args = parser.parse_args()
    ds = load_from_disk(args.path)
    if hasattr(ds, "keys") and "train" in ds:
        ds = ds["train"]
    print(f"Dataset: {args.path}, examples: {len(ds)}, columns: {ds.column_names}")
    invalid = 0
    for i, row in enumerate(ds):
        try:
            completion = extract_completion(row)
            if not completion:
                raise ValueError("empty assistant answer")
        except ValueError as error:
            invalid += 1
            print(f"{i}: INVALID: {error}")
            continue
        if i < 10:
            has_fence = bool(re.search(r"```", completion))
            has_code = any(k in completion for k in ("#include", "int main", "using namespace std"))
            print(f"{i}: fence={has_fence} code={has_code}")
            print(completion[:2000].replace("\n", "\\n"))
    print(f"Invalid examples: {invalid}/{len(ds)}")
    return int(invalid > 0)


if __name__ == "__main__":
    raise SystemExit(main())
