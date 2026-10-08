"""Convert Luogu preference data to the single-text ChatML training format."""

INSTRUCTION = """你是一个C++代码生成器，专为信息学竞赛普及组难度的题目设计。请严格遵守：
1. 仅输出C++代码，不包含任何其他文本。
2. 代码必须正确、无编译错误、逻辑正确。
3. 代码应简洁、朴实，避免冗余。
4. 在代码中，对于关键步骤，用注释简要说明数学本质（如：// 数学本质：求最大公约数）。
5. 不输出任何思考过程或解释。
题目内容如下：
"""


def convert(example):
    """Preserve the full statement and extract code after its opening fence."""
    invalid = {"text": "", "valid": False}
    try:
        statement = ""
        for conversation in example.get("conversations", []):
            value = conversation.get("value", "").strip()
            if "题目描述" in value:
                statement = value
                break
        if not statement:
            return invalid

        answer = example.get("chosen", {}).get("value", "").strip()
        start = answer.find("#include")
        if start == -1:
            return invalid
        # The first fence may open the block; only search after the code starts.
        end = answer.find("```", start)
        completion = answer[start:end if end != -1 else len(answer)].strip()
        if not completion:
            return invalid

        prompt = INSTRUCTION + statement
        text = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{completion}<|im_end|>"
        )
        return {"text": text, "valid": True}
    except (AttributeError, TypeError):
        return invalid


def main():
    from datasets import DatasetDict, load_from_disk

    dataset = load_from_disk("./local_luogu_dpo")["train"]
    mapped = dataset.map(convert, batched=False, remove_columns=dataset.column_names)
    print("After map, columns:", mapped.column_names, "len=", len(mapped))
    mapped = mapped.filter(lambda row: row["valid"])
    print("After filter, len=", len(mapped))
    mapped = mapped.remove_columns(["valid"])
    out = DatasetDict({"train": mapped})
    out.save_to_disk("./local_luogu_dataset")
    print("Saved converted dataset with train len=", len(out["train"]))


if __name__ == "__main__":
    main()
