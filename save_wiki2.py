from datasets import load_dataset

print(">>> Loading small C4 validation dataset...")
ds = load_dataset("brando/small-c4-dataset", split="validation")

print(">>> Number of examples:", len(ds))

text = "\n".join(ds["text"])

with open("ppl_corpus.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Saved ppl_corpus.txt, chars =", len(text))