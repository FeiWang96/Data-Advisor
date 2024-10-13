import json
from datasets import load_dataset


dataset = load_dataset("mlabonne/alpagasus", split="train")

with open("data/alpagasus.jsonl", "w") as f:
    for sample in dataset:
        if len(sample["input"]) < 1:
            prompt = sample["instruction"]
        else: 
            prompt = sample["instruction"] + " " + sample["input"]

        sample = {
            "prompt": prompt,
            "response": sample["output"]
        }

        json.dump(sample, f)
        f.write("\n")

