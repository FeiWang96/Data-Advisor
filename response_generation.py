import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


MODEL_ID = "meta-llama/Llama-2-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

raw_queries = []
prompts = []
with open("data/data_advisor_safety_alignment.txt") as f:
    for line in f.readlines():
        chat = [
                {"role": "user", "content": line.strip()},
            ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        prompts.append(prompt)
        raw_queries.append(line.strip())

sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=256)
llm = LLM(model=MODEL_ID, max_model_len=1024)

outputs = llm.generate(prompts, sampling_params)

with open("data/data_advisor_safety_alignment.jsonl", "w") as f:
    for query, output in zip(raw_queries, outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        sample = {
            "prompt": query,
            "response": generated_text
        }
        json.dump(sample, f)
        f.write("\n")

