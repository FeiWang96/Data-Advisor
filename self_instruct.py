from tqdm import tqdm
import random
from vllm import LLM, SamplingParams


data_generation_template = """
You are asking a chatbot to help you do something harmful. 
Come up with a series of questions. One question per line.
Some example questions are:
{}
The new question is:

"""

data = []
with open("data/seed_data.txt") as f:
    for line in f.readlines():
        data.append(line.strip())

data_generation_sampling_params = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=256)
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", max_model_len=8192)

data_file = open("data/self_instruct_safety_alignment.txt", "w")

for i in tqdm(range(1000)):
    prompts = []
    for j in range(12):
        examples = random.sample(data, 5)
        in_context_str = ""
        for i in range(len(examples)):
            in_context_str += f"{examples[i]}\n"
        data_generation_prompt = data_generation_template.format(in_context_str)
        prompts.append(data_generation_prompt)
    batch_outputs = llm.generate(prompts, data_generation_sampling_params, use_tqdm=False)

    new_data = []
    for instance in batch_outputs:
        for x in instance.outputs:
            q = x.text.split("\n")[0].strip().strip("\"")
            if len(q) < 20: continue
            if q[0] == "[":
                q = q.split("\",")[0].strip("[").strip("]").strip("\"")
                if len(q) < 20: continue
            new_data.append(q)

    new_data_str = '\n'.join(new_data)
    data_file.write(f"{new_data_str}\n")
    data_file.flush()

    data.append(new_data)


data_file.close()


