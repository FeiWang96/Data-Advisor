import argparse
import json
from vllm import LLM, SamplingParams
from datasets import load_dataset
from vllm.lora.request import LoRARequest
from .parse_prediction import parse_multi_choice_response

def generate_response(args):
    prompts = []
    dataset = load_dataset("cais/mmlu", 'all', split="validation")

    for sample in dataset:
        option_str = '\nChoices: '
        for k, c in enumerate(sample["choices"]):
            option_str += f"({chr(65+k)}) {c} " 
        prompts.append(F"You are a helpful assistant. USER: {sample['question']} {option_str} ASSISTANT:")

    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new_tokens)
    
    if len(args.lora_path) > 0:
        llm = LLM(model=args.model_path, max_model_len=args.max_model_length, enable_lora=True, max_lora_rank=32, max_num_seqs=64)
        outputs = llm.generate(prompts, sampling_params, lora_request=LoRARequest("lora_adapter", 1, args.lora_path))
    else:
        llm = LLM(model=args.model_path, max_model_len=args.max_model_length)
        outputs = llm.generate(prompts, sampling_params)

    correct = 0
    with open(args.prediction_file, "w") as f:
        for output, sample in zip(outputs, dataset):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            
            index2ans = {}
            options = []
            for k, c in enumerate(sample["choices"]):
                index = chr(65+k)
                index2ans[index] = c
                options.append(index)

            gold = chr(65+sample["answer"])
            pred = parse_multi_choice_response(generated_text, options, index2ans)

            if gold == pred:
                correct += 1 

            sample = {
                "prompt": prompt,
                "response": generated_text,
                "answer": gold,
                "prediction": pred
            }
            json.dump(sample, f)
            f.write("\n")

    print(correct / len(dataset))
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--prediction_file", type=str)
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--max_model_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1)
    args = parser.parse_args()

    generate_response(args)