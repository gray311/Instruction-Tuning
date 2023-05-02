import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse
import json, os
from typing import List

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel,
)

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--tokenizer_path',default=None,type=str)
    parser.add_argument('--data_file',default=None, type=str,help="file that contains instructions (one instruction per line).")
    parser.add_argument('--with_prompt',action='store_true')
    parser.add_argument('--interactive',action='store_true')
    parser.add_argument('--predictions_file', default='./predictions.json', type=str)
    args = parser.parse_args()
    return args

generation_config = dict(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.3,
    max_new_tokens=400
)


 # The prompt template below is taken from llama.cpp
 # and is slightly different from the one used in training.
 # But we find it gives better results
prompt_input = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
)

sample_data = ["How can I comfort others, can you give me some advice?"]

def generate_prompt(instruction, input=None):
    if input:
        instruction = instruction + '\n' + input
    return prompt_input.format_map({'instruction': instruction})


def main(
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: List[str] = ["q_proj", "v_proj",],
    ):
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    accelerator = Accelerator()

    args = parse_args()

    print(args.tokenizer_path)
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    device_map = "auto"
    base_model = LlamaForCausalLM.from_pretrained(
        args.tokenizer_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size != tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenzier_vocab_size)

    #base_model = prepare_model_for_int8_training(base_model)

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, config)

    print(model)
    state_dict = torch.load(os.path.join(args.base_model, "pytorch_model.bin"))

    
    for name, param in model.named_parameters():
        if "lora" in name:
            model.state_dict()[name].copy_(state_dict[name])

    if device == torch.device('cpu'):
        model.float()
        
    # test data
    if args.data_file is None:
        examples = sample_data
    else:
        with open(args.data_file, 'r') as f:
            examples = [l.strip() for l in f.readlines()]
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)

    model.to(device)
    model.eval()

    with torch.no_grad():
        if args.interactive:
            print("Start inference with interactive mode.")
            while True:
                raw_input_text = input("Input:")
                if len(raw_input_text.strip()) == 0:
                    break
                if args.with_prompt:
                    input_text = generate_prompt(instruction=raw_input_text)
                else:
                    input_text = raw_input_text
                inputs = tokenizer(input_text, return_tensors="pt")  # add_special_tokens=False ?
                generation_output = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs['attention_mask'].to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_config
                )
                s = generation_output[0]
                output = tokenizer.decode(s, skip_special_tokens=True)
                if args.with_prompt:
                    response = output.split("### Response:")[1].strip()
                else:
                    response = output
                print("Response: ", response)
                print("\n")
        else:
            print("Start inference.")
            results = []
            for index, example in enumerate(examples):
                if args.with_prompt is True:
                    input_text = generate_prompt(instruction=example)
                else:
                    input_text = example
                with torch.autocast("cuda"):
                    print("model input:")
                    print("-" * 50)
                    print(input_text)
                    inputs = tokenizer(input_text, return_tensors="pt")  # add_special_tokens=False ?

                    generation_output = model.generate(
                        input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs['attention_mask'].to(device),
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        **generation_config
                    )
                    s = generation_output[0]
                    output = tokenizer.decode(s, skip_special_tokens=True)

                    print("model output:")
                    print("-"*50)
                    print(output)
                    if args.with_prompt:
                        response = output.split("### Response:")[1].strip()
                    else:
                        response = output

                    print(f"======={index}=======")
                    print(f"Input: {example}\n")
                    print(f"Output: {response}\n")

                    results.append({"Input": input_text, "Output": response})

            dirname = os.path.dirname(args.predictions_file)
            os.makedirs(dirname, exist_ok=True)
            with open(args.predictions_file, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            with open(dirname + '/generation_config.json', 'w') as f:
                json.dump(generation_config, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
