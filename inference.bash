base_model="/root/autodl-tmp/alpaca-lora/lora-alpaca/step_20"
tokenizer_path="/root/autodl-tmp/models/llama-7b-hf"

python inference.py \
    --base_model $base_model \
    --tokenizer_path $tokenizer_path \
    --with_prompt