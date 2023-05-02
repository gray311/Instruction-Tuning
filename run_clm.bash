base_model="/root/autodl-tmp/models/llama-7b-hf"
data_path="/root/autodl-tmp/alpaca-lora/alpaca_data_gpt4.json"
output_dir="./lora-alpaca"

batch_size=4
num_epochs=3
learning_rate=3e-4
cutoff_len=256
val_set_size=2000

lora_rank=8
lora_alpha=16
lora_dropout=0.05

train_on_inputs=True
add_eos_token=False
group_by_length=False
prompt_template_name="alpaca"

accelerate launch \
    ./run_clm.py \
    --model_name_or_path $base_model \
    --train_file $data_path \
    --output_dir $output_dir \
    --per_device_train_batch_size $batch_size \
    --max_train_samples 1000 \
    --warmup_ratio 0.01 \
    --logging_steps 10 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs $num_epochs \
    --learning_rate $learning_rate \
    --block_size $cutoff_len \
    --val_set_size $val_set_size \
    --lora_rank $lora_rank \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --train_on_inputs $train_on_inputs \
    --add_eos_token $add_eos_token \
    --prompt_template_name $prompt_template_name \
    --report_to "wandb"