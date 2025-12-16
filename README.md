Our project is built upon a larger code base called LLaMA-Factory
## How to install the dependency
Here are the steps to run the project:
```shell
git clone --depth 1 https://github.com/EnsueCollectR/LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```  
Framework's readme in https://github.com/EnsueCollectR/LLaMA-Factory/blob/main/Framework_README.md

## Our Project Code
### Dataset Processing
The [Code](https://github.com/EnsueCollectR/LLaMA-Factory/blob/main/scripts/prepare_pubmedqa.py) is converting the dataset into a format that could be used inside LLaMA-Factory for training.
For creating a training set
```shell
cd /LLaMA-Factory
python scripts/prepare_pubmedqa.py --split train --output data/pubmedqa_alpaca.jsonl
```
For creating a test set
```shell
cd /LLaMA-Factory
python scripts/prepare_pubmedqa.py --split test --output data/pubmedqa_alpaca_test.jsonl
```
### Added data
We added our converted dataset into [dataset folder](https://github.com/EnsueCollectR/LLaMA-Factory/tree/main/data)

## How to finetune
Run the code on a high performance GPU server
```shell
llamafactory-cli train \
  --stage sft \
  --do_train True \
  --model_name_or_path Qwen/Qwen3-8B \
  --preprocessing_num_workers 16 \
  --finetuning_type lora \
  --template qwen3 \
  --flash_attn none \
  --dataset_dir data \
  --dataset identity,pubmedqa_alpaca \
  --cutoff_len 2048 \
  --learning_rate 2e-4 \
  --num_train_epochs 3.0 \
  --max_samples 100000 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --max_grad_norm 0.5 \
  --logging_steps 5 \
  --save_strategy steps --save_steps 1000 \
  --packing False \
  --enable_thinking False \
  --run_name qwen3-8b-lora-nothink-v2 \
  --output_dir saves/Qwen3-8B-Thinking/lora/qwen3-2 \
  --fp16 True \
  --plot_loss True \
  --trust_remote_code True \
  --ddp_timeout 180000000 \
  --include_num_input_tokens_seen True \
  --optim adamw_torch \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target all \
  --val_size 0.1 \
  --eval_strategy steps \
  --eval_steps 50 \
  --per_device_eval_batch_size 4 \
  --gradient_checkpointing True
```
