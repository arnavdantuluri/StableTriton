source /p/scratch/ccstdl//dantuluri1/venv/bin/activate
ml --force purge
ml use $OTHERSTAGES
ml Stages/2023
ml GCC
ml OpenMPI
ml CUDA
ml cuDNN
ml NCCL
ml git
ml PyTorch
ml torchvision
module unload PyTorch 
module unload torchvision
srun --pty --nodes=1 -A cstdl --partition develbooster --gres gpu:4 /bin/bash
export HF_HOME="/p/scratch/ccstdl//dantuluri1/transformers_cache"







#Commands to get eval working with lm-eval harness by eleuther
# for personal enviornment


# for Kshitij enviornment
conda activate /p/project/ccstdl/gupta6/miniconda3/envs/gptneox_flas
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
#SBATCH --account=cstdl
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition develbooster

# Unloading torch and torchvision since they load old version that are not compatible with torch 2.0
huggingface-cli login # if necessary

#download models and such with download model.py
python -u download_model.py

cd lm-evaluation-harness
#now download datasets and such from login node
python main.py --model hf-causal --model_args pretrained=EleutherAI/pythia-160m --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa --device cpu

#now connect to compute node
srun --pty --nodes=1 -A cstdl --partition booster --gres gpu --time=00:45:00 /bin/bash

#Set offline mode for datasets and transformers
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

#and now we can run eval main.py with gpus
python main.py --model hf-causal --model_args pretrained=OpenAssistant/oasst-sft-3-llama-13b-epoch-4 --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa --device cuda

#If multiple gpus available use this instead
python main.py --model hf-causal-experimental --model_args pretrained=OpenAssistant/pythia-12b-pre-v8-12.5k-steps --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa --device cuda

# HF Token
TOKEN=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
# Single line version QLoRa
python -u qlora_bpt_attn.py --model_name_or_path EleutherAI/pythia-160m --output_dir ./output --logging_steps 10 --save_strategy steps --data_seed 42 --save_steps 500 --save_total_limit 40 --evaluation_strategy steps --eval_dataset_size 1024 --max_eval_samples 1000 --per_device_eval_batch_size 1 --max_new_tokens 32 --dataloader_num_workers 1 --group_by_length --logging_strategy steps --remove_unused_columns False --do_train --do_eval --lora_r 64 --lora_alpha 16 --lora_modules all --double_quant --quant_type nf4 --bf16 --bits 4 --warmup_ratio 0.0 --lr_scheduler_type constant --gradient_checkpointing --dataset oasst1 --source_max_len 16 --target_max_len 512 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --max_steps 187 --eval_steps 18 --learning_rate 0.000 --adam_beta2 0.999 --max_grad_norm 0.3 --lora_dropout 0.1 --weight_decay 0.0 --seed 0

# Multiple line version QLoRa
python qlora.py \
    --model_name_or_path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5\
    --output_dir ./output/guanaco-65b \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 200 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset oasst1 \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1875 \
    --eval_steps 187 \
    --learning_rate 0.0001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --weight_decay 0.0 \
    --seed 0

python -u eval/perplexity.py -m daryl149/llama-2-7b-chat-hf --load-in-4bit --output-file="./eval/outputs/ntk_llama2.csv" --ntk=2.0
python -u eval/perplexity.py -m daryl149/llama-2-7b-chat-hf --load-in-4bit --output-file="./eval/outputs/dynamic-ntk_llama2.csv" --dynamic-ntk=2.0

TORCH_CUDA_ARCH_LIST="8.0" CUDA_HOME='/p/software/juwelsbooster/stages/2023/software/CUDA/11.7' pip install git+https://github.com/Dao-AILab/flash-attention.git

deepspeed --master_port 12802 --launcher slurm --hostfile '/p/home/jusers/dantuluri1/juwels/hostfiles/hostfile.txt' --no_ssh_check /p/project/ccstdl/dantuluri1/scaled-rope/finetune.py --output_dir saved_ckpts_32k --configs lora-7b-llama2 --deepspeed