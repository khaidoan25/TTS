#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --qos=gpu-8
#SBATCH --output=full_sample_5_wer_0_22k.log

hf_home="/l/users/chiyu.zhang/hf_cache"
lhotse_dir="/tmp/QASR/lhotse_dir"
hf_dir="UBC-NLP/QASR"

export HF_HOME=$hf_home

python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')"

python convert_hf_lhotse.py --hf_dir $hf_dir --lhotse_dir $lhotse_dir

export run_name="XTTSv2.0_full_sample_5_wer_0_22k"
export out_path="/l/users/chiyu.zhang/tts/checkpoints/qasr/full_qasr"
export data_path="/home/chiyu.zhang/tts/work_dir/aratts/TTS/recipes/qasr/did/sub_data_5_sample_0_wer/full"
export epochs="10"
export multi_gpu="1"
export batch_size="6"
export grad_acumm_steps="38"
export total_step="1671401"
export sampling_rate="22050"

ckpt_path=$(python find_ckpt.py --ckpt_dir /l/users/chiyu.zhang/tts/checkpoints/qasr/full_qasr/XTTSv2.0_full_sample_5_wer_0_22k-May-09-2024_09+57AM-2ae7458 2>&1)

ulimit -n 2048

CUDA_VISIBLE_DEVICES="0,1,2,3" python -m trainer.distribute \
    --gpus "0,1,2,3" --script train_gpt_xtts.py \
    --restore_path $ckpt_path
    
