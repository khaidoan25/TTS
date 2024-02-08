#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-8
#SBATCH --output=qasr_xtts_debug.log


hf_dir="UBC-NLP/QASR"
lhotse_dir="/tmp/QASR/lhotse_dir"
manifests_dir="/home/chiyu.zhang/tts/work_dir/aratts/TTS/recipes/qasr/xtts_v2/data"
hf_home="/l/users/chiyu.zhang/hf_cache"
log_dir="/l/users/chiyu.zhang/tts/checkpoints/qasr"
debug=true
multi_gpu=false

export HF_HOME=$hf_home

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ ! -f $manifests_dir/qasr_cuts.jsonl.gz ]; then
  log "Prepare QASR manifests!"
  if [ $debug == true ]; then
    python convert_hf_lhotse.py --hf_dir $hf_dir --lhotse_dir $lhotse_dir --output_dir $manifests_dir --debugging
  else
    python convert_hf_lhotse.py --hf_dir $hf_dir --lhotse_dir $lhotse_dir --output_dir $manifests_dir
  fi
else
  log "QASR manifests already existed!"
fi

if [ $multi_gpu == true ]; then
  CUDA_VISIBLE_DEVICES="0,1,2,3" python -m trainer.distribute --gpus "0,1,2,3" --script train_gpt_xtts.py
else
  CUDA_VISIBLE_DEVICES="0" python train_gpt_xtts.py
fi