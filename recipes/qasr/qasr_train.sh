#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --qos=gpu-8
#SBATCH --output=qasr_training.log


hf_dir="UBC-NLP/QASR"
lhotse_dir="/tmp/QASR/lhotse_dir"
manifests_dir="/home/chiyu.zhang/tts/work_dir/aratts/TTS/recipes/qasr/data"
hf_home="/l/users/chiyu.zhang/hf_cache"
log_dir="/tmp/QASR/logs"

export HF_HOME=$hf_home

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ ! -f $output_dir/qasr_cuts.jsonl.gz ]; then
  log "Prepare QASR manifestes!"
  python convert_hf_lhotse.py --hf_dir $hf_dir --lhotse_dir $lhotse_dir --output_dir $manifests_dir --debugging
else
  log "QASR manifests already existed!"
fi

CUDA_VISIBLE_DEVICES="0,1" python -m trainer.distribute --gpus "0,1" --script train_yourtts.py