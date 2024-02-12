import os
import argparse
import json
import random

from lhotse import CutSet

import torch
from TTS.utils.synthesizer import Synthesizer


def get_eval_samples(manifest_dir):
    with open(f"{manifest_dir}/speaker_id_split.json", "r") as f:
        unseen_speaker_ids = json.load(f)["test"]["unseen"]
        
    cuts = CutSet.from_file(f"{manifest_dir}/qasr_cuts.jsonl.gz")
    
    speaker_sample_dict = {}
    items = []
    for sample in cuts:
        if sample.supervisions[0].speaker not in unseen_speaker_ids:
            continue
        text = sample.supervisions[0].text
        audio_file = sample.recording.sources[0].source
        speaker_name = sample.supervisions[0].speaker
        audio_name = sample.supervisions[0].recording_id + ".wav"
        items.append(
            {
                "text": text,
                "audio_file": audio_file,
                "speaker_name": speaker_name,
                "audio_name": audio_name
            }
        )
        if speaker_name not in speaker_sample_dict:
            speaker_sample_dict[speaker_name] = [audio_file]
        else:
            speaker_sample_dict[speaker_name].append(audio_file)
            
    return items, speaker_sample_dict

def main(args):
    items, speaker_sample_dict = get_eval_samples(args.manifest_dir)
    s = Synthesizer(
        tts_checkpoint=os.path.join(args.ckpt_dir, "best_model.pth"),
        tts_config_path=os.path.join(args.ckpt_dir, "config.json"),
        use_cuda=True
    )
    
    for item in items:
        print(item)
        speaker_dir = os.path.join(args.wav_dir, item["speaker_name"])
        if not os.path.exists(speaker_dir):
            os.makedirs(speaker_dir)

        samples = speaker_sample_dict[item["speaker_name"]]
        samples = [sample for sample in samples if sample != item["audio_file"]]
        ref_wav = random.choice(samples)
        output_wav = s.tts(
            text=item["text"],
            speaker_wav=ref_wav
        )
        s.save_wav(output_wav, os.path.join(speaker_dir, item["audio_name"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir")
    parser.add_argument("--manifest_dir")
    parser.add_argument("--wav_dir")
    args = parser.parse_args()
    
    main(args)