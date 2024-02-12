from datasets import load_dataset
from lhotse import (
    Recording, SupervisionSegment,
    RecordingSet, SupervisionSet,
    fix_manifests, validate_recordings_and_supervisions,
    CutSet
)
from scipy.io.wavfile import write
import os
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from pathlib import Path

def get_rec_sup(sample):
    if sample["is_copy"]:
        write(
            sample["wav_path"],
            sample['audio']['sampling_rate'],
            sample['audio']['array'].astype(np.float32))
    if sample["add_manifest"]:
        recording = Recording.from_file(sample["wav_path"])
        supervision = SupervisionSegment(
            id=recording.id,
            recording_id=recording.id,
            start=0.0,
            duration=recording.duration,
            text=sample['transcription'],
            channel=0,
            speaker=sample['speaker_id'],
            language='Arabic',
            gender=sample["gender"],
            custom=sample["custom"]
        )
    else:
        recording = None
        supervision = None
    
    return {"recording": recording, "supervision": supervision}

def main(args):
    speaker_df = pd.read_csv(f"{args.output_dir}/speakers.tsv", sep="\t", index_col=None)
    spk2att = {}
    for _, row in speaker_df.iterrows():
        spk2att[row["speaker_id"]] = {
            "name": row["name"],
            "normalized_name": row["normalized_name"],
            "gender": row["gender"],
            "unique": row["unique"]
        }
    # Get existed files
    existed_files = []
    for file in Path(args.lhotse_dir).rglob("*.wav"):
        existed_files.append(str(file))
        
    # Get existed manifests
    existed_manifests = []
    if os.path.exists(f"{args.output_dir}/qasr_recordings.jsonl.gz"):
        old_recs = RecordingSet.from_file(f"{args.output_dir}/qasr_recordings.jsonl.gz")
        old_sups = SupervisionSet.from_file(f"{args.output_dir}/qasr_supervisions.jsonl.gz")
        for rec in old_recs:
            existed_manifests.append(rec.sources[0].source)
        recording_list = list(old_recs)
        supervision_list = list(old_sups)
    else:
        recording_list = []
        supervision_list = []

    dataset = load_dataset(args.hf_dir)["train"]
    if args.debugging:
        num_tasks = int(len(dataset)/50)
        print(f"Run with a portion of data, {num_tasks} samples")
    else:
        num_tasks = len(dataset)
    tasks = []
    i = 0
    pool = mp.Pool(os.cpu_count())
    for sample in tqdm(dataset, desc="Generating lhotse samples"):
        speaker_dir = f"{args.lhotse_dir}/{sample['speaker_id']}"
        if not os.path.exists(speaker_dir):
            os.makedirs(speaker_dir)
        wav_path = f"{speaker_dir}/{sample['audio']['path']}"
        if wav_path in existed_files:
            sample['is_copy'] = False
        else:
            sample['is_copy'] = True
        if wav_path in existed_manifests:
            sample['add_manifest'] = False
        else:
            sample['add_manifest'] = True
        sample['wav_path'] = wav_path
        sample['gender'] = spk2att.get(sample['speaker_id']).get("gender")
        sample['custom'] = spk2att.get(sample['speaker_id'])
        tasks.append(sample)
        i += 1
        
        if i % int(num_tasks/50) == 0:
            for ret in tqdm(
                pool.imap(get_rec_sup, tasks),
                total=len(tasks),
                desc="Generating lhotse manifests"
            ):
                if ret["recording"] is not None:
                    recording_list.append(ret["recording"])
                    supervision_list.append(ret["supervision"])
                    
            tasks = []
            supervisions = SupervisionSet.from_segments(supervision_list)
            recordings = RecordingSet.from_recordings(recording_list)
            recordings, supervisions = fix_manifests(recordings, supervisions)
            validate_recordings_and_supervisions(recordings, supervisions)
            supervisions.to_file(f"{args.output_dir}/qasr_supervisions.jsonl.gz")
            recordings.to_file(f"{args.output_dir}/qasr_recordings.jsonl.gz")
                
        if i % num_tasks == 0:
            break
            
    # Check last batch
    if len(tasks) != 0:
        for ret in tqdm(
            pool.imap(get_rec_sup, tasks),
            total=len(tasks),
            desc="Generating lhotse manifests"
        ):
            if ret["recording"] is not None:
                recording_list.append(ret["recording"])
                supervision_list.append(ret["supervision"])
                
        supervisions = SupervisionSet.from_segments(supervision_list)
        recordings = RecordingSet.from_recordings(recording_list)
        recordings, supervisions = fix_manifests(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)
        supervisions.to_file(f"{args.output_dir}/qasr_supervisions.jsonl.gz")
        recordings.to_file(f"{args.output_dir}/qasr_recordings.jsonl.gz")
    
    pool.close()
    pool.join()
    cuts = CutSet.from_manifests(
        recordings=recordings,
        supervisions=supervisions
    )
    cuts.to_file(f"{args.output_dir}/qasr_cuts.jsonl.gz")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dir")
    parser.add_argument("--lhotse_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--debugging", action="store_true")
    
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.lhotse_dir):
        os.makedirs(args.lhotse_dir)
    main(args)