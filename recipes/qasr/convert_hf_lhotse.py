from datasets import load_dataset
from lhotse import (
    Recording, SupervisionSegment,
    RecordingSet, SupervisionSet,
    fix_manifests, validate_recordings_and_supervisions
)
from scipy.io.wavfile import write
import os
import pandas as pd
import argparse
from tqdm import tqdm

def main(args):
    speaker_df = pd.read_csv("data/speakers.tsv", sep="\t", index_col=None)
    spk2att = {}
    for _, row in speaker_df.iterrows():
        spk2att[row["speaker_id"]] = {
            "name": row["name"],
            "normalized_name": row["normalized_name"],
            "gender": row["gender"],
            "unique": row["unique"]
        }

    dataset = load_dataset(args.hf_dir)["train"]
    recording_list = []
    supervision_list = []
    for sample in tqdm(dataset, desc="Generating lhotse samples"):
        speaker_dir = f"{args.lhotse_dir}/{sample['speaker_id']}"
        if not os.path.exists(speaker_dir):
            os.makedirs(speaker_dir)
        wav_path = f"{speaker_dir}/{sample['audio']['path']}"
        write(
            wav_path,
            sample['audio']['sampling_rate'],
            sample['audio']['array'].astype(np.float32))
        recording = Recording.from_file(wav_path)
        supervision = SupervisionSegment(
            id=recording.id,
            recording_id=recording.id,
            start=0.0,
            duration=recording.duration,
            text=sample['transcription'],
            channel=0,
            speaker=sample['speaker_id'],
            language='Arabic',
            gender=spk2att.get(sample['speaker_id']).get("gender"),
            custom=spk2att.get(sample['speaker_id'])
        )
        recording_list.append(recording)
        supervision_list.append(supervision)
    supervisions = SupervisionSet.from_segments(supervision_list)
    recordings = RecordingSet.from_recordings(recording_list)
    recordings, supervisions = fix_manifests(recordings, supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)
    supervisions.to_file("test_data/qasr_supervisions.jsonl.gz")
    recordings.to_file("test_data/qasr_recordings.jsonl.gz")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dir")
    parser.add_argument("--lhotse_dir")
    
    args = parser.parse_args()
    main(args)