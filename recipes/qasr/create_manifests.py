from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from tqdm import tqdm

from lhotse import (
    CutSet,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    fix_manifests,
    validate_recordings_and_supervisions,
)
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.utils import Pathlike

import pandas as pd
import os

    
def prepare_qasr(
    corpus_dir: Pathlike,
    dataset_parts: Sequence[str] = ["all"],
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    
    manifests = {}
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts, output_dir=output_dir, prefix="qasr"
        )
        
    spk2att = {}
    if (corpus_dir / "speakers.tsv").is_file():
        speaker_df = pd.read_csv(str(corpus_dir / "speakers.tsv"), sep="\t", index_col=None)
        for _, row in speaker_df.iterrows():
            spk2att[row["speaker_id"]] = {
                "name": row["name"],
                "normalized_name": row["normalized_name"],
                "gender": row["gender"],
                "unique": row["unique"]
            }
    for part in tqdm(dataset_parts, desc="Preparing QASR parts"):
        path = corpus_dir / part
        recordings = RecordingSet.from_dir(
            path,
            "*clean.wav",
            num_jobs=num_jobs,
            recording_id=lambda r: str(r.name).replace("_clean.wav", "")
        )
        recordings.to_file("./data/recordings.jsonl.gz")
        supervisions = []
        for trans_path in tqdm(
            path.rglob("*.txt"),
            desc="Scanning transcripts files (progbar for all speakers)",
            leave=False
        ):
            rec_id = trans_path.stem
            transcript = open(trans_path, "r", encoding="utf8").read()
            spk_id = trans_path.parts[-2]
            supervisions.append(
                SupervisionSegment(
                    id=rec_id,
                    recording_id=rec_id,
                    start=0.0,
                    duration=recordings[rec_id].duration,
                    channel=0,
                    text=transcript,
                    language="Arabic",
                    speaker=spk_id,
                    gender=spk2att.get(spk_id).get("gender"),
                    custom=spk2att.get(spk_id),
                )
            )
            
        supervisions = SupervisionSet.from_segments(supervisions)
        recordings, supervisions = fix_manifests(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)
        if output_dir is not None:
            supervisions.to_file(output_dir / f"qasr_supervisions_{part}.jsonl.gz")
            recordings.to_file(output_dir / f"qasr_recordings_{part}.jsonl.gz")
        manifests[part] = {"recordings": recordings, "supervisions": supervisions}
    return manifests

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    manifests = prepare_qasr(
        args.data_dir,
    )

    cuts = CutSet.from_manifests(
        recordings=manifests["all"]["recordings"],
        supervisions=manifests["all"]["supervisions"]
    )

    cuts.to_file(Path(args.output_dir) / "cuts_all.jsonl.gz")