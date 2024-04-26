import argparse
from lhotse import CutSet
import pandas as pd

from transformers import pipeline
from tqdm import tqdm


def main(args):
    cuts = CutSet.from_file(args.manifest_file)

    did = pipeline(model=args.model_name, device="cuda")

    batch_count = 0
    batch = []
    dialects = []
    scores = []
    filenames = []
    for cut in tqdm(cuts):
        filenames.append("/".join(cuts[0].recording.sources[0].source.split("/")[-2:]))
        batch.append(cut.supervisions[0].text)
        batch_count += 1
        if batch_count % args.batch_size == 0:
            results = did(batch)
            batch = []
            for result in results:
                dialects.append(result["label"])
                scores.append(result["score"])
    if len(batch) != 0:
        results = did(batch)
        batch = []
        for result in results:
            dialects.append(result["label"])
            scores.append(result["score"])
    output_name = args.model_name.replace("/", "_")
    df = pd.DataFrame(
        {
            "filename": filenames,
            "dialect": dialects,
            "score": scores
        }
    )
    df.to_csv(f"{output_name}.csv", sep=",", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_file")
    parser.add_argument("--model_name")
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()
    main(args)