from lhotse import RecordingSet, SupervisionSet
from tqdm import tqdm

recs = RecordingSet.from_file("./data/qasr_recordings_all.jsonl.gz")

rec_list = []
for rec in tqdm(recs):
    rec_dict = rec.to_dict()
    source = rec_dict["sources"][0]
    rec_dict["sources"] = [
        {
            "type": source["type"],
            "channels": source["channels"],
            "source": f"/tmp/QASR/{source['source'].replace('.wav', '_clean.wav')}"
        }
    ]
    rec_list.append(rec_dict)
new_recs = RecordingSet.from_dicts(rec_list)
new_recs.to_file("qasr_recordings_all_new.jsonl.gz")