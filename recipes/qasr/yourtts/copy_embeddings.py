import torch
import os
from tqdm import tqdm

speakers = torch.load("test_mp/speakers.pth")

output_dir = "/tmp/QASR/lhotse_dir"
for k, v in tqdm(speakers.items()):
    speaker_name = v["name"]
    embedding = v["embedding"]
    if not os.path.exists(os.path.join(output_dir, speaker_name)):
        os.makedirs(os.path.join(output_dir, speaker_name))
        
    torch.save(embedding, os.path.join(output_dir, speaker_name, f"{k}.pth"))