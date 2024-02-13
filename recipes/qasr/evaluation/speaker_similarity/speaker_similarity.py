import os
import argparse
import pathlib
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Resample
from models.ecapa_tdnn import ECAPA_TDNN_SMALL
from tqdm import tqdm

def verification(wav1, wav2):
    signal1, sr1 = torchaudio.load(wav1)
    signal2, sr2 = torchaudio.load(wav2)

    resample1 = Resample(orig_freq=sr1, new_freq=16000)
    resample2 = Resample(orig_freq=sr2, new_freq=16000)

    signal1 = resample1(signal1)
    signal2 = resample2(signal2)
    
    model.eval()
    with torch.no_grad():
        emb1 = model(signal1)
        emb2 = model(signal2)
    
    sim = F.cosine_similarity(emb1, emb2)
    
    return sim[0].item()
    
def main(args):
    
    global model
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
    state_dict = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict['model'], strict=False)
    
    generated_dir = pathlib.Path(args.generated_dir)
    
    generated_wavs = [file for file in generated_dir.rglob("*.wav")]
    
    speaker_similarity = 0
    for pred in tqdm(generated_wavs):
        if args.single:
            filename = str(pred).split("/")[-1]
        else:
            filename = os.path.relpath(str(pred), args.generated_dir)
        label = os.path.abspath(os.path.join(args.groundtruth_dir, filename))
        speaker_similarity += verification(label, pred)
    
    print("Avg speaker similarity: ", speaker_similarity/len(generated_wavs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_dir", type=str, required=True)
    parser.add_argument("--groundtruth_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--single", action="store_true")
    
    args = parser.parse_args()
    main(args)