import torch
from TTS.utils.synthesizer import Synthesizer

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "/l/users/chiyu.zhang/tts/checkpoints/qasr/YourTTS-AR-QASR-February-10-2024_07+20AM-7a06436/best_model.pth"
CONFIG_PATH = "/l/users/chiyu.zhang/tts/checkpoints/qasr/YourTTS-AR-QASR-February-10-2024_07+20AM-7a06436/config.json"

MODEL_DIR = "/l/users/chiyu.zhang/tts/checkpoints/qasr/YourTTS-AR-QASR-February-10-2024_07+20AM-7a06436"

s = Synthesizer(
    tts_checkpoint=MODEL_PATH,
    tts_config_path=CONFIG_PATH,
    use_cuda=True
)

wav = s.tts(
    text="الذي يتجه جنوباً صوب العاصمة دمشق ،",
    speaker_wav="/l/users/chiyu.zhang/tts/dataset/lhotse_dir/972C30A4_A941_4CEA_8844_45EFE75FBEDF_speaker3_align/972C30A4_A941_4CEA_8844_45EFE75FBEDF_utt_20_align_clean.wav"
)

s.save_wav(wav, "output/test.wav")