# 🐸💬 TTS Training Recipes

TTS recipes intended to host scripts running all the necessary steps to train a TTS model on a particular dataset.

For each dataset, you need to download the dataset once. Then you run the training for the model you want.

Run each script from the root TTS folder as follows.

```console
$ sh ./recipes/<dataset>/download_<dataset>.sh
$ python recipes/<dataset>/<model_name>/train.py
```

For some datasets you might need to resample the audio files. For example, VCTK dataset can be resampled to 22050Hz as follows.

```console
python TTS/bin/resample.py --input_dir recipes/vctk/VCTK/wav48_silence_trimmed --output_sr 22050 --output_dir recipes/vctk/VCTK/wav48_silence_trimmed --n_jobs 8 --file_ext flac
```

If you train a new model using TTS, feel free to share your training to expand the list of recipes.

You can also open a new discussion and share your progress with the 🐸 community.

## Train AraTTS on QASR

- Run `convert_hf_lhotse.py` to download QASR dataset from huggingface
and convert it to lhotse manifests.

```console
python convert_hf_lhotse.py \
    --hf_dir UBC-NLP/QASR \
    --lhotse_dir <lhotse_directory> \ # find a free disk space to store extracted wav files
    --output_dir ./data # directory to store lhotse manifests
```

- Run `train_yourtts.py` is the training script.

- Modify variables in `qasr_train.sh` for training. Also the `OUT_PATH` and `DATA_PATH` variables in `train_yourtts.py` script (line 40).

- *Note*: need at least 128G mem ram to store the speaker embedding matrix.