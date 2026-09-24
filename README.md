# whisper-transcribe

A small command-line tool that turns an audio file, or a folder of audio files, into
timestamped transcripts using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

- Writes an `.srt` subtitle file and a plain `.txt` transcript with `[HH:MM:SS]` timestamps for each input.
- **Runs offline by default.** It only uses models that are already on your machine and never downloads anything unless you pass `--allow-download`.
- Uses English by default. Pass `--language` to change it.
- Reports the duration, segment count, and any error for each file, and warns when a transcript looks broken.

## Setup

You need Python 3.9 or newer. `ffmpeg` is not required, because faster-whisper decodes audio through PyAV.

```bash
git clone <this-repo> whisper-transcribe
cd whisper-transcribe
python3 -m venv .venv
.venv/bin/pip install -e .
```

This installs `faster-whisper` and its dependencies (CTranslate2, PyAV, onnxruntime), which take up about 400 MB once installed. If you already have an environment with `faster-whisper`, you can skip the install and run the tool from the repo folder:

```bash
/path/to/env/bin/python -m whisper_transcribe <input>
```

### Models must be available locally for offline mode

Offline mode, the default, loads a model from one of these two places:

1. **The Hugging Face cache** (`~/.cache/huggingface/hub`, or `$HF_HOME/hub`). `--model base` resolves to `models--Systran--faster-whisper-base` there. Check what you have with:
   ```bash
   ls ~/.cache/huggingface/hub | grep faster-whisper
   ```
2. **A local folder** that contains a CTranslate2 Whisper model (`model.bin`, `config.json`, `tokenizer.json`, `vocabulary.*`):
   ```bash
   whisper-transcribe talk.mp3 --model /models/faster-whisper-small
   ```

If the model is not found, the tool stops with an error and does not download it. To fetch a model once, run with `--allow-download`. Later runs then work offline.

| model | download | notes |
|---|---|---|
| `tiny` | ~75 MB | fastest, lowest accuracy |
| `base` | ~145 MB | default; good for clear English talks |
| `small` | ~480 MB | noticeably better with names and jargon |
| `medium` | ~1.5 GB | slower on CPU |
| `large-v3` | ~3 GB | best accuracy; a GPU is recommended |

## Usage

```bash
# One file -> ./transcripts/talk.srt and ./transcripts/talk.txt
whisper-transcribe talk.mp3

# Every audio file in a folder (not recursive), written to a chosen folder
whisper-transcribe ~/Downloads/podcasts -o ~/Downloads/podcasts/transcripts

# Another language, or auto-detect
whisper-transcribe wawancara.m4a --language id
whisper-transcribe clip.mp3 --language auto

# Bigger model, GPU, and resume a folder run without redoing finished files
whisper-transcribe lectures/ --model small --device cuda --compute-type float16 --skip-existing
```

Without installing, replace `whisper-transcribe` with `python -m whisper_transcribe`.

| option | default | meaning |
|---|---|---|
| `-o, --output` | `transcripts` | output folder, created if missing |
| `-l, --language` | `en` | language code, or `auto` to detect it |
| `-m, --model` | `base` | model size or local model folder |
| `--device` | `cpu` | `cpu`, `cuda`, or `auto` |
| `--compute-type` | `int8` | CTranslate2 compute type |
| `--threads` | all cores | CPU threads |
| `--beam-size` | `5` | beam search width |
| `--no-vad` | off | turn off voice-activity filtering (on by default) |
| `--skip-existing` | off | skip inputs that already have both `.srt` and `.txt` |
| `--allow-download` | off | allow downloading a model that is not cached |

A folder input picks up `.mp3 .wav .m4a .flac .ogg .opus .aac .wma .webm .mp4 .mkv .mov` files. Other files, such as existing `.vtt` subtitles, are ignored.

### Output

`talk.txt`:
```
[00:00:02] Hello everyone, and welcome to today's session on build systems.
[00:00:07] Let's get started.
```

`talk.srt`:
```
1
00:00:02,220 --> 00:00:06,900
Hello everyone, and welcome to today's session on build systems.
```

At the end of a run the tool prints a report:

```
Summary
status   duration segments  file
ok          31:27      161  talk.mp3
WARN         4:10       12  clip.mp3
        warning: text does not look like 'en' (common-word ratio 4%, expected >= 20%); wrong language or garbled output?
ERROR           -        0  broken.mp3
        error: InvalidDataError: Invalid data found when processing input: 'broken.mp3'

3 file(s): 1 ok, 1 with warnings, 1 failed
```

A file that fails does not stop the run. The exit code is `1` if any file failed and `0` otherwise. Warnings do not change the exit code.

## Quality checks

Whisper sometimes fails without raising an error. For example, it can auto-detect English speech as Welsh and produce pages of made-up words. After each file, the tool runs cheap heuristics in `whisper_transcribe/checks.py` and warns when:

- the language was auto-detected with a probability below 0.80;
- the text does not look like the expected language, meaning that fewer than 20% of the words are common function words (checked for `en` and `id`; normal speech scores 35–55%);
- more than 2% of the letters are non-Latin in a language that uses Latin script;
- a single line makes up more than 20% of all segments, which suggests a hallucination loop;
- there are fewer than 20 words per minute of audio, or no text at all;
- the model's mean `avg_logprob` is below -1.0.

A warning means you should look at the transcript. It does not prove the transcript is wrong. If the language is the problem, re-run with an explicit `--language`.

## Development

```bash
python -m unittest -v
```

The tests cover the `.srt` and `.txt` formatting and the quality checks. They do not need a model or any audio.
