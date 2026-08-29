# Data Directory

This project does **not** ship the audio dataset. You must obtain your
own copy and point the code at it via configuration — see
`config/config.py` and `.env.example`.

## Where to place the dataset

By default, the code looks for the dataset at `data/raw/` (this
directory), but you can point it anywhere on disk via the
`DATASET_ROOT` (or `STUTTERING_DATASET_DIR`) environment variable, for
example by copying `.env.example` to `.env` and editing it:

```text
DATASET_ROOT=/absolute/path/to/your/dataset
```

## Expected structure

The pipeline expects a metadata CSV named `audio_dataset.csv` directly
under the configured dataset root, with (at minimum) the following
columns:

| Column        | Description                                              |
|---------------|-----------------------------------------------------------|
| `file_path`   | Path to the raw audio file                                 |
| `file_name`   | File name (used to detect "fluent"/"dysfluent" in filename)|
| `label`       | Top-level source label: `"normal"` or `"stuttering"`       |
| `speaker_id`  | Speaker identifier (present for the normal-speech source)  |

```text
data/raw/                      <- DATASET_ROOT
├── audio_dataset.csv
├── normal/
│   └── speakers/
│       └── <speaker-id>/
│           └── *.wav
└── stuttering/
    └── clips/
        └── <speaker-code>_<...>_{fluent|dysfluent}_<index>.wav
```

Stuttering-set filenames are expected to carry a leading speaker code
of the form `M_####` or `F_####` (e.g. `M_0030_16y4m_1_dysfluent_000.wav`),
which `src/data/load_dataset.fix_speaker_id()` parses out via regex.
Files that don't follow this pattern keep their original
`speaker_id` value.

## What NOT to upload to GitHub

- Raw or processed audio files (`*.wav`, `*.mp3`, `*.flac`, ...) — these
  are excluded via `.gitignore`.
- Any personally identifying information linked to a speaker beyond the
  anonymized speaker code already used in the source dataset.
- Your local `.env` file (only `.env.example`, with placeholders, is
  committed).

## How processed data is generated

Running `scripts/prepare_dataset.py` (or `scripts/full_pipeline.py`)
populates `data/processed/` with:

- `processed_audio/` — cleaned, resampled (16 kHz mono), silence-trimmed,
  amplitude-normalized WAV files.
- `processed_dataset.csv`, `failed_files.csv` — preprocessing manifest
  and failure log.
- `final_dataset_fixed.csv` — final binary-labeled, speaker-ID-corrected
  metadata.
- `train.csv`, `validation.csv`, `test.csv` — speaker-independent splits.
- `features/` — extracted MFCC+delta+delta-delta feature tensors
  (`X_*.npy`, `y_*.npy`).

None of `data/processed/` is committed to the repository (see
`.gitignore`); it is fully regenerable from the raw dataset by running
the scripts above.

## Dataset provenance

See the top-level `README.md` ("Dataset" section) and
`docs/methodology.md` for what is known about the dataset's origin
from the supplied notebook. **The exact source/download URL and
license of the dataset were not present in the original notebook** and
are therefore not documented here — the researcher should add this
information once known.
