# Security & Privacy Audit

This audit covers `model.ipynb` (source code **and** cell outputs) as
supplied. It documents every issue found and the fix applied in the
refactored repository.

## Summary

| # | Category | Instances | Status |
|---|----------|-----------|--------|
| 1 | Hard-coded Windows absolute paths (source code) | 15 cells | Fixed |
| 2 | Hard-coded Windows absolute paths (cell outputs) | 20 cells | Fixed |
| 3 | Local username exposure (`au84b`) | throughout | Fixed |
| 4 | Local Python installation paths (pip output) | 1 cell | Fixed |
| 5 | Dataset directory names revealed in code/outputs | throughout | Fixed |
| 6 | API keys / tokens / passwords / credentials | — | None found |
| 7 | Personal information (name, email) | — | None found |
| 8 | Speaker-identifying filenames in outputs | 2 cells | Documented (see below) |

No API keys, tokens, passwords, or cloud/database credentials were
found anywhere in the notebook (source or outputs).

## Detailed findings

### 1–3. Hard-coded local paths and username

**Issue.** Nearly every code cell defined paths such as:

```python
DATASET_DIR = Path(r"C:\Users\au84b\Downloads\stuttering project\dataset")
```

and printed outputs (e.g. "Saved to: C:\Users\au84b\Downloads\stuttering
project\dataset\final_dataset.csv") that reveal:
- the researcher's Windows username (`au84b`),
- the exact local folder structure of their machine,
- their local Python installation path
  (`c:\Users\au84b\AppData\Local\Programs\Python\Python312\...`), printed
  in full during a `pip install` cell's output.

**Risk.** Reveals the researcher's local machine/user information,
prevents the code from running on any other computer, and is
unnecessary and inappropriate for a public repository.

**Fix.** All hard-coded paths were replaced with the configurable
system in `config/config.py`, which resolves the dataset root from an
environment variable / `.env` file (see `.env.example`) with a
project-relative fallback (`data/raw/`). Model, results, and feature
directories are all derived from `PROJECT_ROOT` rather than an
absolute personal path. The `pip install` cell (and its verbose
"Requirement already satisfied" output revealing the local Python
installation path) was removed from the cleaned notebook and replaced
with a standard `requirements.txt`-based install instruction.

### 4. Local package installation output

**Issue.** One cell's output was the full `pip install librosa
soundfile pandas tqdm` log, which echoed dozens of lines containing
the local Windows installation path
(`c:\Users\au84b\AppData\Local\Programs\Python\Python312\Lib\site-packages\...`).

**Fix.** Removed from the cleaned notebook and from all generated
source files. `requirements.txt` is provided instead.

### 5. Dataset directory names

**Issue.** Source code and outputs revealed the exact folder layout of
the original dataset on the researcher's machine, e.g.:

```text
C:\Users\au84b\Downloads\stuttering project\dataset\normal\speakers\...
C:\Users\au84b\Downloads\stuttering project\dataset\sttutering\clips\...
```

(note the source data itself used a folder named `sttutering`, a typo
in the original collection, not introduced by this refactor).

**Risk.** Low on its own (folder names, not personal data), but
combined with the absolute path prefix it reinforces machine/user
fingerprinting and provides no benefit to a public repository.

**Fix.** `data/README.md` documents the *relative* expected structure
(`normal/speakers/...`, `stuttering/clips/...`) without any
machine-specific prefix. The actual dataset files/directories are
excluded from version control via `.gitignore`.

### 6. Credentials, tokens, secrets

A full-text search of the notebook (source and outputs) for
`password`, `api_key`, `apikey`, `secret`, `token`, `credential`,
`Authorization`, `Bearer`, and similar terms found **no matches**
beyond a metric named `"binary_label"` and normal variable names —
i.e., no actual secrets are present.

### 7. Personal information

No email addresses, phone numbers, or other personal identifiers
belonging to the researcher were found. The only personally-adjacent
string is the Windows username `au84b`, addressed in finding #1–3
above.

### 8. Speaker-identifying information in filenames

**Issue.** Several notebook outputs print individual audio filenames,
e.g. `M_0030_16y4m_1_dysfluent_000.wav`, and one normal-speech example
filename that includes a long alphanumeric speaker hash
(`2BqVo8kVB2Skwgyb`) inherited from the original dataset's own naming
convention.

**Assessment.** These filenames are the dataset's own anonymized
speaker/recording codes (not names, emails, or other directly
identifying information), and are scientifically relevant since the
project's speaker-independent evaluation protocol depends on exactly
this kind of speaker-code metadata. They are not real names.

**Fix.** No metadata CSVs containing these codes are committed to the
repository (see `.gitignore`; `data/processed/` is excluded), and the
cleaned notebook (`notebooks/model.ipynb`) does not print
individual raw filenames in its example/demonstration outputs. The
underlying speaker-code convention is documented in
`data/README.md` and `docs/methodology.md` for reproducibility, since
this is required to understand and reproduce the speaker-independent
split.

## Dataset redistribution

The complete audio dataset (13,534 recordings) is **not** included in
this repository. `data/` contains only a `README.md` describing the
expected structure; raw and processed audio, feature `.npy` files, and
trained model checkpoints are all excluded via `.gitignore`. This is a
privacy/licensing precaution: the original notebook does not state a
license for the dataset, so redistribution rights cannot be assumed
(see `README.md` → "Dataset" and `docs/methodology.md` for what is and
isn't known about dataset provenance).

## Final verification

A final scan of every generated file in this repository for the
patterns `C:\Users\`, `/Users/`, `au84b`, `password`, `api_key`,
`token`, `secret`, `credential`, `Authorization`, `Bearer`, and
absolute filesystem paths in general returned no matches, other than
this document and `.env.example`'s intentionally generic placeholder
path (`/path/to/dataset`).
