# Repository Guidelines

## Project Structure & Module Organization

This is a small Python learning project that implements tokenizer examples from an LLM study workflow. `run.py` is the application entry point: it downloads sample training text and runs the tokenizer demonstration. Keep reusable code in `src/`:

- `download_text.py` retrieves the source corpus.
- `embeding.py` assembles a vocabulary and runs the V2 tokenizer example.
- `SimpleTokenizerV1.py`, `SimpleTokenizerV2.py`, and `BytePairEncoding.py` contain tokenizer experiments.

Downloaded inputs belong under `data/raw/`; do not commit generated corpora or virtual-environment files. Add new focused modules under `src/` and keep executable orchestration in `run.py` or a clearly named script.

## Build, Test, and Development Commands

Use Python 3 and a local virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py
```

`python run.py` downloads `data/raw/the-verdict.txt` when absent, then prints encoding and decoding output. Run individual explorations directly, for example `python src/BytePairEncoding.py`. The repository currently has no configured formatter, linter, or test runner.

## Coding Style & Naming Conventions

Follow existing Python conventions: four-space indentation, imports at the top, `snake_case` for functions and variables, and `PascalCase` for classes. Add short docstrings to public functions that perform I/O or processing. Preserve the current module names for compatibility; use descriptive `snake_case.py` filenames for newly introduced modules where practical. Prefer `pathlib.Path` and explicit UTF-8 encoding for file operations.

## Testing Guidelines

There is no automated test suite yet. Before committing, run `python run.py` and any changed standalone module. For new behavior, add `unittest` tests in `tests/` named `test_<feature>.py`; cover token encoding, decoding, punctuation handling, and unknown-token behavior. Keep tests offline by using temporary/local fixture text instead of downloading data.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, feature-focused summaries (often Korean), such as `토큰 인코딩 추가`. Keep each commit scoped to one change and write a concise summary in the repository's established language. Pull requests should explain the behavioral change, list validation commands run, link relevant issues when available, and include sample console output when tokenizer results change.
