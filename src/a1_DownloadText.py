from pathlib import Path
import urllib.request


TRAINING_TEXT_URL = (
    "https://raw.githubusercontent.com/rickiepark/"
    "llm-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
)


def download_training_text() -> Path:
    """Download the training text once and return its local path."""
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "data" / "raw" / "the-verdict.txt"

    if file_path.exists():
        print(f"Already exists: {file_path}")
        return file_path

    file_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading training text: {file_path}")
    urllib.request.urlretrieve(TRAINING_TEXT_URL, file_path)
    return file_path
