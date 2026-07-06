from pathlib import Path
import subprocess
import sys


def main() -> None:
    # 1. download the training text if it doesn't exist
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    file_path = project_root / "data/raw/the-verdict.txt"
    download_script = src_dir / "ch2_1_download_training_text.py"

    if file_path.exists():
        print(f"Already exists: {file_path}")
        return

    print(f"Missing file, running download script: {download_script}")
    subprocess.run([sys.executable, str(download_script)], check=True, cwd=project_root)


if __name__ == "__main__":
    main()
