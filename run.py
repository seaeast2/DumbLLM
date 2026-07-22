from src.download_text import download_training_text
from src.embeding import run_embedding


def main() -> None:
    file_path = download_training_text()
    run_embedding(file_path)


if __name__ == "__main__":
    main()
