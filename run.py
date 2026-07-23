from src.a1_DownloadText import download_training_text
# from src.a2_Embeding import run_embedding1
from src.a2_Embeding import run_embedding2
from src.a3_SlidingWindow import make_sliding_window


def main() -> None:
    file_path = download_training_text()
    #run_embedding1(file_path)
    encoded_tokens = run_embedding2(file_path)
    make_sliding_window(encoded_tokens, context_size=4)


if __name__ == "__main__":
    main()
