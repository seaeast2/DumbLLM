from pathlib import Path
import re

from src.SimpleTokenizerV2 import SimpleTokenizerV2
import tiktoken

# 간단한 토크나이저를 사용하여 텍스트를 인코딩하고 디코딩하는 예제입니다.
def run_embedding1(file_path: Path) -> list[int]:
    """Load the training text, then encode and decode a sample with V2."""
    with file_path.open("r", encoding="utf-8") as file:
        raw_text = file.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    tokens = [token.strip() for token in preprocessed if token.strip()]
    vocab = {
        token: index
        for index, token in enumerate(
            sorted(set(tokens)) + ["<|unk|>", "<|endoftext|>"]
        )
    }

    tokenizer = SimpleTokenizerV2(vocab)
    text_to_encode = raw_text[:500]
    encoded_text = tokenizer.encode(text_to_encode)
    # decoded_text = tokenizer.decode(encoded_text)

    # print("Total characters:", len(raw_text))
    # print("\nEncoded text:", encoded_text)
    # print("Decoded text:", decoded_text)
    return encoded_text


# tiktoken 을 tokenizer 로 사용하여 텍스트를 인코딩하고 디코딩하는 예제입니다.
def run_embedding2(file_path: Path) -> list[int]:
    """tiktoken 을 tokenizer 로 사용하여 텍스트를 인코딩하고 디코딩하는 예제입니다."""
    with file_path.open("r", encoding="utf-8") as file:
        raw_text = file.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_text = tokenizer.encode(raw_text)
    # decoded_text = tokenizer.decode(encoded_text)

    # print("Total characters:", len(raw_text))
    # print("\nEncoded text:", encoded_text)
    # print("Decoded text:", decoded_text)
    return encoded_text