from importlib.metadata import version
import tiktoken

# tiktoken 은 앞서 만들었던 SimpleTokenizer와 동일한 역할을 수행하는 라이브러리이다.


## tiktoken 사용법

print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = ( 
    "Hello, world! This is a test of the tiktoken tokenizer. <|endoftext|> Hi there! This is another test. Let's see how well it tokenizes this text."
    " It should be able to handle various punctuation, whitespace, and special characters."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})   
print("Encoded integers:", integers)

strings = tokenizer.decode(integers)
print("Decoded string:", strings)


