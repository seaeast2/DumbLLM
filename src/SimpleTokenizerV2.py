import re

# <|unk|>, <|endoftext|> 의 추가 토큰을 지원하도록 만든 간단 토크나이저
class SimpleTokenizerV2:
    def __init__(self, vocab):
        # 어휘사전 저장
        self.str_to_int = vocab
        # 키와 값의 위치를 바꾼 역어휘사전을 만듦
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        # 입력 텍스트에 어휘사전에 없는 단어가 있으면 <|unk|> 토큰으로 대체
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # 공백과 구두점 사이의 공백을 제거
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

    