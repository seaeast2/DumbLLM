
import re

# 간단한 토크나이저를 구현한 클래스
class SimpleTokenizerV1:
    def __init__(self, vocab):
        # encode 메서드와 decode 메서드에서 참조할 수 있도록
        # 어휘사전을 클래스의 속성으로 저장
        self.str_to_int = vocab 
        # 키와 값의 위치를 바꾼 역어휘사전을 만듦
        self.int_to_str = {i:s for s, i in vocab.items()} 

    # 입력 텍스트를 처리하여 토큰 ID로 바꾼다.
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    # 토큰 ID를 처리하여 원래 텍스트로 되돌린다.
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # 공백과 구두점 사이의 공백을 제거
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text