

# LLM 훈련-타깃 쌍을 생성하기 위하여 슬라이딩 윈도, 데이터 샘플 생성
# embedded 토큰들을 받아서 생성
def make_sliding_window(embedded_tokens: list[int], context_size: int) -> None:
    """슬라이딩 윈도우 생성"""
    x = embedded_tokens[:context_size]
    y = embedded_tokens[1:context_size + 1]
    print(f"x: {x}")
    print(f"y:     {y}")