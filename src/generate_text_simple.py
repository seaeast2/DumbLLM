

# idx 는 현재 문맥이 담긴 (batch, num_tokens) 크기의 인덱스 배열입니다.
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # 현재 문맥이 모델이 지원하는 문맥 크기를 초과하면 잘라냅니다.
        # 예를 들어, LLM 이 5개 토큰만 지원하고 문맥 크기가 10이라면,
        # 마지막 5개 토큰만 문맥으로 사용합니다.
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
            
        # 마지막 타임 스텝만 사용하므로(batch, num_tokens, vocab_size)가
        # (batch, vocab_size) 가 됩니다.
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1) # probas 의 크기는 (batch, vocab_size)입니다.
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # idx_next의 크기는 (batch, 1) 입니다.
        
        # 선택한 인덱스를 현재 시퀀스에 추가하므로 idx 의 크기는 
        # (batch, num_tokens +1) 가 됩니다.
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
                        