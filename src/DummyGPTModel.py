import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # 더미 트랜스포머 블록을 사용한다.
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])])
        # 더미 층 정규화를 사용
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # 토큰 임베딩
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 위치 임베딩
        x = tok_embeds + pos_embeds  # 토큰 임베딩과 위치 임베딩을 합친다.
        x = self.drop_emb(x)  # 드롭아웃 적용
        x = self.trf_blocks(x)  # 더미 트랜스포머 블록 통과
        x = self.final_norm(x)  # 더미 층 정규화 적용
        logits = self.out_head(x)  # 출력 헤드 통과
        return logits


# 나중에 실제 트랜스포머 블록으로 교체될 간단한 더미 클래스
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    # 이 블록은 아무것도 하지 않고 입력을 그냥 반환한다.
    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    # 층 정규화 인터페이스를 흉내내기 위한 매개 변수
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
