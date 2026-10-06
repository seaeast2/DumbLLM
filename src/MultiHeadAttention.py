import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    mask: torch.Tensor

    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out 은 num_heads 로 나누어 떨어져야 합니다."

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # 원하는 출력 차원에 맞도록 투영 차원을 낮춥니다.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # Linear 층을 사용해 헤드의 출력을 결합합니다.
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 텐서크기: (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # num_heads 차원을 추가함으로써 암묵적으로 행렬을 분할합니다.
        # 그런 다음 마지막 차원을 num_heads 에 맞춰 채웁니다.
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # (b, num_tokens, num_heads, head_dim) 크기를
        # (b, num_heads, num_tokens, head_dim) 크기로 바꿉니다.
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 각 헤드에 대해 점곱을 수행합니다.
        attn_scores = queries @ keys.transpose(2, 3)
        # 토큰의 개수로 마스크를 자릅니다.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 마스크를 사용하여 attention scores를 채웁니다.
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        #텐서크기 : (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # 헤드를 결합합니다. self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # 선영투영을 추가합니다.
        return context_vec
        
