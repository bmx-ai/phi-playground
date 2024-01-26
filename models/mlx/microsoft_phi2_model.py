# inspired from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/phi2.py
import math
from dataclasses import dataclass
from typing import Tuple
import inspect
import json
import glob
from pathlib import Path
from typing import Generator

import mlx.core as mx
import mlx.nn as nn

@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

@dataclass
class ModelArgs(BaseModelArgs):
    max_position_embeddings: int = 2048
    vocab_size: int = 51200
    hidden_size: int = 2560
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: int = 32
    partial_rotary_factor: float = 0.4
    intermediate_size: int = 10240
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class LayerNorm(nn.LayerNorm):
    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.astype(mx.float32)).astype(x.dtype)


class PhiAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.repeats = self.num_heads // self.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.dense = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=True
        )

        self.rope = nn.RoPE(
            int(self.partial_rotary_factor * self.head_dim),
            traditional=False,
            base=self.rope_theta,
        )

    def __call__(self, x, mask=None, cache=None):
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Extract some shapes
        B, L, D = queries.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(
            B, L, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.num_heads, L, -1])

        if self.repeats > 1:
            keys, values = map(repeat, (keys, values))

        # Add RoPE to the queries and keys and combine them with the cache
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        queries = queries.astype(mx.float32)
        keys = keys.astype(mx.float32)

        # Finally perform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask

        scores = mx.softmax(scores, axis=-1).astype(values.dtype)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.dense(values_hat), (keys, values)


class PhiMLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU(approx="precise")

    def __call__(self, x) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class PhiDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.self_attn = PhiAttention(config=config)
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = PhiMLP(config)

    def __call__(self, x, mask, cache):
        h = self.input_layernorm(x)
        attn_h, cache = self.self_attn(h, mask, cache)
        ff_h = self.mlp(h)
        return attn_h + ff_h + x, cache


class PhiModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [PhiDecoderLayer(config) for i in range(config.num_hidden_layers)]
        self.final_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x, mask, cache):
        x = self.embed_tokens(x)
        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            x, cache[e] = layer(x, mask, cache[e])
        return self.final_layernorm(x), cache


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.model = PhiModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache: mx.array = None,
    ) -> Tuple[mx.array, mx.array]:
        mask = None
        if x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(x.dtype)

        y, cache = self.model(x, mask, cache)
        return self.lm_head(y), cache
    

def generate_step(
    prompt: mx.array,
    model: Model,
    temp: float = 0.0,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling, if 0 the argmax is used.
    Yields:
        Generator[Tuple[mx.array, mx.array]]: A generator producing
        one token and probability per call.
    """

    def sample(logits: mx.array) -> Tuple[mx.array, float]:
        softmax_logits = mx.softmax(logits)

        if temp == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            token = mx.random.categorical(logits * (1 / temp))

        prob = softmax_logits[0, token]
        return token, prob

    y = prompt
    B = 1
    H = 32
    C = 32
    D = 2048
    cache = None
    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y, prob = sample(logits)
        yield y, prob


def load_model(model_path: str):
    import os
    weight_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    with open(os.path.join(model_path, "config.json")) as fd:
        config = json.load(fd)
    model_args = ModelArgs.from_dict(config)
    model = Model(model_args)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    model.eval()
    return model 

def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path)


def load(model_path):
    model, tokenizer = load_model(model_path), load_tokenizer(model_path)
    return model, tokenizer


def benchmark(model, x, cache, runs=10):
    import time
    # warmup
    for _ in range(runs):
        y, c = model(x, mask=None, cache=cache)
        mx.eval(y, c)
    # measure
    start = time.time()
    rs = []
    for _ in range(runs):
        y, c = model(x, mask=None, cache=cache)
        rs.append((y, c))
    mx.eval(rs)
    end = time.time()

    return (end - start) * 1000 / 5