from typing import NamedTuple

import einops as op
import jax
from jax import Array
import jax.numpy as jnp
import jax.nn as nn
from functools import partial
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental.pallas.ops.tpu import flash_attention
import os
import math
from typing import Any
import jax.random as rand
from itertools import repeat
from tqdm import tqdm


class Proj(NamedTuple):
    weight: Array
    bias: Array

class LayerNorm(NamedTuple):
    weight: Array
    bias: Array

class Attention(NamedTuple):
    q_proj: Proj
    k_proj: Proj
    v_proj: Proj
    dense: Proj

class DecoderBlock(NamedTuple):
    input_layernorm: LayerNorm
    attention: Attention
    fc1: Proj
    fc2: Proj

class PhiModel(NamedTuple):
    embedding: Array
    decoder: DecoderBlock
    final_layernorm: LayerNorm

class Phi(NamedTuple):
    model: PhiModel
    lm_head: Proj

class ModelConfig(NamedTuple):
    d_ff: int
    head_dim: int
    d_model: int
    n_heads_kv: int
    n_layers: int
    n_rep_kv: int
    partial_rotary_factor: float
    layer_norm_epsilon: float
    vocab_size: int

    # TODO: move out of model config
    dropout_rate: float | None
    return_kv_cache: bool

phi_config = ModelConfig(
    d_ff=11008,
    head_dim=80,
    d_model=2560,
    n_heads_kv=32,
    n_layers=32,
    n_rep_kv=1,
    partial_rotary_factor=0.4,
    layer_norm_epsilon=1e-6,
    vocab_size=32000,
    dropout_rate=0.1,
    return_kv_cache=False,
)

from jax import Array
import torch
import torch.nn as tnn
from transformers import PhiForCausalLM, PhiModel as PhiModelPt
from transformers.models.phi.modeling_phi import PhiAttention, PhiDecoderLayer

import torch
from jax import Array
import jax.numpy as jnp
import numpy as np

def jax2np(x: Array) -> np.ndarray:
    '''
    Converts a JAX array into a NumPy array.

    Args:
        x (Array): JAX array to convert.

    Returns:
        np.ndarray: Converted NumPy array.
    '''
    return np.asarray(x)

def np2jax(x: np.ndarray) -> Array:
    '''
    Converts a NumPy array into a JAX array.

    Args:
        x (np.ndarray): NumPy array to convert.

    Returns:
        Array: Converted JAX array.
    '''
    return jnp.asarray(x)

def pt2np(x: torch.Tensor) -> np.ndarray:
    '''
    Converts a PyTorch tensor into a NumPy array.

    Args:
        x (torch.Tensor): PyTorch tensor to convert.

    Returns:
        np.ndarray: Converted NumPy array.
    '''
    with torch.no_grad():
        return x.numpy()

def np2pt(x: np.ndarray) -> torch.Tensor:
    '''
    Converts a NumPy array into a PyTorch tensor.

    Args:
        x (np.ndarray): NumPy array to convert.

    Returns:
        torch.Tensor: Converted PyTorch tensor.
    '''
    return torch.from_numpy(x)

def jax2pt(x: Array) -> torch.Tensor:
    '''
    Converts a JAX array into a PyTorch tensor using NumPy as intermediate.

    Args:
        x (Array): JAX array to convert.

    Returns:
        torch.Tensor: Converted PyTorch tensor.
    '''
    return np2pt(jax2np(x))

def pt2jax(x: torch.Tensor) -> Array:
    '''
    Converts a PyTorch tensor into a JAX array using NumPy as intermediate.

    Args:
        x (torch.Tensor): PyTorch tensor to convert.

    Returns:
        Array: Converted JAX array.
    '''
    return np2jax(pt2np(x))

def stack_leaves(pytrees, axis: int=0):
    return jax.tree_map(lambda *xs: jnp.stack(xs, axis=axis), *pytrees)


def convert_q_proj(x: tnn.Linear, *, model_config: ModelConfig) -> Array:
    weight = pt2jax(x.weight.T.reshape(model_config.d_model, model_config.n_heads_kv,  model_config.n_rep_kv, model_config.head_dim)).transpose(0, 2, 1, 3)
    bias = pt2jax(x.bias)
    return Proj(weight=weight, bias=bias)

def convert_k_proj(x: tnn.Linear, *, model_config: ModelConfig) -> Array:
    weight = pt2jax(x.weight.T.reshape(model_config.d_model, model_config.n_heads_kv, model_config.head_dim))
    bias = pt2jax(x.bias)
    return Proj(weight=weight, bias=bias)

def convert_v_proj(x: tnn.Linear, *, model_config: ModelConfig) -> Array:
    weight = pt2jax(x.weight.T.reshape(model_config.d_model, model_config.n_heads_kv, model_config.head_dim))
    bias = pt2jax(x.bias)
    return Proj(weight=weight, bias=bias)

def convert_out_proj(x: tnn.Linear, *, model_config: ModelConfig) -> Array:
    weight = pt2jax(x.weight.T.reshape(model_config.n_heads_kv, model_config.n_rep_kv, model_config.head_dim, model_config.d_model)).transpose(1, 0, 2, 3)
    bias = pt2jax(x.bias)
    return Proj(weight=weight, bias=bias)

def convert_attention(x: PhiAttention, *, model_config: ModelConfig) -> Attention:
    q_proj = convert_q_proj(x.q_proj, model_config=model_config)
    k_proj = convert_k_proj(x.k_proj, model_config=model_config)
    v_proj = convert_v_proj(x.v_proj, model_config=model_config)
    dense = convert_out_proj(x.dense, model_config=model_config)
    return Attention(q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, dense=dense)

def convert_layernorm(x: Any) -> Attention:
    return LayerNorm(weight=pt2jax(x.weight), bias=pt2jax(x.bias))

def convert_decoder_block(x: PhiDecoderLayer, *, model_config: ModelConfig) -> DecoderBlock:
    input_norm = pt2jax(x.input_layernorm.weight)
    attention = convert_attention(x.self_attn, model_config=model_config)
    input_layernorm = convert_layernorm(x.input_layernorm)
    fc1 = Proj(weight=pt2jax(x.mlp.fc1.weight.T), bias=pt2jax(x.mlp.fc1.bias))
    fc2 = Proj(weight=pt2jax(x.mlp.fc2.weight.T), bias=pt2jax(x.mlp.fc2.bias))
    return DecoderBlock(input_layernorm=input_layernorm, attention=attention, fc1=fc1, fc2=fc2)

def convert_phi_model(model: PhiModelPt, *, model_config: ModelConfig) -> PhiModel:
    embedding = pt2jax(model.embed_tokens.weight)
    decoder = stack_leaves([convert_decoder_block(model.layers[i], model_config=model_config) for i in tqdm(range(model_config.n_layers))])
    final_layernorm = convert_layernorm(model.final_layernorm)
    return PhiModel(embedding=embedding, decoder=decoder, final_layernorm=final_layernorm)

def convert_phi(model_pt: PhiForCausalLM, *, model_config: ModelConfig) -> Phi:
    with torch.no_grad():
        model = convert_phi_model(model_pt.model, model_config=model_config)
        lm_head = Proj(weight=pt2jax(model_pt.lm_head.weght.T), bias=pt2jax(model_pt.lm_head.bias))
        return Phi(model=model, lm_head=lm_head)


class KVCache(NamedTuple):
    k_cache: Array
    v_cache: Array

class RotaryValues(NamedTuple):
    sin_val: Array
    cos_val: Array

def forward_embedding(params: Array, x: Array) -> Array:
    return params[x]

def split_key_nullable(key: Array | None, num: int=2):
    if key is None:
        return tuple(repeat(None, num))
    return rand.split(key, num)

@partial(jax.jit, static_argnames=('model_config',))
def forward_dropout(x: Array, *, key: Array | None=None, model_config: ModelConfig) -> Array:
    if key is None or model_config.dropout_rate is None:  # should disable dropout
        return x

    assert 0. <= model_config.dropout_rate <= 1.
    assert isinstance(x, Array)
    assert isinstance(key, Array)

    keep_rate = 1. - model_config.dropout_rate
    out = x * rand.bernoulli(key, p=keep_rate, shape=x.shape) / keep_rate
    assert x.shape == out.shape
    return out

def shift_left_kv_cache(kv_cache: KVCache) -> KVCache:
    k_cache, v_cache = kv_cache
    k_cache = jnp.roll(k_cache, -1, axis=-2)  # -2: dimension L
    v_cache = jnp.roll(v_cache, -1, axis=-2)  # -2: dimension L
    return KVCache(k_cache, v_cache)

def _make_weights(seq_len: int, d_k: int) -> tuple[Array, Array]:
    inv_freq = 1. / (10000 ** (jnp.arange(0, d_k, 2) / d_k))
    sinusoid_inp = op.einsum(jnp.arange(seq_len), inv_freq, 'L, j -> L j')
    sin_val = jnp.sin(sinusoid_inp)
    cos_val = jnp.cos(sinusoid_inp)
    sin_val = op.repeat(sin_val, 'L K -> L (i K)', i=2)
    cos_val = op.repeat(cos_val, 'L K -> L (i K)', i=2)
    return sin_val, cos_val

def _rotate_half(x: Array) -> Array:
    x = op.rearrange(x, '... (i x) -> ... i x', i=2)  # split the last dimension: (..., n) -> (..., 2, n // 2)
    x = x[..., ::-1, :]  # reverse dimension -2
    x = x.at[..., 0, :].multiply(-1)  # negate the first half of dimension -2
    x = op.rearrange(x, '... i x -> ... (i x)')  # merge the last two dimensions: (..., 2, n // 2) -> (..., n)
    return x

def forward_rotary_embedding(m: Array, *, rotary_values: RotaryValues) -> Array:
    sin_val, cos_val = rotary_values
    assert sin_val.dtype == jnp.float32
    assert cos_val.dtype == jnp.float32
    n = _rotate_half(m)
    a = op.einsum(m, cos_val, 'B ... L K, B L K -> B ... L K').astype(m.dtype)
    b = op.einsum(n, sin_val, 'B ... L K, B L K -> B ... L K').astype(m.dtype)
    return a + b

def make_rotary_values(leftpad_len: Array | None, batch_size: int, seq_len: int, *, model_config: ModelConfig) -> RotaryValues:
    rotary_dim = int(model_config.partial_rotary_factor * model_config.head_dim)
    sin_val, cos_val = _make_weights(seq_len, rotary_dim)

    sin_val = jnp.repeat(sin_val[None], batch_size, axis=0)
    cos_val = jnp.repeat(cos_val[None], batch_size, axis=0)

    if leftpad_len is not None:
        roll_func = jax.vmap(lambda a, shift: jnp.roll(a, shift, axis=-2))  # -2: dimension L
        sin_val = roll_func(sin_val, leftpad_len)
        cos_val = roll_func(cos_val, leftpad_len)

    return RotaryValues(sin_val, cos_val)

def get_rotary_values_at_position(rotary_values: RotaryValues, position: Array) -> RotaryValues:
    sin_val, cos_val = rotary_values
    sin_val = sin_val[:, position][:, None]
    cos_val = cos_val[:, position][:, None]
    rotary_values = RotaryValues(sin_val, cos_val)
    return rotary_values

@partial(jax.jit, static_argnames=('model_config',))
def forward_layer_norm(params: Array, x: Array, *, model_config: ModelConfig) -> Array:
    return (x - x.mean(-1, keepdims=True)) / jnp.sqrt(x.var(-1, keepdims=True) + model_config.layer_norm_epsilon) * params.weight + params.bias

def repeat_kv_bnsh(x: Array, n_rep: int) -> Array:
    bs, n_kv_heads, s, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, jax.numpy.newaxis, :, :]
    x = jax.numpy.repeat(x, n_rep, axis=2)

    return x.reshape(bs, n_kv_heads * n_rep, s, head_dim)

@partial(jax.jit, static_argnames=('model_config',))
def forward_attention(params: Attention, src_seq: Array, dst_seq: Array, qk_mask: Array, *, rotary_values: RotaryValues, kv_cache: KVCache | None=None, model_config: ModelConfig) -> tuple[Array, KVCache | None]:

    q = op.einsum(src_seq, params.q_proj.weight, 'B S M, M R H K -> B R H S K')
    q += params.q_proj.bias.reshape(1, 1, q.shape[2], 1, q.shape[4])
    
    k = op.einsum(dst_seq, params.k_proj.weight, 'B D M, M H K -> B H D K')
    k += params.k_proj.bias.reshape(1, k.shape[1], 1, k.shape[3])
    
    v = op.einsum(dst_seq, params.v_proj.weight, 'B D M, M H V -> B H D V')
    v += params.v_proj.bias.reshape(1, v.shape[1], 1, k.shape[3])

    rotary_dim = int(model_config.partial_rotary_factor * model_config.head_dim)

    q_rot, q_pass = (
        q[..., :rotary_dim],
        q[..., rotary_dim:],
    )
    k_rot, k_pass = (
        k[..., :rotary_dim],
        k[..., rotary_dim:],
    )

    # [batch_size, seq_length, num_heads, head_dim * config.partial_rotary_factor]
    q_rot = forward_rotary_embedding(q_rot, rotary_values=rotary_values)
    k_rot = forward_rotary_embedding(k_rot, rotary_values=rotary_values)

    q = jnp.concatenate((q_rot, q_pass), axis=-1)
    k = jnp.concatenate((k_rot, k_pass), axis=-1)


    q_shape = q.shape

    if kv_cache is not None:
        assert src_seq.shape[1] == 1
        assert dst_seq.shape[1] == 1
        k_cache, v_cache = kv_cache
        k = k_cache.at[:, :, -1:].set(k)
        v = v_cache.at[:, :, -1:].set(v)



    qk = op.einsum(q, k, 'B R H S K, B H D K -> B R H S D')
    qk /= math.sqrt(model_config.head_dim)
    qk = jnp.where(qk_mask, qk, -jnp.inf)
    qk = nn.softmax(qk)  # TODO: use `where`
    qk = jnp.where(qk_mask, qk, 0)  # TODO: why this line?

    qkv = op.einsum(qk, v, 'B R H S D, B H D V -> B R H S V')
    out = op.einsum(qkv, params.dense.weight, 'B R H S V, R H V M -> B S M')

    out += params.dense.bias.reshape(1, 1, out.shape[2])
    
    kv_cache = None if not model_config.return_kv_cache else KVCache(k, v)

    return out, kv_cache

@partial(jax.jit)
def forward_mlp(params: DecoderBlock, seq: Array) -> Array:

    seq = seq @ params.fc1.weight + params.fc1.bias
    print(seq.shape)
    seq = jax.nn.gelu(seq, approximate=False)
    seq = seq @ params.fc2.weight + params.fc2.bias

    return seq

@partial(jax.jit, static_argnames=('model_config',))
def forward_decoder_block(params: DecoderBlock, seq: Array, qk_mask: Array, *, rotary_values: RotaryValues, kv_cache: KVCache | None=None, key: Array | None=None, model_config: ModelConfig) -> tuple[Array, KVCache | None]:
    key0, key1, key2 = split_key_nullable(key, num=3)
    
    seq_ = seq

    seq = forward_layer_norm(params.input_layernorm, seq, model_config=model_config)
    attn_seq, kv_cache = forward_attention(params.attention, seq, seq, qk_mask, rotary_values=rotary_values, kv_cache=kv_cache, model_config=model_config)
    attn_seq = forward_dropout(attn_seq, key=key0, model_config=model_config)
    
    mlp_seq = forward_mlp(params, seq)
    mlp_seq = forward_dropout(mlp_seq, key=key0, model_config=model_config)
    
    
    seq = mlp_seq + seq_ + attn_seq
    return seq, kv_cache

@partial(jax.jit, static_argnames=('model_config',))
def forward_decoder(params: DecoderBlock, seq: Array, qk_mask: Array, *, rotary_values: RotaryValues, kv_cache: KVCache | None=None, key: Array | None=None, model_config: ModelConfig) -> tuple[Array, KVCache | None]:
    def inner(state, input_):
        key, seq = state
        params, kv_cache = input_
        key, subkey = split_key_nullable(key)
        seq, kv_cache = forward_decoder_block(params, seq, qk_mask, rotary_values=rotary_values, kv_cache=kv_cache, key=subkey, model_config=model_config)
        return (key, seq), kv_cache

    (key, seq), kv_cache = jax.lax.scan(inner, (key, seq), (params, kv_cache))
    return seq, kv_cache

@partial(jax.jit, static_argnames=('model_config'))
def forward_phi_model(params: PhiModel, seq: Array, qk_mask: Array, *, rotary_values: RotaryValues, kv_cache: KVCache | None=None, key: Array | None=None, model_config: ModelConfig) -> tuple[Array, KVCache | None]:
    assert isinstance(seq, Array)
    assert isinstance(qk_mask, Array)
    assert qk_mask.dtype == jnp.bool_
    assert model_config.head_dim % 2 == 0
    assert key is None or model_config.dropout_rate is not None

    seq = forward_embedding(params.embedding, seq)

    seq, kv_cache = forward_decoder(params.decoder, seq, qk_mask, rotary_values=rotary_values, kv_cache=kv_cache, key=key, model_config=model_config)
    seq = forward_layer_norm(params.final_layernorm, seq, model_config=model_config)
    return seq, kv_cache

@partial(jax.jit, static_argnames=('model_config'))
def forward_phi(params: Phi, seq: Array, qk_mask: Array, *, rotary_values: RotaryValues, kv_cache: KVCache | None=None, key: Array | None=None, model_config: ModelConfig) -> tuple[Array, KVCache | None]:
    outputs, kv_cache = forward_phi_model(params.model, seq, qk_mask, rotary_values=rotary_values, kv_cache=kv_cache, key=key, model_config=model_config)
    logits = outputs @ params.lm_head.weight.T
    logits += params.lm_head.bias.reshape(1, 1, -1)
    return logits, kv_cache
