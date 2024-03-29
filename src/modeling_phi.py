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
from .ring_attention import ring_attention
from typing import Any
import jax.random as rand
from itertools import repeat


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

class KVCache(NamedTuple):
    k_cache: Array  # Array
    v_cache: Array  # Array

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
    size_num = 128
    attn_impl = os.getenv('ATTN_IMPL')
    n_devices = jax.device_count()
    devices = mesh_utils.create_device_mesh((n_devices, ))
    if n_devices == 32:
        device_tuple = (4, 8)
    else:
        device_tuple = (2, n_devices // 2)

    q_axes = (0, 2)
    k_axes = (0, 1)
    v_axes = (0, 1)
    out_axes = (0, 2)

    sharding_tuple_q = [1] * 5
    sharding_tuple_k = [1] * 4
    sharding_tuple_v = [1] * 4
    sharding_tuple_out = [1] * 3

    for axis_num, axis in enumerate(q_axes):
        sharding_tuple_q[axis]=device_tuple[axis_num]
    for axis_num, axis in enumerate(k_axes):
        sharding_tuple_k[axis]=device_tuple[axis_num]
    for axis_num, axis in enumerate(v_axes):
        sharding_tuple_v[axis]=device_tuple[axis_num]
    for axis_num, axis in enumerate(out_axes):
        sharding_tuple_out[axis]=device_tuple[axis_num]

    sharding_tuple_q = tuple(sharding_tuple_q)
    sharding_tuple_k = tuple(sharding_tuple_k)
    sharding_tuple_v = tuple(sharding_tuple_v)
    sharding_tuple_out = tuple(sharding_tuple_out)
    
    name_tuple_q = tuple('abcdefghijklmnopqrstuvwxyz'[:5])
    mesh_q = Mesh(devices.reshape(sharding_tuple_q), name_tuple_q)     
    sharding_q = NamedSharding(mesh_q, P(*name_tuple_q))

    name_tuple_k = tuple('abcdefghijklmnopqrstuvwxyz'[:4])
    mesh_k = Mesh(devices.reshape(sharding_tuple_k), name_tuple_k)     
    sharding_k = NamedSharding(mesh_k, P(*name_tuple_k))

    name_tuple_v = tuple('abcdefghijklmnopqrstuvwxyz'[:4])
    mesh_v = Mesh(devices.reshape(sharding_tuple_v), name_tuple_v)     
    sharding_v = NamedSharding(mesh_v, P(*name_tuple_v))

    name_tuple_out = tuple('abcdefghijklmnopqrstuvwxyz'[:3])
    mesh_out = Mesh(devices.reshape(sharding_tuple_out), name_tuple_out)     
    sharding_out = NamedSharding(mesh_out, P(*name_tuple_out))

    q = op.einsum(src_seq, params.q_proj.weight, 'B S M, M R H K -> B R H S K')
    q += params.q_proj.bias.reshape(1, 1, q.shape[2], 1, q.shape[4])
    
    k = op.einsum(dst_seq, params.k_proj.weight, 'B D M, M H K -> B H D K')
    k += params.k_proj.bias.reshape(1, k.shape[1], 1, k.shape[3])
    
    v = op.einsum(dst_seq, params.v_proj.weight, 'B D M, M H V -> B H D V')
    v += params.v_proj.bias.reshape(1, v.shape[1], 1, k.shape[3])

    q = jax.lax.with_sharding_constraint(q, sharding_q)
    k = jax.lax.with_sharding_constraint(k, sharding_k)
    v = jax.lax.with_sharding_constraint(v, sharding_v)

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

    if n_devices == 32:
        q = q.astype(jnp.float32)
        k = k.astype(jnp.float32)
        v = v.astype(jnp.float32)

    q_shape = q.shape

    if kv_cache is not None:
        assert src_seq.shape[1] == 1
        assert dst_seq.shape[1] == 1
        k_cache, v_cache = kv_cache
        k = k_cache.at[:, :, -1:].set(k)
        v = v_cache.at[:, :, -1:].set(v)

    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)

    q_shape = q.shape
    if attn_impl == 'normal':
        qk = op.einsum(q, k, 'B R H S K, B H D K -> B R H S D')
        qk /= math.sqrt(model_config.head_dim)
        qk = jnp.where(qk_mask, qk, -jnp.inf)
        qk = nn.softmax(qk)  # TODO: use `where`
        qk = jnp.where(qk_mask, qk, 0)  # TODO: why this line?

        qkv = op.einsum(qk, v, 'B R H S D, B H D V -> B R H S V')
        qkv = qkv.astype(jnp.bfloat16)
        out = op.einsum(qkv, params.dense.weight, 'B R H S V, R H V M -> B S M')
    else:

        # q = q.reshape(q.shape[0], model_config.n_rep_kv * model_config.n_heads_kv, q.shape[3], model_config.d_k)
        q = q.reshape(q_shape[0], q_shape[1] * q_shape[2], q_shape[3], q_shape[4]) # [B, H, S, K]
        q_shape = q.shape
    
        qk_mask = qk_mask.squeeze(1)
        qk_mask = jnp.broadcast_to(qk_mask, (qk_mask.shape[0], q_shape[1], q_shape[2], k.shape[2]))
    
    
        attention_bias = jax.lax.select(
                qk_mask == True,
                jnp.full(qk_mask.shape, 0.0).astype(jnp.bfloat16),
                jnp.full(qk_mask.shape, -10.0**6).astype(jnp.bfloat16),
            )
        specs_tuple = (P(*name_tuple_k),
                       P(*name_tuple_k),
                       P(*name_tuple_k),
                       P(*name_tuple_k))
        
        if attn_impl == 'flash':
            qkv = shard_map(partial(flash_attention, sm_scale=math.sqrt(model_config.head_dim), debug=False, causal=False, block_sizes=block_sizes), mesh=mesh_k, in_specs=specs_tuple, out_specs=P(*name_tuple_k), check_rep=False)(q, k, v, attention_bias)
        if attn_impl == 'ring':
            qkv = shard_map(partial(ring_attention, sm_scale=math.sqrt(model_config.head_dim), debug=False, causal=True), mesh=mesh_k, in_specs=specs_tuple, out_specs=P(*name_tuple_k), check_rep=False)(q, k, v, attention_bias)
            
        qkv = qkv.astype(jnp.bfloat16)
    
        qkv = qkv.reshape(qkv.shape[0], model_config.n_rep_kv, qkv.shape[1] // model_config.n_rep_kv, qkv.shape[2], -1)
        out = op.einsum(qkv, params.dense.weight, 'B R H S V, R H V M -> B S M')

    out = jax.lax.with_sharding_constraint(out, sharding_out)
    out += params.dense.bias.reshape(1, 1, out.shape[2])
    
    kv_cache = None if not model_config.return_kv_cache else KVCache(k, v)

    return out, kv_cache

@partial(jax.jit, static_argnames=('model_config',))
def forward_decoder_block(params: DecoderBlock, seq: Array, qk_mask: Array, *, rotary_values: RotaryValues, kv_cache: KVCache | None=None, key: Array | None=None, model_config: ModelConfig) -> tuple[Array, KVCache | None]:
    key0, key1, key2 = split_key_nullable(key, num=3)
    n_devices = jax.device_count()
    devices = mesh_utils.create_device_mesh((n_devices, ))
    if n_devices == 32:
        device_tuple = (4, 8)
    else:
        device_tuple = (2, n_devices // 2)

    ff_axes = (0, 2)
    seq_axes = (0, 2)

    sharding_tuple_ff = [1] * 3
    sharding_tuple_seq = [1] * 3

    for axis_num, axis in enumerate(ff_axes):
        sharding_tuple_ff[axis]=device_tuple[axis_num]
    for axis_num, axis in enumerate(seq_axes):
        sharding_tuple_seq[axis]=device_tuple[axis_num]

    sharding_tuple_ff = tuple(sharding_tuple_ff)
    sharding_tuple_seq = tuple(sharding_tuple_seq)

    name_tuple_ff = tuple('abcdefghijklmnopqrstuvwxyz'[:3])
    mesh_ff = Mesh(devices.reshape(sharding_tuple_ff), name_tuple_ff)     
    sharding_ff = NamedSharding(mesh_ff, P(*name_tuple_ff))

    name_tuple_seq = tuple('abcdefghijklmnopqrstuvwxyz'[:3])
    mesh_seq = Mesh(devices.reshape(sharding_tuple_seq), name_tuple_seq)     
    sharding_seq = NamedSharding(mesh_seq, P(*name_tuple_seq))
    
    seq_ = seq

    seq = forward_layer_norm(params.input_layernorm, seq, model_config=model_config)
    attn_seq, kv_cache = forward_attention(params.attention, seq, seq, qk_mask, rotary_values=rotary_values, kv_cache=kv_cache, model_config=model_config)
    attn_seq = forward_dropout(attn_seq, key=key0, model_config=model_config)


    seq = seq @ params.fc1.weight.T + params.fc1.bias.reshape(1, 1, -1)
    seq = jax.lax.with_sharding_constraint(seq, sharding_ff)

    seq = jax.nn.silu(seq)
    seq = seq @ params.fc2.weight.T + params.fc2.bias.reshape(1, 1, -1)
    seq = jax.lax.with_sharding_constraint(seq, sharding_seq)
    seq = forward_dropout(seq, key=key0, model_config=model_config)
    
    seq += seq_ + attn_seq
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
