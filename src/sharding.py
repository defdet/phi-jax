from types import EllipsisType
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from jax.experimental import mesh_utils
import gc
import jax
from .modeling_phi import Phi, PhiModel, Attention, DecoderBlock, Layernorm, Proj

def shard_array(arr: Array, axes: tuple | EllipsisType) -> Array:
    num_axes = 1 if isinstance(axes, EllipsisType) else len(axes)
    if num_axes == 2:
        n_devices = jax.device_count()
        devices = mesh_utils.create_device_mesh((n_devices, ))
        if n_devices == 32:
            device_tuple = (4, 8)
        else:
            device_tuple = (2, n_devices // 2)          
    elif num_axes == 3:
        device_tuple = (2, 2, 4)
    else:
        device_tuple = (jax.device_count(), )
    
    devices = mesh_utils.create_device_mesh((jax.device_count(), ))
    shape = arr.shape

    if axes is ...:
        mesh = Mesh(devices, ('a',))
        sharding = NamedSharding(mesh, P(None))
    else:
        sharding_tuple_ = [1] * len(shape)
        for axis_num, axis in enumerate(axes):
            sharding_tuple_[axis]=device_tuple[axis_num]
        sharding_tuple = tuple(sharding_tuple_)
        name_tuple = tuple('abcdefghijklmnopqrstuvwxyz'[:len(shape)])
        mesh = Mesh(devices.reshape(sharding_tuple), name_tuple)     
        sharding = NamedSharding(mesh, P(*name_tuple))

    xs = [jax.device_put(arr[i], device) for device, i in sharding.addressable_devices_indices_map(shape).items()]
    return jax.make_array_from_single_device_arrays(shape, sharding, xs)

sharding_mp = Phi(
    model=PhiModel(
        embedding=...,
        decoder=DecoderBlock(
            input_layernorm=Layernorm(weight=..., bias=...),
            attention=Attention(q_proj=Proj(weight=(1, 3), bias=...), k_proj=Proj(weight=(1, 2), bias=...), v_proj=Proj(weight=(1, 2), bias=...), dense=Proj(weight=(2, 4), bias=...)),
            gate_proj=Proj(weight=(1, 2), bias=...),
            up_proj=Proj(weight=(1, 2), bias=...),
            down_proj==Proj(weight=(2, 1), bias=...),
        ),
        final_layernorm=Layernorm(weight=..., bias=...),
    ),
    lm_head=Proj(weight=..., bias=...),
)

def shard_model_params(params: Phi) -> Phi:
    return jax.tree_map(shard_array, params, sharding_mp)
