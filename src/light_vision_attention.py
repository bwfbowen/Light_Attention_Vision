# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""(Multi-Head) Attention module for use in Transformer architectures.
Altered the haiku version MultiHeadAttention module to be similar to Flax ones.
"""

import types
from typing import (Any, Callable, Optional, Tuple)
import warnings
import functools 

import jax 
from jax import numpy as jnp
from jax import lax
from flax.linen.linear import PrecisionLike
from flax.linen.module import merge_param
import haiku as hk

from fast_attention import make_fast_generalized_attention, make_fast_softmax_attention
from attention import dot_product_attention, combine_masks

Array = Any
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any


def img_to_patch(x):
    """ Converts RGB array to patches
    Args:
        x - jax.numpy.array representing the image of shape [B, H, W, C]
    Returns:
        x: jax.numpy.array of shape [B, H*W, C]
    """
    B, H, W, C = x.shape
    x = x.reshape(B, -1, C)
    return x


def dropout(x, rate, deterministic: bool = True):
    """Flax-style haiku dropout"""
    if deterministic: return x
    else: return hk.dropout(hk.next_rng_key(), rate, x)

class InputLayer(hk.Module):
  """Converts RGB array to patches and projects to given embedding shape"""
  def __init__(
      self,
      embed_dim: int, 
      name: Optional[str] = None
  ):
      super().__init__(name=name)
      self.embed_dim = embed_dim
    
  def __call__(self, x):
      x = img_to_patch(x)
      x = hk.Linear(self.embed_dim)(x)
      return x 

class MultiHeadAttention(hk.Module):
  """Multi-headed attention (MHA) module."""

  def __init__(
      self,
      num_heads: int,
      w_init: Optional[hk.initializers.Initializer] = None,
      bias_init = jnp.zeros,
      dtype: Optional[Dtype] = None,
      param_dtype: Dtype = jnp.float32,
      qkv_features: Optional[int] = None,
      out_features: Optional[int] = None,
      broadcast_dropout: bool = True,
      dropout_rate: float = 0.,
      deterministic: Optional[bool] = None,
      precision: PrecisionLike = None,
      use_bias: bool = True,
      attention_fn: Callable[..., Array] = dot_product_attention,
      decode: bool = False,
      name: Optional[str] = None,
  ):
    """Initialises the module."""
    super().__init__(name=name)
    self.num_heads = num_heads
    self.dtype = dtype
    self.param_dtype = param_dtype 
    self.qkv_features = qkv_features
    self.out_features = out_features
    self.broadcast_dropout = broadcast_dropout
    self.dropout_rate = dropout_rate
    self.deterministic = deterministic
    self.precision = precision
    self.bias_init = bias_init
    self.use_bias = use_bias
    self.attention_fn = attention_fn
    self.decode = decode
    self.w_init = w_init

  def __call__(
      self,
      x,
      mask: Optional[Array] = None,
      deterministic: Optional[bool] = None):
    inputs_q = x
    inputs_kv = x
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    projection = self._linear_projection
    # Compute key/query/values (overload K/Q/V to denote the respective sizes).
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query = projection(inputs_q, head_dim, "query")  # [T', H, Q=K]
    key = projection(inputs_kv, head_dim, "key")  # [T, H, K]
    value = projection(inputs_kv, head_dim, "value")  # [T, H, V]

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable('cache', 'cached_key',
                                 jnp.zeros, key.shape, key.dtype)
      cached_value = self.variable('cache', 'cached_value',
                                   jnp.zeros, value.shape, value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        *batch_dims, max_length, num_heads, depth_per_head = (
            cached_key.value.shape)
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError('Autoregressive cache shape error, '
                           'expected query shape %s instead got %s.' %
                           (expected_shape, query.shape))
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # causal mask for cached decoder self-attention:
        # our single query position should only attend to those key
        # positions that have already been generated and cached,
        # not the remaining zero elements.
        mask = combine_masks(
            mask,
            jnp.broadcast_to(jnp.arange(max_length) <= cur_index,
                             tuple(batch_dims) + (1, 1, max_length)))

    dropout_rng = None
    if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
      m_deterministic = merge_param('deterministic', self.deterministic,
                                    deterministic)
      if not m_deterministic:
        dropout_rng = self.make_rng('dropout')
    else:
      m_deterministic = True

    # apply attention
    x = self.attention_fn(
        query,
        key,
        value,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=m_deterministic,
        dtype=self.dtype,
        precision=self.precision)  # pytype: disable=wrong-keyword-args
    # back to the original inputs dimensions
    B, T, H, V = x.shape
    final_projection = hk.Linear(features,
                                 w_init=self.w_init,
                                 b_init=self.bias_init,
                                 with_bias=self.use_bias)
    
    return final_projection(x.reshape(B, T, -1))
  
  @hk.transparent
  def _linear_projection(
      self,
      x: jnp.ndarray,
      head_size: int,
      name: Optional[str] = None,
  ) -> jnp.ndarray:
    y = hk.Linear(self.num_heads * head_size,
                  w_init=self.w_init,
                  b_init=self.bias_init,
                  with_bias=self.use_bias,
                  name=name)(x)
    *leading_dims, _ = x.shape
    return y.reshape((*leading_dims, self.num_heads, head_size))

class AttentionBlock(hk.Module):
    def __init__(self,
                 embed_dim : int,   
                 hidden_dim : int,  
                 num_heads : int,   
                 dropout_prob : float = 0.0,  
                 use_fask_attn: bool = True,
                 name: str = None
    ):
      """
      Args:
        embed_dim: Dimensionality of input and attention feature vectors
        hidden_dim: Dimensionality of hidden layer in feed-forward network
        num_heads: Number of heads to use in the Multi-Head Attention block
        dropout_prob: Amount of dropout to apply in the feed-forward network
        use_fask_attn: Whether or not to use fast attention module
      """
      super().__init__(name=name)
      self.embed_dim = embed_dim
      self.hidden_dim = hidden_dim  
      self.num_heads = num_heads 
      self.dropout_prob = dropout_prob  
      self.use_fask_attn = use_fask_attn 
      self.qkv_dim = self.embed_dim
      if self.use_fask_attn:
        attn_func = make_fast_softmax_attention(
            self.qkv_dim // self.num_heads, 
            nb_features=self.qkv_dim
            )
        self.attn = MultiHeadAttention(
            num_heads=self.num_heads, 
            attention_fn=attn_func
            )
      else:
        self.attn = MultiHeadAttention(
            num_heads=self.num_heads,
            attention_fn=dot_product_attention
        )

    #   self.dense1 = hk.Linear(self.hidden_dim)
    #   self.dense2 = hk.Linear(self.embed_dim)
      self.layer_norm_1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      self.layer_norm_2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, x, train=True):
        self.dropout = functools.partial(dropout,
                                         rate=self.dropout_prob,
                                         deterministic=not train)
        # TODO: without MLP layers
        # self.linear = [
        #     self.dense1,
        #     jax.nn.gelu,
        #     self.dropout,
        #     self.dense2
        #     ]
        inp_x = self.layer_norm_1(x)
        attn_out = self.attn(x=inp_x)
        x = x + self.dropout(attn_out)

        linear_out = self.layer_norm_2(x)
        # for l in self.linear:
        #     linear_out = l(linear_out)
        x = x + self.dropout(linear_out)
        return x    

class PositionEmbeddings(hk.Module):
    """
    A position embedding of shape [1, 1+max number of patches, embedding dim]
    """
    def __init__(self, num_patches: int, embed_dim: int, name: str = None):
        super().__init__(name=name)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.pos_embed_shape = (1, 1+self.num_patches, self.embed_dim)

    def __call__(self, x):
        position_weights = hk.get_parameter(
            "position_embeddings", 
            self.pos_embed_shape,
            init=hk.initializers.Constant(1)
        )
        return position_weights[:, :self.num_patches]        

class VisionAttn(hk.Module):
    def __init__(self, 
                 embed_dim : int,     
                 hidden_dim : int,    
                 num_heads : int,     
                 num_layers : int,        
                 num_patches : int,   
                 dropout_prob : float = 0.0,  
                 use_fask_attn: bool = False,
                 name: str = None  
    ):
        """
        embed_dim: Dimensionality of input and attention feature vectors
        hidden_dim: Dimensionality of hidden layer in feed-forward network
        num_heads: Number of heads to use in the Multi-Head Attention block
        num_layers: Number of layers to use in the Transformer
        num_patches: Maximum number of patches an image can have
        dropout_prob: Amount of dropout to apply in the feed-forward network 
        use_fask_attn: Whether or not to use fast attention module
        name: Name of the module
        """
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_patches = num_patches
        self.dropout_prob = dropout_prob
        self.use_fask_attn = use_fask_attn

        self.input_layer = InputLayer(embed_dim=self.embed_dim,
                                      name='InputLayer')
        self.transformer = [AttentionBlock(embed_dim=self.embed_dim, 
                                           hidden_dim=self.hidden_dim,
                                           num_heads=self.num_heads,
                                           dropout_prob=self.dropout_prob,
                                           use_fask_attn=self.use_fask_attn) for _ in range(self.num_layers)]
        self.pos_embedding = PositionEmbeddings(num_patches=self.num_patches, 
                                                embed_dim=self.embed_dim)                                    
        
    def __call__(self, x, train=True):
        self.dropout = functools.partial(dropout, 
                                         rate=self.dropout_prob,
                                         deterministic=not train)
        # Preprocess input
        x = self.input_layer(x)
        B, T, _ = x.shape

        # Add positional encoding
        pos_embedding = self.pos_embedding(x)
        x = x + pos_embedding[:,:T]

        # Apply Transforrmer
        x = self.dropout(x)
        for attn_block in self.transformer:
            x = attn_block(x, train=train)

        return x    