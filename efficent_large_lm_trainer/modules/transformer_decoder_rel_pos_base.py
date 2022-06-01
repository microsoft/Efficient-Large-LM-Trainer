# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer import TransformerDecoderBase
from fairseq.modules.checkpoint_activations import checkpoint_wrapper

from .transformer_layer_rel_pos import TransformerDecoderLayerRelPosBase


class TransformerDecoderRelPosBase(TransformerDecoderBase):
    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = TransformerDecoderLayerRelPosBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
