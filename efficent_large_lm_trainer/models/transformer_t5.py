# Copyright (c) Microsoft. All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models import register_model_architecture
from fairseq.utils import safe_getattr

from .transformer_rel_pos import base_architecture


def transformer_t5_common(args):
    args.encoder_normalize_before = safe_getattr(args, "encoder_normalize_before", False)
    args.decoder_normalize_before = safe_getattr(args, "decoder_normalize_before", False)
    args.encoder_learned_pos = safe_getattr(args, "encoder_learned_pos", True)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.0)
    args.activation_fn = safe_getattr(args, "activation_fn", "relu")
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.share_decoder_input_output_embed = safe_getattr(args, "share_decoder_input_output_embed", True)
    args.share_all_embeddings = safe_getattr(args, "share_all_embeddings", True)
    args.no_token_positional_embeddings = safe_getattr(args, "no_token_positional_embeddings", False)
    args.no_scale_embedding = safe_getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", True)
    args.apply_t5_init = safe_getattr(args, "apply_t5_init", True)
    return base_architecture(args)


@register_model_architecture("transformer_rel_pos", "transformer_t5_base")
def transformer_t5_base(args):
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 12)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 12)
    args.decoder_output_dim = safe_getattr(args, "decoder_output_dim", 768)
    args.decoder_input_dim = safe_getattr(args, "decoder_input_dim", 768)
    return transformer_t5_common(args)


@register_model_architecture("transformer_rel_pos", "transformer_t5_base_rel_pos")
def transformer_t5_base_rel_pos(args):
    args.encoder_rel_pos = safe_getattr(args, "encoder_rel_pos", True)
    args.encoder_rel_pos_bins = safe_getattr(args, "encoder_rel_pos_bins", 32)
    args.encoder_rel_pos_max_dist = safe_getattr(args, "encoder_rel_pos_max_dist", 128)
    args.decoder_rel_pos = safe_getattr(args, "decoder_rel_pos", True)
    args.decoder_rel_pos_bins = safe_getattr(args, "decoder_rel_pos_bins", 32)
    args.decoder_rel_pos_max_dist = safe_getattr(args, "decoder_rel_pos_max_dist", 128)
    return transformer_t5_base(args)


@register_model_architecture("transformer_rel_pos", "transformer_t5_base_rel_pos_encoder")
def transformer_t5_base_rel_pos_encoder(args):
    args.encoder_rel_pos = safe_getattr(args, "encoder_rel_pos", True)
    args.encoder_rel_pos_bins = safe_getattr(args, "encoder_rel_pos_bins", 32)
    args.encoder_rel_pos_max_dist = safe_getattr(args, "encoder_rel_pos_max_dist", 128)
    return transformer_t5_base(args)
