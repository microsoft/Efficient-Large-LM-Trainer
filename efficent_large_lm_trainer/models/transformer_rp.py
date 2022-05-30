import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel, register_model, register_model_architecture
from fairseq.utils import safe_getattr

from .transformer_rp_config import (
    TransformerRpConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from ..modules import (
    RelativePositionalEmbedding,
    TransformerEncoderRpBase,
    TransformerDecoderRpBase,
)


def init_t5_params(module):
    def normal_(data, scale):
        data.copy_(data.cpu().normal_(mean=0.0, std=scale).to(data.device))

    def uniform_(data, scale):
        data.copy_(data.cpu().uniform_(-math.sqrt(3.0) * scale, math.sqrt(3.0) * scale).to(data.device))

    if isinstance(module, nn.Linear):
        fan_out = module.weight.data.size(0)
        fan_in = module.weight.data.size(0)
        normal_(module.weight.data, (fan_in + fan_out) ** -0.5)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        embed_dim = module.weight.data.size(1)
        uniform_(module.weight.data, embed_dim ** -0.5 / math.sqrt(2))
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, RelativePositionalEmbedding):
        module.rel_attn_bias.weight.data.zero_()


@register_model("transformer_rp")
class TransformerRpModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.
    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder
    The Transformer model provides the following named architectures and
    command-line arguments:
    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        cfg = TransformerRpConfig.from_namespace(args)
        super().__init__(encoder, decoder)
        self.args = args
        self.cfg = cfg
        self.supports_align_args = True

        if cfg.apply_t5_init:
            self.apply(init_t5_params)

    @classmethod
    def add_args(cls, parser):
        gen_parser_from_dataclass(
            parser, TransformerRpConfig(), delete_default=True, with_prefix=""
        )

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
        cfg = TransformerRpConfig.from_namespace(args)

        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        if not cfg.share_all_embeddings:
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=cfg.min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=cfg.min_params_to_wrap)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        cfg = TransformerRpConfig.from_namespace(args)

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        cfg = TransformerRpConfig.from_namespace(args)
        return TransformerEncoderRpBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        cfg = TransformerRpConfig.from_namespace(args)
        return TransformerDecoderRpBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


@register_model_architecture("transformer_rp", "transformer_rp")
def base_architecture(args):
    args.encoder_embed_path = safe_getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = safe_getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = safe_getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = safe_getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = safe_getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = safe_getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = safe_getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.0)
    args.activation_fn = safe_getattr(args, "activation_fn", "relu")
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.share_decoder_input_output_embed = safe_getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = safe_getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = safe_getattr(
        args, "no_token_positional_embeddings", False
    )
    args.no_cross_attention = safe_getattr(args, "no_cross_attention", False)
    args.cross_self_attention = safe_getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = safe_getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = safe_getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = safe_getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = safe_getattr(args, "checkpoint_activations", False)
    args.offload_activations = safe_getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = safe_getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = safe_getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = safe_getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = safe_getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = safe_getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = safe_getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = safe_getattr(args, "quant_noise_scalar", 0)

    args.encoder_rel_pos = safe_getattr(args, "encoder_rel_pos", False)
    args.encoder_rp_bins = safe_getattr(args, "encoder_rp_bins", 32)
    args.encoder_rp_max_dist = safe_getattr(args, "encoder_rp_max_dist", 128)
    args.decoder_rel_pos = safe_getattr(args, "decoder_rel_pos", False)
    args.decoder_rp_bins = safe_getattr(args, "decoder_rp_bins", 32)
    args.decoder_rp_max_dist = safe_getattr(args, "decoder_rp_max_dist", 128)
