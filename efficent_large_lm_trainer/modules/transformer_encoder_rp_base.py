from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer import TransformerEncoderBase
from fairseq.modules.checkpoint_activations import checkpoint_wrapper

from .transformer_layer_rp import TransformerEncoderLayerRp


class TransformerEncoderRpBase(TransformerEncoderBase):
    def build_encoder_layer(self, cfg):
        layer = TransformerEncoderLayerRp(
            cfg, return_fc=self.return_fc
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
