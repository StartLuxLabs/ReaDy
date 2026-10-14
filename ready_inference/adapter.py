"""Attach the reader to the final draft layer and refresh graph-stable state storage."""
import os
from pathlib import Path
import torch
from safetensors.torch import load_file
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from .reader import StateReader


def install():
    if getattr(DFlashDraftModel, '_ready_installed', False):
        return
    original_init = DFlashDraftModel.__init__
    original_propose = DraftBlockProposer.propose

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        param = next(self.parameters())
        reader = StateReader().to(device=param.device, dtype=param.dtype)
        reader.load_state_dict(load_file(str(Path(os.environ['READY_CHECKPOINT']) / 'reader/reader.safetensors')), strict=True)
        reader.eval().requires_grad_(False)
        self.add_module('state_reader', reader)
        self._ready_backend = os.environ.get('READY_READER_BACKEND', 'triton')
        if self._ready_backend not in ('reference', 'triton'):
            raise ValueError(f"Unsupported reader backend: {self._ready_backend!r}; choose 'triton' or 'reference'.")
        if self._ready_backend == 'triton':
            from .kernels import reader_forward, gather_states
            self._ready_forward = reader_forward
            self._ready_gather = gather_states
        self.register_buffer('_ready_state', torch.zeros(4, 2, 48, 128, 128,
                             device=param.device, dtype=torch.float32), persistent=False)

        def inject(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            assert hidden.ndim == 2 and hidden.shape[0] % 7 == 0
            batch = hidden.shape[0] // 7
            assert batch <= 4
            shaped = hidden.reshape(batch, 7, -1)
            result = (self._ready_forward(self.state_reader, shaped, self._ready_state[:batch])
                      if self._ready_backend == 'triton'
                      else self.state_reader(shaped, self._ready_state[:batch]))
            result = result.reshape_as(hidden)
            return (result, *output[1:]) if isinstance(output, tuple) else result

        self.layers[4].post_attention_layernorm.register_forward_hook(inject)

    def propose(self, **kwargs):
        batch, size = kwargs['batch'], kwargs['bs']
        pool = batch.req_to_token_pool
        indices = pool.translate_mamba_indices(pool.get_mamba_indices(batch.req_pool_indices))
        assert len(indices) == size and size <= 4
        model = self.draft_model
        if model._ready_backend == 'triton':
            model._ready_gather(pool.mamba2_layer_cache(30).temporal,
                                pool.mamba2_layer_cache(62).temporal,
                                indices, model._ready_state[:size])
        else:
            for slot, layer in enumerate((30, 62)):
                model._ready_state[:size, slot].copy_(pool.mamba2_layer_cache(layer).temporal[indices])
        return original_propose(self, **kwargs)

    DFlashDraftModel.__init__ = initialize
    DraftBlockProposer.propose = propose
    DFlashDraftModel._ready_installed = True
