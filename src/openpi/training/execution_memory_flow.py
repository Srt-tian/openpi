"""Experimental causal execution-memory adapter; no backbone parameters owned.

This is not ActMem's Mamba/PAE reproduction. A GRU encodes past normalized
state8/action7/state-change8 transitions. Four memory tokens condition a small
flow adapter using frozen native action hidden features. Deployment modifies
only physical7 velocity in the first four of ten denoising steps.
"""
from __future__ import annotations

import torch
from torch import nn

SCHEMA = 'pi05-execution-memory-flow.v1'


def validate_history(history, mask):
    if history.ndim != 3 or history.shape[-1] != 23 or history.shape[1] > 520:
        raise ValueError('history must be [batch,past<=520,23]')
    if mask.shape != history.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('history needs a matching boolean prefix mask')
    if mask.device != history.device or not torch.isfinite(history).all():
        raise ValueError('history must be finite and on the mask device')
    if history.shape[1] > 1 and ((~mask[:, :-1]) & mask[:, 1:]).any():
        raise ValueError('valid history must precede padding')


class ExecutionMemoryEncoder(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.cell = nn.GRUCell(23, hidden)

    def forward(self, history, mask):
        validate_history(history, mask)
        h = history.new_zeros((len(history), self.hidden))
        for step in range(history.shape[1]):
            proposal = self.cell(history[:, step], h)
            h = torch.where(mask[:, step, None], proposal, h)
        return h

    def advance(self, transition, h):
        if transition.ndim != 2 or transition.shape[1] != 23 or h.shape != (len(transition), self.hidden):
            raise ValueError('one observed transition23 per recurrent state required')
        if not torch.isfinite(transition).all() or not torch.isfinite(h).all():
            raise ValueError('recurrent inputs must be finite')
        return self.cell(transition, h)


class MemoryFlowAdapter(nn.Module):
    def __init__(self, feature_width=1024, width=256, memory_hidden=128):
        super().__init__()
        self.feature_width = feature_width
        self.memory = ExecutionMemoryEncoder(memory_hidden)
        self.memory_tokens = nn.Linear(memory_hidden, 4 * width)
        self.current = nn.Linear(feature_width, width)
        self.time = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.position = nn.Parameter(torch.zeros(1, 10, width))
        layer = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0,
                                          activation='gelu', batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.output = nn.Linear(width, 7)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def from_memory(self, frozen_hidden, time, memory_state):
        if frozen_hidden.ndim != 3 or frozen_hidden.shape[1:] != (10, self.feature_width):
            raise ValueError('native hidden must be [batch,10,feature_width]')
        if time.shape != (len(frozen_hidden),) or memory_state.shape != (len(frozen_hidden), self.memory.hidden):
            raise ValueError('time/memory batch mismatch')
        if not all(torch.isfinite(x).all() for x in (frozen_hidden, time, memory_state)):
            raise ValueError('nonfinite flow adapter input')
        if ((time < 0) | (time > 1)).any():
            raise ValueError('flow time must be in [0,1]')
        # Backbone tensors are read-only conditioning, never optimization targets.
        tokens = self.current(frozen_hidden.detach().to(self.current.weight)) + self.position + self.time(time.detach().to(self.current.weight)[:, None])[:, None]
        memory = self.memory_tokens(memory_state.to(self.memory_tokens.weight)).reshape(len(tokens), 4, -1)
        fused = self.fusion(torch.cat([memory, tokens], dim=1))[:, 4:]
        return .5 * torch.tanh(self.output(fused))

    def forward(self, frozen_hidden, time, history, mask):
        return self.from_memory(frozen_hidden, time, self.memory(history, mask))


def apply_early_velocity(base_velocity, delta7, *, denoise_index, enabled=True):
    if not enabled:
        return base_velocity
    if type(denoise_index) is not int or not 0 <= denoise_index < 10:
        raise ValueError('denoising protocol requires exactly ten steps')
    if denoise_index >= 4:
        return base_velocity
    if base_velocity.ndim != 3 or base_velocity.shape[1:] != (10, 32):
        raise ValueError('base velocity must be [batch,10,32]')
    if delta7.shape != (*base_velocity.shape[:2], 7):
        raise ValueError('physical correction shape mismatch')
    if not torch.isfinite(base_velocity).all() or not torch.isfinite(delta7).all():
        raise ValueError('nonfinite velocity')
    if (delta7.abs() > .500001).any():
        raise ValueError('correction exceeds normalized-velocity bound')
    result = base_velocity.clone()
    result[..., :7] = result[..., :7] + delta7.to(base_velocity)
    return result


def memory_flow_loss(adapter, *, frozen_hidden, time, history, history_mask,
                     base_velocity, target_velocity7):
    # Cache/labels must represent the actual early sampler times 1,.9,.8,.7.
    allowed = time.new_tensor([1., .9, .8, .7])
    if not ((time[:, None] - allowed).abs().min(dim=1).values < 1e-5).all():
        raise ValueError('memory pilot trains only the four early sampler times')
    if base_velocity.shape != (*frozen_hidden.shape[:2], 32) or target_velocity7.shape != (*frozen_hidden.shape[:2], 7):
        raise ValueError('flow target shape mismatch')
    if not torch.isfinite(base_velocity).all() or not torch.isfinite(target_velocity7).all():
        raise ValueError('flow labels must be finite')
    base = base_velocity[..., :7].detach()
    target = target_velocity7.detach()
    delta = adapter(frozen_hidden, time, history, history_mask)
    # All H10 positions are supervised, including the official repeat-last tail.
    error = ((base + delta - target) ** 2).mean(dim=(1, 2))
    base_error = ((base - target) ** 2).mean(dim=(1, 2))
    regret = torch.relu(error - base_error)
    shrinkage = (delta ** 2).mean()
    loss = error.mean() + .1 * regret.mean() + 1e-4 * shrinkage
    return loss, {'flow_mse': error.mean().detach(), 'base_mse': base_error.mean().detach(),
                  'paired_regret': regret.mean().detach(), 'correction_mse': shrinkage.detach()}
