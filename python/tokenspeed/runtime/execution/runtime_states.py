# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Runtime state tensors shared by the model executor."""

from __future__ import annotations

import torch

from tokenspeed.runtime.execution.request_token_history import RequestTokenHistoryView


class RuntimeStates:
    """Own runtime state tensors keyed by request-pool index."""

    def __init__(
        self,
        req_pool_size: int,
        vocab_size: int,
        output_length: int,
        device: str = "cuda",
        request_token_history_capacity: int = 0,
    ):
        if request_token_history_capacity < 0:
            raise ValueError("request token history capacity must be non-negative")

        self.device = device
        self.vocab_size = vocab_size
        self._request_pool_size = req_pool_size
        self.ngram_accepted_tokens: torch.Tensor | None = None
        self.ngram_needs_seed: torch.Tensor | None = None
        self.ngram_request_ids: list[str | None] = []

        self.valid_cache_lengths = torch.zeros(
            req_pool_size + 1, dtype=torch.int32, device=device
        )
        self.request_token_history_ids = (
            torch.zeros(
                (req_pool_size + 1, request_token_history_capacity),
                dtype=torch.int32,
                device=device,
            )
            if request_token_history_capacity
            else None
        )
        # Resolve input ids from here when overlap scheduling.
        self.future_input_map = torch.empty(
            (req_pool_size + 1, output_length), dtype=torch.int32, device=device
        )
        self.remote_spec_candidate_ready = torch.zeros(
            req_pool_size + 1, dtype=torch.bool, device=device
        )

    def init_ngram_state(self, context_len: int) -> None:
        """Allocate a bounded, newest-first accepted input tail per pool slot.

        ``context_len`` is zero without Engram, three for V4.1. This executor
        state follows valid_cache_lengths, not sampled output or model KV. It
        is reseeded from host snapshots on admission/recovery, never transferred
        or prefix-matched as a backend cache. Returns None.
        """
        if context_len == 0:
            return
        pool_size = self.valid_cache_lengths.shape[0]
        self.ngram_accepted_tokens = torch.full(
            (pool_size, context_len), -1, dtype=torch.int64, device=self.device
        )
        self.ngram_needs_seed = torch.ones(
            pool_size, dtype=torch.bool, device=self.device
        )
        self.ngram_request_ids = [None] * pool_size

    @property
    def has_request_token_history(self) -> bool:
        """Whether this runtime owns request-persistent token histories."""
        return self.request_token_history_ids is not None

    def request_token_history_view(
        self,
        *,
        req_pool_indices: torch.Tensor,
        input_start_offsets: torch.Tensor,
        active_request_mask: torch.Tensor,
    ) -> RequestTokenHistoryView | None:
        """Combine persistent history with the current packed-batch layout."""
        if self.request_token_history_ids is None:
            return None
        return RequestTokenHistoryView(
            history_token_ids=self.request_token_history_ids,
            committed_lengths=self.valid_cache_lengths,
            req_pool_indices=req_pool_indices,
            input_start_offsets=input_start_offsets,
            active_request_mask=active_request_mask,
        )

    def update_valid_cache_length(
        self, req_pool_indices: torch.Tensor, increment_lengths: torch.Tensor
    ) -> None:
        self.valid_cache_lengths.index_add_(0, req_pool_indices, increment_lengths)

    def reset_states(
        self,
        extend_request_pool_indices: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
    ) -> None:
        self.valid_cache_lengths[extend_request_pool_indices] = extend_prefix_lens
        # Scalar indexed assignment stages a CPU tensor and synchronizes CUDA.
        # Keep the reset ordered on the execution stream without a host wait.
        self.remote_spec_candidate_ready.index_fill_(
            0, extend_request_pool_indices, False
        )
        if self.ngram_accepted_tokens is not None:
            assert self.ngram_needs_seed is not None
            self.ngram_accepted_tokens.index_fill_(0, extend_request_pool_indices, -1)
            self.ngram_needs_seed.index_fill_(0, extend_request_pool_indices, True)

    def seed_request_token_history(
        self,
        *,
        req_pool_indices,
        prefix_lengths,
        request_token_ids,
    ) -> None:
        """Restore the history required at each request's prefix boundary.

        Args:
            req_pool_indices: Request-pool slots to restore.
            prefix_lengths: Absolute prefix boundary for each request.
            request_token_ids: Full token prefixes ending at the corresponding
                prefix boundary.
        """
        history = self.request_token_history_ids
        if history is None:
            raise RuntimeError("request token history is not enabled")
        if not (len(req_pool_indices) == len(prefix_lengths) == len(request_token_ids)):
            raise ValueError(
                "request token history seed inputs must have matching lengths"
            )

        capacity = history.shape[1]
        for row, prefix_length, token_ids in zip(
            req_pool_indices, prefix_lengths, request_token_ids
        ):
            row = int(row)
            prefix_length = int(prefix_length)
            seed_tokens = tuple(int(token_id) for token_id in token_ids)
            if not 0 <= row < self._request_pool_size:
                raise ValueError(f"request token history slot {row} is out of range")
            if not 0 <= prefix_length <= capacity:
                raise ValueError(
                    f"request token history prefix length {prefix_length} "
                    f"exceeds capacity {capacity}"
                )
            if len(seed_tokens) != prefix_length:
                raise ValueError(
                    "request token history seed has "
                    f"{len(seed_tokens)} tokens but must cover prefix length "
                    f"{prefix_length}"
                )
            if seed_tokens:
                history[row, :prefix_length].copy_(
                    torch.as_tensor(seed_tokens, dtype=torch.int32, device=self.device)
                )

    def write_remote_spec_candidate_ids(
        self, req_pool_idx: int, candidate_ids: list[int]
    ) -> None:
        width = self.future_input_map.shape[1]
        if len(candidate_ids) != width:
            raise RuntimeError(
                f"remote spec candidate width mismatch: got {len(candidate_ids)}, expected {width}"
            )
        ids = torch.tensor(
            candidate_ids,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        self.future_input_map[req_pool_idx, :width] = ids
        self.remote_spec_candidate_ready[req_pool_idx] = True
