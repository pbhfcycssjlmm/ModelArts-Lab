"""Backport EAGLE multimodal encoder-cache handling to vLLM v0.23.0.

Target and necessity
====================

This patch changes the upstream GPU vLLM v0.23.0 package used by the NPU
deployment. It does not modify the external ``vllm-ascend`` repository.

It backports the relevant vLLM v0.24.0 changes from upstream commit
``3e6529cc0e`` ("Fix EAGLE drafter multimodal encoder cache misses"):

* ``Scheduler._free_encoder_inputs`` keeps a multimodal encoder-cache
  reference until the EAGLE/MTP drafter's one-token look-ahead has passed it.
* ``GPUModelRunner._gather_mm_embeddings`` treats an encoder-cache miss
  strictly beyond the target model's scheduled range as a drafter-only read,
  so that position uses the normal token embedding. A miss inside the target
  range remains an error.

Why this is needed for NPU
==========================

For the investigated Qwen3.5 MTP request, the NPU ``NPUModelRunner`` uses the
upstream ``GPUModelRunner`` gather path when ``PCP=1``. The service under
investigation runs with ``DP=1, PP=1, PCP=1, TP=1``, so patching the upstream
base class covers its active path without changing ``vllm-ascend``.

This patch is necessary for correct EAGLE encoder-cache lifetime, but it is
not sufficient to avoid the two-video v0.23.0 deadlock by itself. It must be
used together with ``patch_qwen3vl_multivideo.py``, which restores Qwen3VL's
correct visual boundary tokens between adjacent videos.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.multimodal.utils import get_mm_features_in_window

logger = init_logger(__name__)
_PATCH_APPLIED = False


def _patch_scheduler() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    original = Scheduler._free_encoder_inputs
    if getattr(original, "_ascend_vllm_eagle_encoder_cache_patch", False):
        return

    def free_encoder_inputs_with_eagle_lookahead(self: Scheduler, request: Any) -> None:
        cached_input_ids = self.encoder_cache_manager.get_cached_input_ids(request)
        if not cached_input_ids:
            return

        # Match 3e6529cc0e: an EAGLE drafter reads one token beyond the target
        # range, so that reference must survive through the look-ahead.
        spec_lookahead = 1 if self.use_eagle else 0
        confirmed_tokens = request.num_computed_tokens - getattr(request, "num_output_placeholders", 0)

        for input_id in list(cached_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if (
                self.is_encoder_decoder
                and request.num_computed_tokens > 0
                or start_pos + num_tokens + spec_lookahead <= confirmed_tokens
            ):
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    free_encoder_inputs_with_eagle_lookahead._ascend_vllm_eagle_encoder_cache_patch = True
    Scheduler._free_encoder_inputs = free_encoder_inputs_with_eagle_lookahead


def _patch_gpu_model_runner() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner._gather_mm_embeddings
    if getattr(original, "_ascend_vllm_eagle_encoder_cache_patch", False):
        return

    def gather_mm_embeddings_with_eagle_boundary_fallback(
        self: GPUModelRunner,
        scheduler_output: Any,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        mm_embeds: list[torch.Tensor] = []
        is_mm_embed = torch.zeros(total_num_scheduled_tokens, dtype=torch.bool, device="cpu")

        req_start_idx = 0
        should_sync_mrope_positions = False
        should_sync_xdrope_positions = False

        for req_id in self.input_batch.req_ids:
            mm_embeds_req: list[torch.Tensor] = []
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens + shift_computed_tokens
            target_end = req_state.num_computed_tokens + num_scheduled_tokens

            mm_features = req_state.mm_features
            lo, hi = get_mm_features_in_window(
                mm_features,
                start=num_computed_tokens,
                end=num_computed_tokens + num_scheduled_tokens,
            )
            for input_id in range(lo, hi):
                mm_feature = mm_features[input_id]
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens,
                )
                assert start_idx < end_idx
                curr_embeds_start, curr_embeds_end = pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                if curr_embeds_start == curr_embeds_end:
                    continue

                encoder_output = self.encoder_cache.get(mm_feature.identifier)
                if encoder_output is None:
                    # Only the EAGLE +1 look-ahead can see a feature that
                    # starts beyond the target range. It is intentionally not
                    # consumed by the target model in this iteration.
                    if shift_computed_tokens and start_pos >= target_end:
                        continue
                    raise RuntimeError(f"Encoder cache miss for {mm_feature.identifier}.")

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]
                    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
                else:
                    is_embed = None
                    mm_embeds_item = encoder_output[start_idx:end_idx]

                req_start_pos = req_start_idx + start_pos - num_computed_tokens
                if is_embed is None:
                    is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = True
                else:
                    is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] |= is_embed
                mm_embeds_req.append(mm_embeds_item)

            if self.is_multimodal_pruning_enabled and self.uses_mrope:
                assert req_state.mrope_positions is not None
                should_sync_mrope_positions = True
                mm_embeds_req, new_mrope_positions, new_delta = self.model.recompute_mrope_positions(
                    input_ids=req_state.prompt_token_ids,
                    multimodal_embeddings=mm_embeds_req,
                    mrope_positions=req_state.mrope_positions,
                    num_computed_tokens=req_state.num_computed_tokens,
                )
                req_state.mrope_positions.copy_(new_mrope_positions)
                req_state.mrope_position_delta = new_delta

            mm_embeds.extend(mm_embeds_req)
            req_start_idx += num_scheduled_tokens

        if should_sync_mrope_positions:
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.copy_to_gpu(total_num_scheduled_tokens)
        if should_sync_xdrope_positions:
            self._calc_xdrope_positions(scheduler_output)
            self.xdrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        return mm_embeds, is_mm_embed

    gather_mm_embeddings_with_eagle_boundary_fallback._ascend_vllm_eagle_encoder_cache_patch = True
    GPUModelRunner._gather_mm_embeddings = gather_mm_embeddings_with_eagle_boundary_fallback


def apply_patch() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    _patch_scheduler()
    _patch_gpu_model_runner()
    logger.info("Applied EAGLE multimodal encoder-cache backport for vLLM v0.23.0")
    _PATCH_APPLIED = True


apply_patch()
