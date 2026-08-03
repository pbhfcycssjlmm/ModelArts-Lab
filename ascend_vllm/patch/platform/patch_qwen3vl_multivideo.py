"""Backport Qwen3VL multi-video placeholder processing to vLLM v0.23.0.

Target and necessity
====================

This is a runtime compatibility patch for the upstream GPU vLLM v0.23.0
package used by the NPU service. No file in the external ``vllm-ascend``
repository is modified.

It backports the Qwen3VL changes from upstream commits:

* ``d272418f459a``: construct each video replacement as token IDs and replace
  placeholders after the combined Hugging Face processor pass.
* ``dd944845777b``: for Transformers >= 5.10 processors, replace only the
  bare ``<|video_pad|>`` token rather than the whole visual triplet.

With Transformers 5.14.1, replacing only the bare video token preserves the
existing ``<|vision_end|><|vision_start|>`` tokens between adjacent videos.
For the reproduced request this changes v0.23.0 feature ranges from:

    video1 [3, 11662), video2 [11662, 23321)

to:

    video1 [4, 11663), video2 [11665, 23324)

The two-token visual boundary gives EAGLE cache release a reachable handoff
point during chunked prefill. This patch must be paired with
``patch_eagle_mm_encoder_cache.py`` for the v0.23.0 EAGLE lifecycle backport.

Affected upstream functions
===========================

``vllm.model_executor.models.qwen3_vl``:

* ``_replace_video_token_placeholders``
* ``Qwen3VLMultiModalProcessor._call_hf_processor``
* ``Qwen3VLMultiModalProcessor._get_prompt_updates``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers import BatchFeature, ProcessorMixin
from vllm.logger import init_logger
from vllm.model_executor.models import qwen3_vl as qwen3_vl_module

logger = init_logger(__name__)
_PATCH_APPLIED = False


def _replace_video_token_placeholders(
    prompt_ids: list[int],
    target: list[int],
    replacements: list[list[int]],
) -> list[int]:
    """Replace each video placeholder with its prepared token sequence."""
    result: list[int] = []
    replacement_index = 0
    prompt_index = 0
    target_length = len(target)

    while prompt_index < len(prompt_ids):
        if prompt_ids[prompt_index : prompt_index + target_length] == target:
            result.extend(replacements[replacement_index])
            replacement_index += 1
            prompt_index += target_length
        else:
            result.append(prompt_ids[prompt_index])
            prompt_index += 1

    assert replacement_index == len(replacements), (
        f"Found {replacement_index} video placeholders but expected {len(replacements)}"
    )
    return result


def _expands_only_video_token(hf_processor: ProcessorMixin) -> bool:
    """Detect Transformers >= 5.10 Qwen3VL bare-video-token expansion."""
    mixin_impl = getattr(ProcessorMixin, "replace_video_token", None)
    processor_impl = getattr(type(hf_processor), "replace_video_token", None)
    return processor_impl is not None and processor_impl is not mixin_impl


def _call_hf_processor(
    self: Any,
    prompt: str,
    mm_data: Mapping[str, object],
    mm_kwargs: Mapping[str, object],
    tok_kwargs: Mapping[str, object],
) -> BatchFeature:
    mm_data = dict(mm_data)
    video_input_ids: list[list[int]] = []

    if videos := mm_data.pop("videos", []):
        video_grid_thw_list = []
        pixel_values_videos_list = []
        timestamps_per_video = []

        hf_config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()
        merge_size = hf_config.vision_config.spatial_merge_size
        video_pruning_rate = self.info.ctx.get_mm_config().video_pruning_rate

        for video_array, metadata in videos:
            video_mm_kwargs = dict(mm_kwargs)
            if "do_sample_frames" not in video_mm_kwargs:
                video_mm_kwargs["do_sample_frames"] = metadata.get("do_sample_frames", False)

            metadata = qwen3_vl_module.VideoMetadata(
                **{key: metadata[key] for key in metadata if key != "do_sample_frames"}
            )
            timestamps = self.info._get_video_second_idx(
                metadata=metadata,
                do_sample_frames=video_mm_kwargs["do_sample_frames"],
                sampled_fps=video_mm_kwargs.get("fps"),
                sampled_num_frames=video_mm_kwargs.get("num_frames"),
            )
            timestamps_per_video.append(timestamps)

            video_mm_data = {
                "videos": [[video_array]],
                "video_metadata": [[metadata]],
            }
            if "num_frames" in video_mm_kwargs and "fps" not in video_mm_kwargs:
                video_mm_kwargs["fps"] = None

            video_outputs = super(qwen3_vl_module.Qwen3VLMultiModalProcessor, self)._call_hf_processor(
                prompt="<|vision_start|><|video_pad|><|vision_end|>",
                mm_data=video_mm_data,
                mm_kwargs=video_mm_kwargs,
                tok_kwargs=tok_kwargs,
            )
            video_outputs.pop("input_ids", None)

            video_grid_thw = video_outputs["video_grid_thw"]
            num_frames = int(video_grid_thw[0, 0])
            tokens_per_frame_base = int(video_grid_thw[0, 1:].prod()) // (merge_size**2)
            if video_pruning_rate is not None and video_pruning_rate > 0.0:
                num_tokens = qwen3_vl_module.compute_retained_tokens_count(
                    tokens_per_frame=tokens_per_frame_base,
                    num_frames=num_frames,
                    q=video_pruning_rate,
                )
                tokens_per_frame = [num_tokens] + [0] * (num_frames - 1)
                select_token_id = False
            else:
                tokens_per_frame = [tokens_per_frame_base] * num_frames
                select_token_id = True

            video_replacement = qwen3_vl_module.Qwen3VLMultiModalProcessor.get_video_repl(
                tokens_per_frame=tokens_per_frame,
                timestamps=timestamps,
                tokenizer=tokenizer,
                vision_start_token_id=hf_config.vision_start_token_id,
                vision_end_token_id=hf_config.vision_end_token_id,
                video_token_id=hf_config.video_token_id,
                select_token_id=select_token_id,
            )
            video_input_ids.append(list(video_replacement.full))
            video_grid_thw_list.append(video_outputs["video_grid_thw"])
            pixel_values_videos_list.append(video_outputs["pixel_values_videos"])

        video_outputs = {
            "pixel_values_videos": torch.cat(pixel_values_videos_list),
            "video_grid_thw": torch.cat(video_grid_thw_list),
            "timestamps": timestamps_per_video,
        }
    else:
        video_outputs = {}

    processed_outputs = super(qwen3_vl_module.Qwen3VLMultiModalProcessor, self)._call_hf_processor(
        prompt=prompt,
        mm_data=mm_data,
        mm_kwargs=mm_kwargs,
        tok_kwargs=tok_kwargs,
    )
    if video_input_ids:
        hf_config = self.info.get_hf_config()
        if _expands_only_video_token(self.info.get_hf_processor()):
            video_target = [hf_config.video_token_id]
        else:
            video_target = [
                hf_config.vision_start_token_id,
                hf_config.video_token_id,
                hf_config.vision_end_token_id,
            ]

        input_ids = processed_outputs.pop("input_ids")
        if not isinstance(input_ids, list):
            input_ids = input_ids.tolist()
        (prompt_ids,) = input_ids
        processed_outputs["input_ids"] = [
            _replace_video_token_placeholders(
                prompt_ids,
                video_target,
                video_input_ids,
            )
        ]

    return BatchFeature(dict(processed_outputs, **video_outputs))


def _get_prompt_updates(
    self: Any,
    mm_items: Any,
    hf_processor_mm_kwargs: Mapping[str, Any],
    out_mm_kwargs: Any,
) -> Sequence[Any]:
    hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
    image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
    tokenizer = self.info.get_tokenizer()
    hf_config = self.info.get_hf_config()
    merge_length = image_processor.merge_size**2

    def get_image_replacement(item_index: int) -> list[int]:
        out_item = out_mm_kwargs["image"][item_index]
        grid_thw = out_item["image_grid_thw"].data
        assert isinstance(grid_thw, torch.Tensor)
        return [hf_processor.image_token_id] * (int(grid_thw.prod()) // merge_length)

    def get_video_replacement(item_index: int) -> Any:
        out_item = out_mm_kwargs["video"][item_index]
        grid_thw = out_item["video_grid_thw"].data
        assert isinstance(grid_thw, torch.Tensor)

        sampled_fps = hf_processor_mm_kwargs.get("fps")
        if qwen3_vl_module.is_list_of(sampled_fps, float):
            sampled_fps = sampled_fps[item_index]

        timestamps = out_item["timestamps"].data
        assert len(timestamps) == grid_thw[0], (
            f"The timestamps length({len(timestamps)}) should be equal video length ({grid_thw[0]})."
        )

        num_frames = int(grid_thw[0])
        tokens_per_frame_base = int(grid_thw[1:].prod()) // merge_length
        video_pruning_rate = self.info.ctx.get_mm_config().video_pruning_rate
        if video_pruning_rate is not None and video_pruning_rate > 0.0:
            num_tokens = qwen3_vl_module.compute_retained_tokens_count(
                tokens_per_frame=tokens_per_frame_base,
                num_frames=num_frames,
                q=video_pruning_rate,
            )
            tokens_per_frame = [num_tokens] + [0] * (num_frames - 1)
            select_token_id = False
        else:
            tokens_per_frame = [tokens_per_frame_base] * num_frames
            select_token_id = True

        return qwen3_vl_module.Qwen3VLMultiModalProcessor.get_video_repl(
            tokens_per_frame=tokens_per_frame,
            timestamps=timestamps,
            tokenizer=tokenizer,
            vision_start_token_id=hf_config.vision_start_token_id,
            vision_end_token_id=hf_config.vision_end_token_id,
            video_token_id=hf_config.video_token_id,
            select_token_id=select_token_id,
        )

    if _expands_only_video_token(hf_processor):
        video_target = hf_processor.video_token
    else:
        video_target = "<|vision_start|><|video_pad|><|vision_end|>"

    return [
        qwen3_vl_module.PromptReplacement(
            modality="image",
            target=hf_processor.image_token,
            replacement=get_image_replacement,
        ),
        qwen3_vl_module.PromptReplacement(
            modality="video",
            target=video_target,
            replacement=get_video_replacement,
        ),
    ]


def apply_patch() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    processor_cls = qwen3_vl_module.Qwen3VLMultiModalProcessor
    if getattr(processor_cls, "_ascend_vllm_qwen3vl_multivideo_patch", False):
        return

    qwen3_vl_module._replace_video_token_placeholders = _replace_video_token_placeholders
    processor_cls._expands_only_video_token = staticmethod(_expands_only_video_token)
    processor_cls._call_hf_processor = _call_hf_processor
    processor_cls._get_prompt_updates = _get_prompt_updates
    processor_cls._ascend_vllm_qwen3vl_multivideo_patch = True
    logger.info("Applied Qwen3VL multi-video placeholder backport for vLLM v0.23.0")
    _PATCH_APPLIED = True


apply_patch()
