"""In-process Hugging Face transformers backend (legacy fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig

from inference.backends.base import InferenceBackend
from inference.model_registry import detect_family
from patch_utils import json_from_text, normalize_and_filter
from prompts import PROMPT_PAIR_DIRECT, SYSTEM_INSTRUCTION

try:
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
except ImportError:
    Qwen3VLForConditionalGeneration = None
    Qwen3VLProcessor = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


@dataclass
class TransformersBackendConfig:
    model_id: str
    max_new_tokens: int
    top_p: float
    seed: int
    load_4bit: bool
    cache_dir: Path
    use_flash_attn2: bool


class TransformersBackend(InferenceBackend):
    def __init__(self, cfg: TransformersBackendConfig):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for transformers backend.")
        self.cfg = cfg
        self._family = detect_family(cfg.model_id)
        torch.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        self.processor = self._load_processor()
        self.model = self._load_model()

    @property
    def family(self) -> str:
        return self._family

    def _load_processor(self):
        if self._family == "qwen35":
            return AutoProcessor.from_pretrained(
                self.cfg.model_id, trust_remote_code=True, cache_dir=self.cfg.cache_dir
            )
        if self._family == "qwen3_vl":
            if Qwen3VLProcessor is None:
                raise RuntimeError("Qwen3VLProcessor not available.")
            if self.cfg.model_id == "Qwen/Qwen3-VL-32B-Instruct":
                return AutoProcessor.from_pretrained(
                    self.cfg.model_id, trust_remote_code=True, cache_dir=self.cfg.cache_dir
                )
            return Qwen3VLProcessor.from_pretrained(
                self.cfg.model_id, trust_remote_code=True, cache_dir=self.cfg.cache_dir
            )
        return AutoProcessor.from_pretrained(
            self.cfg.model_id, trust_remote_code=True, cache_dir=self.cfg.cache_dir
        )

    def _load_model(self):
        model_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "trust_remote_code": True,
        }
        if self.cfg.load_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        if self.cfg.use_flash_attn2:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            model_kwargs["attn_implementation"] = (
                "eager" if self._family == "qwen25_vl" else "sdpa"
            )
        if self._family == "qwen35":
            if AutoModelForImageTextToText is None:
                raise RuntimeError("AutoModelForImageTextToText not available.")
            ModelClass = AutoModelForImageTextToText
        elif self._family == "qwen3_vl":
            if Qwen3VLForConditionalGeneration is None:
                raise RuntimeError("Qwen3VLForConditionalGeneration not available.")
            ModelClass = Qwen3VLForConditionalGeneration
        else:
            if Qwen2_5_VLForConditionalGeneration is None:
                raise RuntimeError("Qwen2_5_VLForConditionalGeneration not available.")
            ModelClass = Qwen2_5_VLForConditionalGeneration
        try:
            model = ModelClass.from_pretrained(self.cfg.model_id, **model_kwargs)
        except (ImportError, OSError) as exc:
            msg = str(exc)
            if model_kwargs.get("attn_implementation") == "flash_attention_2" and (
                "flash_attn" in msg or "GLIBC_" in msg or "undefined symbol" in msg
            ):
                model_kwargs["attn_implementation"] = "sdpa"
                model = ModelClass.from_pretrained(self.cfg.model_id, **model_kwargs)
            else:
                raise
        model.eval()
        return model

    def infer_pair(self, img_a: Path, img_b: Path) -> Dict[str, Any]:
        imgA = Image.open(img_a).convert("RGB")
        imgB = Image.open(img_b).convert("RGB")
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": imgA},
                    {"type": "image", "image": imgB},
                    {"type": "text", "text": PROMPT_PAIR_DIRECT},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text] if isinstance(text, str) else text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                top_p=self.cfg.top_p,
                do_sample=False,
            )
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        text_out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return normalize_and_filter(json_from_text(text_out))
