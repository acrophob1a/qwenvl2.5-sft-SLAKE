import copy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import transformers
import ujson as json
from PIL import Image
from torch.utils.data import Dataset
from utils.constants import IGNORE_INDEX
from utils.rope2d import get_rope_index_25

from .tsv_dataset import preprocess_qwen_2_visual


class SlakeVQADataset(Dataset):
    """SLAKE medical VQA dataset for Qwen2.5-VL SFT."""

    def __init__(
        self,
        ann_json_file: str,
        image_root: str,
        tokenizer,
        data_args,
        image_min_pixels,
        image_max_pixels,
        max_num_samples: Optional[int] = None,
        q_lang: Optional[str] = "en",
        system_message: str = "You are a helpful medical assistant.",
        ratio_range=(0.0, 1.0),
        dataset_name: str = "slake_vqa",
        max_length: int = 4096,
    ):
        super().__init__()
        ann_path = Path(ann_json_file)
        image_root = Path(image_root)
        samples = json.load(open(ann_path))

        if q_lang is not None:
            samples = [s for s in samples if s.get("q_lang") == q_lang]

        start = int(len(samples) * ratio_range[0])
        end = int(len(samples) * ratio_range[1])
        samples = samples[start:end]

        valid_samples = []
        for sample in samples:
            img_path = image_root / sample["img_name"]
            if img_path.exists():
                valid_samples.append(sample)
        self.samples = valid_samples

        if max_num_samples is not None:
            np.random.shuffle(self.samples)
            self.samples = self.samples[:max_num_samples]

        self.image_root = image_root
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.system_message = system_message
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.get_rope_index = get_rope_index_25

        self.data_args.image_processor.max_pixels = image_max_pixels
        self.data_args.image_processor.min_pixels = image_min_pixels
        self.data_args.image_processor.size["longest_edge"] = image_max_pixels
        self.data_args.image_processor.size["shortest_edge"] = image_min_pixels

    def __len__(self):
        return len(self.samples)

    def process_image(self, image_file: Path):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image, image_tensor, grid_thw

    def build_conversations(self, sample: dict) -> List[dict]:
        return [
            {"from": "human", "value": f"<image>\n{sample['question']}"},
            {"from": "gpt", "value": sample["answer"]},
        ]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            sample = self.samples[i]
            image_path = self.image_root / sample["img_name"]
            image_pil, image_tensor, grid_thw = self.process_image(image_path)
        except Exception as e:
            print(f"error sample {self.dataset_name} id: {i}. {e}")
            return self.__getitem__((i + 1) % len(self.samples))

        w, h = image_pil.size
        if w < 28 or h < 28:
            return self.__getitem__((i + 1) % len(self.samples))

        grid_thw_merged = [grid_thw.prod() // self.data_args.image_processor.merge_size**2]
        conversations = self.build_conversations(sample)
        data_dict = preprocess_qwen_2_visual(
            [conversations],
            self.tokenizer,
            grid_thw=grid_thw_merged,
            visual_type="image",
            system_message=self.system_message,
        )
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            torch.stack([grid_thw], dim=0),
        )

        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
            position_ids=position_ids,
            pixel_values=image_tensor,
            image_grid_thw=[grid_thw],
        )

        if data_dict["input_ids"].size(0) > self.max_length:
            return self.__getitem__((i + 1) % len(self.samples))

        return data_dict
