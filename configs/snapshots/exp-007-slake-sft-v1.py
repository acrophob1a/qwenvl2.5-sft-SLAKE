from dataset import (
    ConcatDataset,
    DataCollatorForSupervisedDataset,
    SlakeVQADataset,
)

add_grounding_specific_tokens = False

min_pixels = 16 * 28 * 28
max_pixels = 1280 * 28 * 28

model_name_or_path = "pretrained/Qwen2.5-VL-3B-Instruct"
dataset_root = "datasets/SLAKE"

slake_train = dict(
    type=SlakeVQADataset,
    ann_json_file=f"{dataset_root}/train.json",
    image_root=f"{dataset_root}/imgs",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    q_lang="en",
    dataset_name="slake_vqa_train",
)

train_dataset = dict(
    type=ConcatDataset,
    datasets=[slake_train],
)

data_collator = dict(type=DataCollatorForSupervisedDataset)
