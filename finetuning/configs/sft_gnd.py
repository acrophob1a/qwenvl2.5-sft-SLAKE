from dataset import (
    ConcatDataset,
    DataCollatorForSupervisedDataset,
    GroundingTSVDataset,
)
from dataset.task_fns import GroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import (
    GROUNDING_SINGLE_REGION_STAGE_XYXY,
)

add_grounding_specific_tokens = True

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28

model_name_or_path = "pretrained/Qwen2.5-VL-3B-Instruct"
dataset_path = "datasets/Grounding-ToyData"

grounding_data = dict(
    type=GroundingTSVDataset,
    img_tsv_file=f"{dataset_path}/toy_data.images.tsv",
    ann_tsv_file=f"{dataset_path}/toy_data.annotations.tsv",
    ann_lineidx_file=f"{dataset_path}/toy_data.annotations.tsv.lineidx",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="grounding_toydata",
)

train_dataset = dict(
    type=ConcatDataset,
    datasets=[grounding_data],
)

data_collator = dict(type=DataCollatorForSupervisedDataset)
