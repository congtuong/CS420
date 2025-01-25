from sahi.slicing import slice_coco, slice_image
from pathlib import Path
import shutil
import cv2


root_img = Path(
    "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/data/yolo_all_classes"
)
output_img = Path(
    "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/data/sliced_square"
)


def generate(splits):
    for split in splits:
        output_img.mkdir(parents=True, exist_ok=True)
        (output_img / split).mkdir(parents=True, exist_ok=True)

        coco_dict, coco_path = slice_coco(
            coco_annotation_file_path=str(root_img / f"{split}.json"),
            image_dir=str(root_img / f"{split}" / "images"),
            output_coco_annotation_file_name=f"{split}_sliced",
            output_dir=str(output_img / split / "images"),
            slice_height=None,
            slice_width=None,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            square=True,
        )

        shutil.copy(
            output_img / split / "images" / f"{split}_sliced_coco.json",
            output_img / split / f"{split}.json",
        )


# generate(["test"])
generate(["train", "val"])
