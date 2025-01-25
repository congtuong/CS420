from sahi.slicing import slice_coco, slice_image
from pathlib import Path
import shutil
import cv2

root_img = Path("/mlcv1/Datasets/NvidiaAICityChallenge/NvidiaAIC2024/images")
output_img = Path("./test_images")


# def generate(splits):
#     for split in splits:
#         (output_img / split / "images").mkdir(parents=True, exist_ok=True)

#         coco_dict, coco_path = slice_coco(
#             coco_annotation_file_path=str(root_img / f"{split}.json"),
#             image_dir=str(root_img / f"{split}" / "images"),
#             output_coco_annotation_file_name=f"{split}_sliced",
#             output_dir=str(output_img / split / "images"),
#             slice_height=480,
#             slice_width=480,
#             overlap_height_ratio=0.2,
#             overlap_width_ratio=0.2,
#         )

#         shutil.copy(output_img / split / "images" / f"{split}_sliced_coco.json", output_img / split / f"{split}.json")


def generate(splits):
    for split in splits:
        output_img.mkdir(parents=True, exist_ok=True)

        for p_img in root_img.glob("*.png"):
            img = cv2.imread(str(p_img))
            img = cv2.resize(img, (1230, int(1230 * img.shape[0] / img.shape[1])))

            slice_image_result = slice_image(
                image=img,
                output_file_name=p_img.name,
                output_dir=output_img,
                slice_height=480,
                slice_width=480,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )

            for img in slice_image_result.images:
                cv2.imwrite(str(output_img / p_img.name), img)
                
            # shutil.copy(output_img / split / "images" / f"{split}_sliced_coco.json", output_img / split / f"{split}.json")

generate(["test"]) 
# generate(["train", "val"])