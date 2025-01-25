_base_ = "./cascade-rcnn_r50_fpn_20e_coco.py"

load_from = "https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth"
backend_args = None
dataset_type = 'CocoDataset'
data_root = "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/data/splited"
classes = ("bus", "bicycle", "car", "person", "truck")
num_classes = len(classes)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="train.json",
        data_prefix=dict(img="train/images"),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="val.json",
        data_prefix=dict(img="val/images"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "/val.json",  # Annotation file path
    metric="bbox",
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator

model = dict(
    backbone=dict(
        type="ResNeXt",
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://resnext101_32x4d"),
    )
)
