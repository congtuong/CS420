_base_ = ["co_dino_5scale_r50_8xb2_1x_coco.py"]

pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth"  # noqa
load_from = "https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth"  # noqa
# load_from = './mmdetection/work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/last_checkpoint'

data_root = (
    "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/data/external/woodscape"
)
backend_args = None
dataset_type = "CocoDataset"
classes = ("vehicles", "persons")
num_classes = len(classes)

custom_imports = dict(
    imports=["mmpretrain.models", "projects.CO-DETR.codetr"], allow_failed_imports=False
)

# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type="mmpretrain.ViTEVA02",
        arch="large",
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.3,
        attn_drop_rate=0.0,
        layer_scale_init_value=0.0,  # disable layer scale when using GRN
        # gap_before_final_norm=False,
        use_grn=True,  # V2 uses GRN
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        feedforward_channels=int(1024 * 4 * 2 / 3),
        patch_norm=True,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-large-p14_pre_in21k_20230505-9072de5d.pth",
            prefix="backbone.",
            pretrain_img_size=384,
            embed_dims=192,
            depths=[2, 2, 18, 2],
        ),
        with_cp=True,
        convert_weights=True,
    ),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    query_head=dict(
        dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)),
        transformer=dict(encoder=dict(with_cp=6)),
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 2048),
                        (512, 2048),
                        (544, 2048),
                        (576, 2048),
                        (608, 2048),
                        (640, 2048),
                        (672, 2048),
                        (704, 2048),
                        (736, 2048),
                        (768, 2048),
                        (800, 2048),
                    ],
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="RandomChoiceResize",
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200)],
                    keep_ratio=True,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 2048),
                        (512, 2048),
                        (544, 2048),
                        (576, 2048),
                        (608, 2048),
                        (640, 2048),
                        (672, 2048),
                        (704, 2048),
                        (736, 2048),
                        (768, 2048),
                        (800, 2048),
                    ],
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type="PackDetInputs"),
]

# train_dataloader = dict(
#     batch_size=1, num_workers=1, dataset=dict(pipeline=train_pipeline))

metainfo = {
    'classes': ('vehicles', 'persons', ),
    'palette': [
        (220, 20, 60),
        (220, 20, 60),
    ]
}

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        # classes=classes,
        pipeline=train_pipeline,
        data_root=data_root,
        ann_file="woodscape_train.json",
        data_prefix=dict(img="images"),
        backend_args=backend_args,
    ),
)

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 1280), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

# val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        metainfo=metainfo,
        type=dataset_type,
        pipeline=test_pipeline,
        # classes=classes,
        data_root=data_root,
        ann_file="woodscape_val.json",
        data_prefix=dict(img="images"),
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(  # Validation evaluator config
    type="CocoMetric",  # The coco metric used to evaluate AR, AP, and mAP for detection and instance segmentation
    # classes=classes,  # Classes to be evaluated
    ann_file=data_root + "/woodscape_val.json",  # Annotation file path
    metric=[
        "bbox"
    ],  # Metrics to be evaluated, `bbox` for detection and `segm` for instance segmentation
    format_only=False,
    backend_args=backend_args,
)

test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(lr=1e-4))

max_epochs = 10
train_cfg = dict(max_epochs=max_epochs, val_interval=100)

param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8],
        gamma=0.1,
    )
]
