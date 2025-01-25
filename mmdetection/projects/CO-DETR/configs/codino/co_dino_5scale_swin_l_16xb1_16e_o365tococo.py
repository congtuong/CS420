_base_ = ['co_dino_5scale_r50_8xb2_1x_coco.py']

dataset_type = 'CocoDataset'
data_root = '/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/data/splited'
backend_args = None

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'  # noqa

# pretrained = None
load_from = None

'CocoDataset'
dataset_type = 'CocoDataset'
classes = ('bus', 'bike', 'car', 'pedestrian', 'truck')
num_classes = len(classes)

# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        convert_weights=True,
        with_cp=True,
        # num_classes=num_classes,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    query_head=dict(
        # num_classes=num_classes,
        dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)),
        transformer=dict(encoder=dict(with_cp=6))))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 2048), (512, 2048), (544, 2048), (576, 2048),
                            (608, 2048), (640, 2048)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 2048), (512, 2048), (544, 2048), (576, 2048),
                            (608, 2048), (640, 2048)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=1, num_workers=1, dataset=dict(
        type=dataset_type,
        # classes=classes,
        pipeline=train_pipeline,
        data_root=data_root,
        ann_file="train.json",
        data_prefix=dict(img='train/images'),
        backend_args=backend_args
    ))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1280), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(dataset=dict(
    type=dataset_type,
    pipeline=test_pipeline,
    # classes=classes,
    data_root=data_root,
    ann_file="val.json",
    data_prefix=dict(img='val/images'),
    backend_args=backend_args
))

test_dataloader = val_dataloader

val_evaluator = dict(  # Validation evaluator config
    type='CocoMetric',  # The coco metric used to evaluate AR, AP, and mAP for detection and instance segmentation
    # classes=classes,  # Classes to be evaluated
    ann_file=data_root + '/val.json',  # Annotation file path
    metric=['bbox'],  # Metrics to be evaluated, `bbox` for detection and `segm` for instance segmentation
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator  # Testing evaluator config

optim_wrapper = dict(optimizer=dict(lr=1e-4))

max_epochs = 16
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8],
        gamma=0.1)
]
