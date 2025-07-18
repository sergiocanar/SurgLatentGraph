import os

_base_ = ['lg_base_box.py']

# dataset
train_dataloader = dict(
    dataset=dict(
        ann_file='train/annotation_coco.json',
        data_prefix=dict(img='train'),
    )
)
val_dataloader = dict(
    dataset=dict(
        ann_file='val/annotation_coco.json',
        data_prefix=dict(img='val'),
    )
)
test_dataloader = dict(
    dataset=dict(
        ann_file='test/annotation_coco.json',
        data_prefix=dict(img='test'),
    )
)

# metric
val_evaluator = dict(
    type='CocoMetricRGD',
    prefix='sages',
    data_root=_base_.data_root,
    data_prefix='val',
    ann_file=os.path.join(_base_.data_root, 'val/annotation_coco.json'),
    metric=['bbox', 'segm'],
    additional_metrics=['reconstruction'],
    use_pred_boxes_recon=False,
    num_classes=-1, # ds_num_classes
)

test_evaluator = dict(
    type='CocoMetricRGD',
    prefix='sages',
    data_root=_base_.data_root,
    data_prefix='test',
    ann_file=os.path.join(_base_.data_root, 'test/annotation_coco.json'),
    #data_prefix='test',
    #ann_file=os.path.join(_base_.data_root, 'test/annotation_coco.json'),
    metric=['bbox', 'segm'],
    additional_metrics=['reconstruction'],
    use_pred_boxes_recon=False,
    num_classes=-1, # ds num classes
    outfile_prefix='./results/endoscapes_preds/test/lg',
    classwise=True,
)

default_hooks = dict(
    checkpoint=dict(save_best='sages/segm_mAP'),
)

# training schedule
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=60,
        by_epoch=True,
        milestones=[32, 54],
        gamma=0.1)
]

#optim_wrapper = dict(
#    optimizer=dict(lr=0.001),
#    paramwise_cfg=dict(
#        custom_keys={
#            'mask_head': dict(lr_mult=10),
#        }
#    ),
#)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=60,
    val_interval=3)
