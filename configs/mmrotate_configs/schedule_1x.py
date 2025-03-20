# evaluation 'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU', 'mAcc', 'aAcc'
evaluation = dict(interval=1, metric='mAP', save_best='bbox_mAP', rule='greater')
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
fp16_cfg = dict(loss_scale="dynamic",) #fp16 optimizer args
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    by_epoch=True,
    warmup_iters=100,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=5e-6)
runner = dict(type='EpochBasedRunner', 
            max_epochs=80,
        )
checkpoint_config = dict(interval=1, save_last = True, by_epoch=True, max_keep_ckpts=3)

