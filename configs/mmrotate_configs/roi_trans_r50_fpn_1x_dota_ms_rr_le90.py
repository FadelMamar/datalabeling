_base_ = ['./roi_trans_r50_fpn_1x_dota_le90.py']


model = dict(train_cfg=dict(rpn=dict(assigner=dict(gpu_assign_thr=200))))
