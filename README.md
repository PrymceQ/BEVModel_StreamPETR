# BEVModel_StreamPETR
The reproduction project of the BEV model # StreamPETR, which includes some code annotation work

Thanks for the CMT authors！[Paper](https://arxiv.org/pdf/2301.01283.pdf) | [Code](https://github.com/junjie18/CMT)

## 🌵Necessary File Format
- data/nuscenes/
  - maps/
  - samples/
  - sweeps/
  - v1.0-test/
  - v1.0-trainval/
- projects/
- tools/
- ckpts/

## 🌵Build Envs
You can refer to the official configuration environment documentation. [Official Git](https://github.com/junjie18/CMT)

Or use the Conda env configuration file we provide.
```
conda env create -f cmt_env.yaml
```

## 🌵Data create

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

## 🌵Train Code
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash tools/dist_train.sh projects/configs/fusion/cmt_voxel0100_r50_800x320_cbgs.py 4
```

## 🌵Test Code
```
python tools/test.py projects/configs/fusion/cmt_voxel0100_r50_800x320_cbgs.py ckpts/voxel0100_r50_800x320_epoch20.pth --eval bbox
```

## 🌵Training Result Record

ID | Name | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE | Per-class results | Epochs | Data | Learning rate | Batch_size | GPUs | Train_time | Eval_time | Log_file
:----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :-----------
0 | voxel0100_r50_800x320_epoch20 | 0.6275 | 0.6784 | 0.3294 | 0.2541 | 0.3035 | 0.2810 | 0.1853 |  ![img1](https://github.com/PrymceQ/BEVModel_CMT/assets/109404970/c8c6b476-3cac-47b8-8cdf-27bf5154910d) | 20 | All | optimizer.lr=0.00007, lr_config.target_ratio=(3, 0.0001), | 8, sample per gpu=2 | 4 x Nvidia Geforce 3090 | 4days8hours | 83.6s | work_dirs/cmt_voxel0100_r50_800x320_cbgs_20230717/


## 🌵Resolved issues
### 😲Out of memory when training.

- Official devices -> 8 x Tesla A100 80G
- Our devices -> 4 x Nvidia Geforce 3090

Config file use "with_cp=True" not "with_cp=False".

### 😨Appear "grad_norm：nan" from 7-8 epoches.

1. Change lr according to your devices and sample_per_gpu;
```python
optimizer = dict(
    type='AdamW',
    lr=0.00007,    # 0.00014 for 8 * 2 batchsize
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.01, decay_mult=5),
            'img_neck': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)  # for 8gpu * 2sample_per_gpu
```
2. Change the lr_config.target_ratio.
```python
lr_config = dict(
    policy='cyclic',
    target_ratio=(3, 0.0001),  # (up_ratio, down_ratio)  # [default] target_ratio=(6, 0.0001) # change the up_ratio=6 to 3
    cyclic_times=1,
    step_ratio_up=0.4)
```

### 😰After CTRL+C terminated the CMT training program, cuda memory still occupied.

1. Use the code to find your [PID];
```
ps -ef ｜grep [command]
```
2. Kill them.
```
kill -9 [PID]
```

## 🌵Key Model Files

Here we have made simple annotations on some key model files in chinese, these annotations are based on "voxel0100_r50_800x320_epoch20" config. 

You can find them in:
- projects\mmdet3d_plugin\models\detectors\cmt.py
- projects\mmdet3d_plugin\models\dense_heads\cmt_head.py
- projects\mmdet3d_plugin\models\necks\cp_fpn.py
- projects\mmdet3d_plugin\models\utils\cmt_transformer.py
- projects\mmdet3d_plugin\models\utils\grid_mask.py

