# BEVModel_StreamPETR
The reproduction project of the BEV model # StreamPETR, which includes some code annotation work.

Thanks for the StreamPETR authors！[Paper](https://arxiv.org/abs/2303.11926) | [Code](https://github.com/exiawsh/StreamPETR)

## 🌵Necessary File Format
- data/nuscenes/
  - maps/
  - samples/
  - sweeps/
  - v1.0-test/
  - v1.0-trainval/
- nusc_tracking/
- projects/
- tools/

## 🌵Build Envs
You can refer to the official configuration environment documentation. [Official Git](https://github.com/exiawsh/StreamPETR)

Or use the Conda env configuration file we provide.
```
conda env create -f streamPETR_env.yaml
```

## 🌵Data create

```
python tools/create_data_nusc.py --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes2d --version v1.0
```

## 🌵Train Code
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
tools/dist_train.sh projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_24e.py 8 --work-dir work_dirs/stream_petr_r50_flash_704_bs2_seq_24e/
```

## 🌵Test Code
```
tools/dist_test.sh projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_24e.py work_dirs/stream_petr_r50_flash_704_bs2_seq_24e/latest.pth 8 --eval bbox
```

## 🌵Training Result Record

ID | Name | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE | Per-class results | Epochs | Data | Learning rate | Batch_size | GPUs | Train_time | Eval_time | Log_file
:----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :-----------
0 | stream_petr_r50_flash_704_bs2_seq_24e | 0.3841 | 0.4859 | 0.6772 | 0.2732 | 0.6244 | 0.2834 | 0.2030 |  ![670fe60d-e182-4dc6-bfa8-872add5b4452](https://github.com/PrymceQ/BEVModel_StreamPETR/assets/109404970/265940ad-8066-4fe8-b803-766897c7d5c7) | 24 | All | optimizer.lr=4e-4 | 16, sample per gpu=2 | 8 x Nvidia Geforce 3090 | 9hours | 113.5s | work_dirs/stream_petr_r50_flash_704_bs2_seq_24e_20230725_bs16_lr4/
1 | stream_petr_vov_flash_800_bs2_seq_24e | 0.4840 | 0.5741 | 0.6153 | 0.2592 | 0.3510 | 0.2567 | 0.1971 | ![15a7dc32-d873-4481-b160-ace9bffd44d3](https://github.com/PrymceQ/BEVModel_StreamPETR/assets/109404970/1a4975e0-955f-4a87-951d-b124ff35a5a4) | 24 | All | optimizer.lr=4e-4 | 16, sample per gpu=2 | 8 x Nvidia Geforce 3090 | 13hours | 104.9s | work_dirs/stream_petr_vov_flash_800_bs2_seq_24e_20230726/

## 🌵Some useful tools
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

