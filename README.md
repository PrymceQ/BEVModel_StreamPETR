# BEVModel_StreamPETR
The reproduction project of the BEV model # StreamPETR, which includes some code annotation work.

Thanks for the StreamPETR authors！[Paper](https://arxiv.org/abs/2303.11926) | [Code](https://github.com/exiawsh/StreamPETR)

## 🌵Necessary File Format
- mmdetectioned/    # git clone https://github.com/open-mmlab/mmdetection3d.git
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
export CUDA_VISIBLE_DEVICES=0,1,2,3,
tools/dist_train.sh projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_24e.py 4 --work-dir work_dirs/stream_petr_r50_flash_704_bs2_seq_24e/
```

> You need to modify the `num_gpus`/`batch_size`/`optimizer.lr` in the CONFIG FILE accordingly, otherwise it will cause the training to not converge.

> num_gpus * batch_size==8: lr=2e-4; num_gpus * batch_size==16: lr=4e-4

## 🌵Test Code
```
tools/dist_test.sh projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_24e.py work_dirs/stream_petr_r50_flash_704_bs2_seq_24e/latest.pth 4 --eval bbox
```

## 🌵Training Result Record

ID | Name | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE | Per-class results | Epochs | Data | Learning rate | Batch_size | GPUs | Train_time | Eval_time | Log_file
:----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :-----------
0 | stream_petr_r50_flash_704_bs2_seq_24e | 0.3841 | 0.4859 | 0.6772 | 0.2732 | 0.6244 | 0.2834 | 0.2030 |  ![670fe60d-e182-4dc6-bfa8-872add5b4452](https://github.com/PrymceQ/BEVModel_StreamPETR/assets/109404970/265940ad-8066-4fe8-b803-766897c7d5c7) | 24 | All | optimizer.lr=4e-4 | 16, sample per gpu=4 | 4 x Nvidia Geforce 3090 | 9hours | 113.5s | work_dirs/stream_petr_r50_flash_704_bs2_seq_24e_20230725_bs16_lr4/
1 | stream_petr_vov_flash_800_bs2_seq_24e | 0.4840 | 0.5741 | 0.6153 | 0.2592 | 0.3510 | 0.2567 | 0.1971 | ![15a7dc32-d873-4481-b160-ace9bffd44d3](https://github.com/PrymceQ/BEVModel_StreamPETR/assets/109404970/1a4975e0-955f-4a87-951d-b124ff35a5a4) | 24 | All | optimizer.lr=4e-4 | 16, sample per gpu=4 | 4 x Nvidia Geforce 3090 | 13hours | 104.9s | work_dirs/stream_petr_vov_flash_800_bs2_seq_24e_20230726/

## 🌵Some useful tools
### 😲Visualization!
<img src="https://github.com/PrymceQ/BEVModel_StreamPETR/blob/master/imgs/1dfecb8189f54b999f4e47ddaa677fd0_pred.png" width="600px">

```
tools/dist_test.sh projects/configs/StreamPETR/stream_petr_vov_flash_800_bs2_seq_24e.py work_dirs/stream_petr_vov_flash_800_bs2_seq_24e/latest.pth 4 --format-only
```


### 😲Create Loss Curve!

<img src="https://github.com/PrymceQ/BEVModel_StreamPETR/blob/master/imgs/loss.png" width="260px">

```
python mmdetection3d/tools/analysis_tools/analyze_logs.py plot_curve /home/wangziqin/StreamPETR/work_dirs/stream_petr_r50_flash_704_bs2_seq_24e_20230724_4e-4/20230724_122923.log.json --keys loss
```

### 😲Create Tracking json with `tracking_id`!

<img src="https://github.com/PrymceQ/BEVModel_StreamPETR/blob/master/imgs/trackjson.png" width="560px">

```
python nusc_tracking/pub_test.py --checkpoint val/work_dirs/stream_petr_vov_flash_800_bs2_seq_24e_20230726/Wed_Jul_26_10_38_31_2023/pts_bbox/results_nusc.json --work_dir ./tracking_out_results
```

## 🌵Key Model Files

Here we have made simple annotations on some key model files in chinese, these annotations are based on "voxel0100_r50_800x320_epoch20" config. 

You can find them in:
- projects/mmdet3d_plugin/models/detectors/petr3d.py
- projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py
- projects/mmdet3d_plugin/models/dense_heads/focal_head.py

