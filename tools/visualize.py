# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
import numpy as np
from tqdm import tqdm
from mmcv import Config
from mmdet3d.datasets import build_dataset
import matplotlib.pyplot as plt
from mot_3d.data_protos import BBox
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from mot_3d.visualization.visualizer2d import Visualizer2D


def parse_args():
    parser = argparse.ArgumentParser(description='3D Tracking Visualization')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--result', help='results file in pickle format')
    parser.add_argument(
        '--show-dir', help='directory where visualize results will be saved')
    args = parser.parse_args()

    return args

def create_info_dict(data: list, thre: int) -> dict:
    trackID_loc_d = {}

    for frame_key in tqdm(data.keys()):
        frame = data[frame_key]
        for sample in frame:
            tmp_sample_token = sample['sample_token']
            tmp_translation = sample['translation']
            tmp_tracking_id = sample['tracking_id'].split("-")[-1]
            # cut thred
            tmp_score = sample['tracking_score']
            if tmp_score < tmp_score:
                continue
            tmp_loc = tmp_translation[:2]  # x,y

            # create trackID_loc_d dict
            if tmp_tracking_id not in trackID_loc_d:
                trackID_loc_d[tmp_tracking_id] = np.array([tmp_loc])
            else:
                trackID_loc_d[tmp_tracking_id] = np.vstack((trackID_loc_d[tmp_tracking_id], tmp_loc))

    print("################### CREATE DOWN trackID_loc_d!")
    return trackID_loc_d
def filter_trajectory_dictionary(dictionary, query_list):
    filtered_dict = {}
    for track_id, trajectory in dictionary.items():
        if track_id in query_list:
            filtered_dict[track_id] = trajectory
    return filtered_dict
def get_max_tracking_points(trackID_loc_d):
    Max_Ntrack = 0
    for v in trackID_loc_d.values():
        Max_Ntrack = max(Max_Ntrack, len(v))
    return Max_Ntrack

def get_track_from_json(json_f: str) -> np.ndarray:
    # read_json and get all tracks
    with open(json_f) as f:
        raw_data = json.load(f)

    print(raw_data.keys())
    print(raw_data['meta'])

    # data
    data = raw_data['results']
    print("Frame num: ", len(data))

    trackID_loc_d = create_info_dict(data,
                                     thre=0.4)  # trackingId_sampleToken_d: trackingId's sampleToken is sorted by time
    print("Tracking id num: ", len(trackID_loc_d.keys()))

    Max_Ntrack = get_max_tracking_points(trackID_loc_d)
    tracking_infos = info_dict2array(trackID_loc_d)
    print("Tracking_infos.shape: ", tracking_infos.shape)
    return tracking_infos
def info_dict2array(trackID_loc_d) -> np.array:
    re = []
    Max_Ntrack = get_max_tracking_points(trackID_loc_d)
    for k in trackID_loc_d.keys():
        v = [list(i) for i in list(trackID_loc_d[k])]
        re.append(v)
    # 2Max_Ntrack
    for i, r in enumerate(re):
        gap = Max_Ntrack - len(r)
        r = r + [r[-1]] * gap
        assert len(r) == Max_Ntrack
        re[i] = r
    re = np.array(re)
    return re

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    dataset = build_dataset(cfg.data.visualization)
    results = json.load(open(args.result))['results']
    sample_tokens = results.keys()
    data_infos = dataset.data_infos
    data_info_sample_tokens = [info['token'] for info in data_infos]
    tmp_track=dict()
    pbar = tqdm(total=len(results))
    for sample_idx, sample_token in enumerate(sample_tokens):
        # locate the information in data_infos
        data_info_idx = data_info_sample_tokens.index(sample_token)
        sample_info = data_infos[data_info_idx]
        raw_data = dataset[data_info_idx]

        # create location for visualization
        scene_token = sample_info['scene_token']
        seq_dir = os.path.join(args.show_dir, scene_token)
        os.makedirs(seq_dir, exist_ok=True)

        # get the point cloud information
        pc = raw_data['points'].data[0].numpy()[:, :3]
        mask = (np.max(pc, axis=-1) < 60)
        pc = pc[mask]

        l2e_r = sample_info['lidar2ego_rotation']
        l2e_t = sample_info['lidar2ego_translation']
        e2g_r = sample_info['ego2global_rotation']
        e2g_t = sample_info['ego2global_translation']
        l2e_r = Quaternion(l2e_r).rotation_matrix
        e2g_r = Quaternion(e2g_r).rotation_matrix
        l2e, e2g = np.eye(4), np.eye(4)
        l2e[:3, :3], l2e[:3, 3] = l2e_r, l2e_t
        e2g[:3, :3], e2g[:3, 3] = e2g_r, e2g_t
        l2g = e2g @ l2e
        new_pcs = np.concatenate((pc,
                                  np.ones(pc.shape[0])[:, np.newaxis]),
                                 axis=1)
        pc = ((new_pcs @ l2e.T) @ e2g.T)[:, :3]

        # gt_bboxes, instance_ids = sample_info['gt_boxes'], sample_info['instance_inds']
        visualizer = Visualizer2D(name=str(data_info_idx), figsize=(20, 20))
        COLOR_KEYS = list(visualizer.COLOR_MAP.keys())
        visualizer.handler_pc(pc)

        ego_xyz = l2g[:3, 3]
        plt.xlim((ego_xyz[0] - 60, ego_xyz[0] + 60))
        plt.ylim((ego_xyz[1] - 60, ego_xyz[1] + 60))

        frame_results = results[sample_token]
        query=[]
        for i, box in enumerate(frame_results):
            if box['tracking_score'] < 0.4:
                continue
            nusc_box = Box(box['translation'], box['size'], Quaternion(box['rotation']))
            mot_bbox = BBox(
                x=nusc_box.center[0], y=nusc_box.center[1], z=nusc_box.center[2],
                w=nusc_box.wlh[0], l=nusc_box.wlh[1], h=nusc_box.wlh[2],
                o=nusc_box.orientation.yaw_pitch_roll[0]
            )
            track_id = int(box['tracking_id'].split('-')[-1])
            color = COLOR_KEYS[track_id % len(COLOR_KEYS)]
            # visualizer.handler_box(mot_bbox, message='', color=color)
            visualizer.handler_box(mot_bbox, message=box['tracking_id'].split('-')[-1], color=color)

        #
        color_list = list()
        for i, box in enumerate(frame_results):
            if box['tracking_score'] < 0.4:
                continue
            trajs = np.array(box['translation'])[:2]
            track_id = int(box['tracking_id'].split('-')[-1])
            color = visualizer.COLOR_MAP[COLOR_KEYS[track_id % len(COLOR_KEYS)]]
            color_list.append(color)
            if track_id not in tmp_track:
                tmp_track[track_id] = np.array([trajs])
            else:
                tmp_track[track_id] = np.vstack((tmp_track[track_id],trajs))

            query.append(track_id)
        tmp_track = filter_trajectory_dictionary(tmp_track,query)
        Max_NTack = get_max_tracking_points(tmp_track)
        tracking_infos=info_dict2array(tmp_track) # 25x 1 x2
        # 可视化过去轨迹
        all_trajs = tracking_infos
        # color_list = list()
        # for i in range(len(tracking_infos)):
        #     color = visualizer.COLOR_MAP[COLOR_KEYS[i% len(COLOR_KEYS)]]
        #     color_list.append(color)

        if len(all_trajs) > 0:
            all_trajs = tracking_infos
            traj_num, T, dim = all_trajs.shape
            new_trajs = all_trajs
            for i in range(traj_num):
                plt.plot(new_trajs[i, :, 0], new_trajs[i, :, 1], color=color_list[i])


        visualizer.save(os.path.join(seq_dir, f'{data_info_idx}.png'))
        visualizer.close()
        pbar.update(1)
    pbar.close()

    print('Making Videos')
    scene_tokens = os.listdir(args.show_dir)
    for video_index, scene_token in enumerate(scene_tokens):
        show_dir = os.path.join(args.show_dir, scene_token)
        fig_names = os.listdir(show_dir)
        indexes = sorted([int(fname.split('.')[0]) for fname in fig_names if fname.endswith('png')])
        fig_names = [f'{i}.png' for i in indexes]

        make_videos(show_dir, fig_names, 'videobev.mp4', show_dir)


def make_videos(fig_dir, fig_names, video_name, video_dir):
    import imageio
    import os
    import cv2

    fileList = list()
    for name in fig_names:
        fileList.append(os.path.join(fig_dir, name))

    writer = imageio.get_writer(os.path.join(video_dir, video_name), fps=2)
    for im in fileList:
        writer.append_data(cv2.resize(imageio.imread(im), (2000, 2000)))
    writer.close()
    return


if __name__ == '__main__':
    main()