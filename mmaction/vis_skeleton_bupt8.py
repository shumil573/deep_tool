'''
本文档整理自 https://github.com/open-mmlab/mmaction2/blob/master/demo/visualize_heatmap_volume.ipynb

'''
import os
import cv2
import os.path as osp
import decord
import numpy as np
import matplotlib.pyplot as plt
import urllib
import moviepy.editor as mpy
import random as rd
from mmpose.apis import vis_pose_result
from mmpose.models import TopDown
from mmcv import load, dump

# We assume the annotation is already prepared
gym_train_ann_file = '../data/skeleton/gym_train.pkl'
gym_val_ann_file = '../data/skeleton/gym_val.pkl'
# #下面这两个路径的文件不存在（？那我之前是怎么跑的？？？
# # ntu60_xsub_train_ann_file = '../data/skeleton/ntu60_xsub_train.pkl'
# # ntu60_xsub_val_ann_file = '../data/skeleton/ntu60_xsub_val.pkl'
# ntu60_xsub_train_ann_file = '../data/posec3d/ntu60_xsub_train.pkl'
# ntu60_xsub_val_ann_file = '../data/posec3d/ntu60_xsub_val.pkl'
# bupt_train_ann_file='../data/posec3d/bupt_check_train20.pkl'
# bupt_val_ann_file='../data/posec3d/bupt_check_val8.pkl'

#下面这两个路径的文件不存在（？那我之前是怎么跑的？？？
# ntu60_xsub_train_ann_file = '../data/skeleton/ntu60_xsub_train.pkl'
# ntu60_xsub_val_ann_file = '../data/skeleton/ntu60_xsub_val.pkl'
ntu60_xsub_train_ann_file = 'tool/ntu60_xsub_train.pkl'
ntu60_xsub_val_ann_file = 'tool/ntu60_xsub_val.pkl'
bupt_train_ann_file='tool/bupt_check_train20.pkl'
bupt_val_ann_file='tool/bupt_check_val8.pkl'

####----------------------------区块----------------------------####

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.6
FONTCOLOR = (255, 255, 255)
BGBLUE = (0, 119, 182)
THICKNESS = 1
LINETYPE = 1

####----------------------------区块----------------------------####

def add_label(frame, label, BGCOLOR=BGBLUE):
    threshold = 30

    def split_label(label):
        label = label.split()
        lines, cline = [], ''
        for word in label:
            if len(cline) + len(word) < threshold:
                cline = cline + ' ' + word
            else:
                lines.append(cline)
                cline = word
        if cline != '':
            lines += [cline]
        return lines

    if len(label) > 30:
        label = split_label(label)
    else:
        label = [label]
    label = ['Action: '] + label

    sizes = []
    for line in label:
        sizes.append(cv2.getTextSize(line, FONTFACE, FONTSCALE, THICKNESS)[0])
    box_width = max([x[0] for x in sizes]) + 10
    text_height = sizes[0][1]
    box_height = len(sizes) * (text_height + 6)

    cv2.rectangle(frame, (0, 0), (box_width, box_height), BGCOLOR, -1)
    for i, line in enumerate(label):
        location = (5, (text_height + 6) * i + text_height + 3)
        cv2.putText(frame, line, location, FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)
    return frame


def vis_skeleton(vid_path, anno, category_name=None, ratio=0.5):
    vid = decord.VideoReader(vid_path)
    frames = [x.asnumpy() for x in vid]

    h, w, _ = frames[0].shape
    new_shape = (int(w * ratio), int(h * ratio))
    frames = [cv2.resize(f, new_shape) for f in frames]

    # assert len(frames) == anno['total_frames']
    # The shape is N x T x K x 3
    kps = np.concatenate([anno['keypoint'], anno['keypoint_score'][..., None]], axis=-1)
    kps[..., :2] *= ratio
    # Convert to T x N x K x 3
    kps = kps.transpose([1, 0, 2, 3])
    vis_frames = []

    # we need an instance of TopDown model, so build a minimal one
    model = TopDown(backbone=dict(type='ShuffleNetV1'))

    for f, kp in zip(frames, kps):
        bbox = np.zeros([0, 4], dtype=np.float32)
        result = [dict(bbox=bbox, keypoints=k) for k in kp]
        vis_frame = vis_pose_result(model, f, result)

        if category_name is not None:
            vis_frame = add_label(vis_frame, category_name)

        vis_frames.append(vis_frame)
    return vis_frames

####----------------------------区块----------------------------####
keypoint_pipeline = [
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(type='GeneratePoseTarget', sigma=0.6, use_score=True, with_kp=True, with_limb=False)
]

limb_pipeline = [
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(type='GeneratePoseTarget', sigma=0.6, use_score=True, with_kp=False, with_limb=True)
]
from mmaction.datasets.pipelines import Compose
def get_pseudo_heatmap(anno, flag='keypoint'):
    assert flag in ['keypoint', 'limb']
    pipeline = Compose(keypoint_pipeline if flag == 'keypoint' else limb_pipeline)
    return pipeline(anno)['imgs']


def vis_heatmaps(heatmaps, channel=-1, ratio=8):
    # if channel is -1, draw all keypoints / limbs on the same map
    import matplotlib.cm as cm
    h, w, _ = heatmaps[0].shape
    newh, neww = int(h * ratio), int(w * ratio)

    if channel == -1:
        heatmaps = [np.max(x, axis=-1) for x in heatmaps]
    cmap = cm.viridis
    heatmaps = [(cmap(x)[..., :3] * 255).astype(np.uint8) for x in heatmaps]
    heatmaps = [cv2.resize(x, (neww, newh)) for x in heatmaps]
    return heatmaps

####----------------------------区块----------------------------####

# # Load GYM annotations
#
# f = open(r'list_for_vis.txt','r')  # 更改
# lines = list(f)
# # gym_categories = [x.decode().strip().split('; ')[-1] for x in lines]  # 更改
# gym_categories = [x.strip().split('; ')[-1] for x in lines]  # 更改
# gym_annos = load(gym_train_ann_file) + load(gym_val_ann_file)
#
# ####----------------------------区块----------------------------####
#
# # download sample videos of GYM
# # !wget https://download.openmmlab.com/mmaction/posec3d/gym_samples.tar
# # !tar -xf gym_samples.tar
# # !rm gym_samples.tar
# # #已下载
#
# ####----------------------------区块----------------------------####
#
# gym_root = 'gym_samples/'
# gym_vids = os.listdir(gym_root)
# # visualize pose of which video? index in 0 - 50.
# idx = 1
# vid = gym_vids[idx]
#
# frame_dir = vid.split('.')[0]
# vid_path = osp.join(gym_root, vid)
# anno = [x for x in gym_annos if x['frame_dir'] == frame_dir][0]
#
# ####----------------------------区块----------------------------####
#
# # Visualize Skeleton
# vis_frames = vis_skeleton(vid_path, anno, gym_categories[anno['label']])
# vid = mpy.ImageSequenceClip(vis_frames, fps=24)
# vid.ipython_display()
#
# ####----------------------------区块----------------------------####
#
# keypoint_heatmap = get_pseudo_heatmap(anno)
# keypoint_mapvis = vis_heatmaps(keypoint_heatmap)
# keypoint_mapvis = [add_label(f, gym_categories[anno['label']]) for f in keypoint_mapvis]
# vid = mpy.ImageSequenceClip(keypoint_mapvis, fps=24)
# vid.ipython_display()
#
# ####----------------------------区块----------------------------####
#
# keypoint_heatmap = get_pseudo_heatmap(anno)
# keypoint_mapvis = vis_heatmaps(keypoint_heatmap)
# keypoint_mapvis = [add_label(f, gym_categories[anno['label']]) for f in keypoint_mapvis]
# vid = mpy.ImageSequenceClip(keypoint_mapvis, fps=24)
# vid.ipython_display()
#
# ####----------------------------区块----------------------------####
#
# limb_heatmap = get_pseudo_heatmap(anno, 'limb')
# limb_mapvis = vis_heatmaps(limb_heatmap)
# limb_mapvis = [add_label(f, gym_categories[anno['label']]) for f in limb_mapvis]
# vid = mpy.ImageSequenceClip(limb_mapvis, fps=24)
# vid.ipython_display()

# # ####----------------------------区块----------------------------####
# #
# # The name list of
# ntu_categories = ['drink water', 'eat meal/snack', 'brushing teeth', 'brushing hair', 'drop', 'pickup',
#                   'throw', 'sitting down', 'standing up (from sitting position)', 'clapping', 'reading',
#                   'writing', 'tear up paper', 'wear jacket', 'take off jacket', 'wear a shoe',
#                   'take off a shoe', 'wear on glasses', 'take off glasses', 'put on a hat/cap',
#                   'take off a hat/cap', 'cheer up', 'hand waving', 'kicking something',
#                   'reach into pocket', 'hopping (one foot jumping)', 'jump up',
#                   'make a phone call/answer phone', 'playing with phone/tablet', 'typing on a keyboard',
#                   'pointing to something with finger', 'taking a selfie', 'check time (from watch)',
#                   'rub two hands together', 'nod head/bow', 'shake head', 'wipe face', 'salute',
#                   'put the palms together', 'cross hands in front (say stop)', 'sneeze/cough',
#                   'staggering', 'falling', 'touch head (headache)', 'touch chest (stomachache/heart pain)',
#                   'touch back (backache)', 'touch neck (neckache)', 'nausea or vomiting condition',
#                   'use a fan (with hand or paper)/feeling warm', 'punching/slapping other person',
#                   'kicking other person', 'pushing other person', 'pat on back of other person',
#                   'point finger at the other person', 'hugging other person',
#                   'giving something to other person', "touch other person's pocket", 'handshaking',
#                   'walking towards each other', 'walking apart from each other']
# ntu_annos = load(ntu60_xsub_train_ann_file) + load(ntu60_xsub_val_ann_file)
#
# ####----------------------------区块----------------------------####
#
# ntu_root = 'ntu_samples/'
# ntu_vids = os.listdir(ntu_root)
# # visualize pose of which video? index in 0 - 50.
# idx = 20
# vid = ntu_vids[idx]
#
# frame_dir = vid.split('.')[0]
# vid_path = osp.join(ntu_root, vid)
# anno = [x for x in ntu_annos if x['frame_dir'] == frame_dir.split('_')[0]][0]###作用：在anno.pkl文件里，找到和第idx个视频配套的anno
# print('for bupt,anno:',anno)
#
# ####----------------------------区块----------------------------####
#
# # download sample videos of NTU-60
# # !wget https://download.openmmlab.com/mmaction/posec3d/ntu_samples.tar
# # !tar -xf ntu_samples.tar
# # #通过桌面的datasets文件夹转存为ntu_samples文件夹
#
# ####----------------------------区块----------------------------####
#
# vis_frames = vis_skeleton(vid_path, anno, ntu_categories[anno['label']])
# vid = mpy.ImageSequenceClip(vis_frames, fps=24)
# vid.ipython_display()
# keypoint_heatmap = get_pseudo_heatmap(anno)
# keypoint_mapvis = vis_heatmaps(keypoint_heatmap)
# keypoint_mapvis = [add_label(f, gym_categories[anno['label']]) for f in keypoint_mapvis]
# vid = mpy.ImageSequenceClip(keypoint_mapvis, fps=24)
# vid.ipython_display()
# limb_heatmap = get_pseudo_heatmap(anno, 'limb')
# limb_mapvis = vis_heatmaps(limb_heatmap)
# limb_mapvis = [add_label(f, gym_categories[anno['label']]) for f in limb_mapvis]
# vid = mpy.ImageSequenceClip(limb_mapvis, fps=24)
# vid.ipython_display()
#
# # ####----------------------------区块----------------------------####

# ####----------------------------bupt 10 可视化 区块----------------------------####
#
# The name list of
# bupt_categories = ['走',
# '跑',
# '左脚跳跃',
# '右脚跳跃',
# '双脚跳跃',
# '平摊双手',
# '单摊右手',
# '单摊左手',
# '高举双手']
bupt_categories = ['walk',
'run',
'left jump',
'right jump',
'jump',
'spread two hands',
'spread right',
'spread left',
'raise hands']
# bupt_categories = [0,1,2,3,4,5,6,7,8]
bupt_annos = load(bupt_train_ann_file) + load(bupt_val_ann_file)

####----------------------------区块----------------------------####


bupt_root = '../../../datasets/hsy_tool'
bupt_vids = os.listdir(bupt_root)
# visualize pose of which video? index in 0 - 50.
# idx = 20
error_list=[]
for idx in range(len(bupt_vids)):
    vid = bupt_vids[idx]

    frame_dir = vid.split('.')[0]
    vid_path = osp.join(bupt_root, vid)
    # anno = [x for x in ntu_annos if x['frame_dir'] == frame_dir.split('_')[0]][0]####ntu的代码，用在bupt会报错：IndexError: list index out of range
    # anno = [x for x in bupt_annos if x['frame_dir'] == frame_dir][0]#27_1_3好像找不到
    tool=[]
    for x in bupt_annos:
        if x['frame_dir'] == frame_dir:
            tool.append(x)
    if len(tool)>0 and vid!='14_1_4.mp4':
        anno=tool[0]
        ####----------------------------区块----------------------------####

        # download sample videos of NTU-60
        # !wget https://download.openmmlab.com/mmaction/posec3d/ntu_samples.tar
        # !tar -xf ntu_samples.tar
        # #通过桌面的datasets文件夹转存为ntu_samples文件夹

        ####----------------------------区块----------------------------####

        vis_frames = vis_skeleton(vid_path, anno, bupt_categories[anno['label']])
        vid = mpy.ImageSequenceClip(vis_frames, fps=24)
        vid.write_videofile('bupt10/%s.mp4' % anno['frame_dir'], fps=24, codec='mpeg4')
        # vid.ipython_display()
        keypoint_heatmap = get_pseudo_heatmap(anno)
        keypoint_mapvis = vis_heatmaps(keypoint_heatmap)
        keypoint_mapvis = [add_label(f, bupt_categories[anno['label']]) for f in keypoint_mapvis]
        vid = mpy.ImageSequenceClip(keypoint_mapvis, fps=24)
        vid.write_videofile('bupt10/%s_key.mp4' % anno['frame_dir'], fps=24, codec='mpeg4')
        # vid.ipython_display()
        limb_heatmap = get_pseudo_heatmap(anno, 'limb')
        limb_mapvis = vis_heatmaps(limb_heatmap)
        limb_mapvis = [add_label(f, bupt_categories[anno['label']]) for f in limb_mapvis]
        vid = mpy.ImageSequenceClip(limb_mapvis, fps=24)
        vid.write_videofile('bupt10/%s_limb.mp4' % anno['frame_dir'], fps=24, codec='mpeg4')
        # vid.ipython_display()
    else:
        error_list.append(frame_dir)

print('bupt error video:',error_list)
    # ####----------------------------区块----------------------------####