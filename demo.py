import argparse
import os

import cv2
import numpy as np
import torch
import frame2flow
from glob import glob

from jpg2mp4 import jpg2mp4
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width
from write_json import write_ck_json



# class ImageReader(object):
#     def __init__(self, file_names):
#         self.file_names = file_names
#         self.max_idx = len(file_names)
#
#     def __iter__(self):
#         self.idx = 0
#         return self
#
#     def __next__(self):
#         if self.idx == self.max_idx:
#             raise StopIteration
#         img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
#         if img.size == 0:
#             raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
#         self.idx = self.idx + 1
#         return img
class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names

        self.frames_path = glob(os.path.join(''.join(file_names), '*.jpg'))
        self.frames_path.sort()
        self.max_idx = len(self.frames_path)


    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.frames_path[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name,)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider, height_size, cpu, track, smooth,image_path):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    for img_num,img in enumerate(image_provider):
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                    #pose_keypoints为18行2列的列表，存储18个关节点的xy坐标
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)




        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses

        count=0
        for pose in current_poses:
            # 骨架丢失
            # count = count + 1
            # if count==3 :
            #      continue
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)

        count = 0
        for pose in current_poses:
            # 骨架boudingbox丢失
            # count = count + 1
            # if count==3 :
            #     continue
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0),3)
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
#####----------------------------保存图片；并写入json文件 ：将名为 的图片的pose_entries个人的pose写入json文件
        if track == 0:
            frames_path = glob(os.path.join(''.join(image_path), '*.jpg'))
            frames_path.sort()

            path=''.join(frames_path[img_num])

            cv2.imwrite('output_images/'+os.path.basename(path),img,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
            #write_ck_json(''.join(os.path.basename(path)),current_poses)

        # 保存视频帧为单张预测图像
        if track==1:
            path=str(img_num)+".jpg"
            cv2.imwrite(os.path.join('output_images/',path), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])



        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1
    # 对生成的骨架/光流单视频帧合成
    jpg2mp4("output_images","output_video/output_skeleton.mp4")
    jpg2mp4("output_flow","output_video/output_flow.mp4")


def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)
# 清理output_images、output_flow、output_video、video2frames
def cleardir():
    del_files("output_images")
    del_files("output_flow")
    del_files("output_video")
    del_files("video2frames")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    # parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load("models/checkpoint_iter_370000.pth", map_location='cpu')
    load_state(net, checkpoint)

    cleardir()
#args.images是list类型，一维list
    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
        # 为了对视频生成光流 分解视频帧
        for i,img in enumerate(frame_provider):
            path = str(i) + ".jpg"
            cv2.imwrite(os.path.join('video2frames/', path), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        frame2flow.extract_flow('video2frames')
    else:
        args.track = 0
        # 对单帧图像，输出光流
        frame2flow.extract_flow(''.join(args.images))


    run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth,args.images)
