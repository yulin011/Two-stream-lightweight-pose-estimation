import os
from glob import glob
import cv2
import numpy as np
import natsort

def compute_farneback(prev, curr, hsv):
    flow = cv2.calcOpticalFlowFarneback(prev=prev,
                                        next=curr,
                                        flow=None,
                                        pyr_scale=0.5,
                                        levels=3,
                                        winsize=15,
                                        iterations=3,
                                        poly_n=5,
                                        poly_sigma=1.2,
                                        flags=0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 1] = 255.0
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # 返回成像的（rgb）的光流图片
    return flow_img
def compute_farneback_bcakgroudwhite(prev, curr, hsv) :
    # 色调H：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°
    # 饱和度S：取值范围为0.0～1.0
    # 亮度V：取值范围为0.0(黑色)～1.0(白色)
    flow = cv2.calcOpticalFlowFarneback(prev=prev,
                                        next=curr,
                                        flow=None,
                                        pyr_scale=0.5,
                                        levels=3,
                                        winsize=15,
                                        iterations=3,
                                        poly_n=5,
                                        poly_sigma=1.2,
                                        flags=0)
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # flownet是将V赋值为255, 此函数遵循flownet，饱和度S代表像素位移的大小，亮度都为最大，便于观看
    # 也有的光流可视化讲s赋值为255，亮度代表像素位移的大小，整个图片会很暗，很少这样用
    hsv[..., 2] = 255
    flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_img

def cal_for_frames(video_path):
    frames_path = glob(os.path.join(video_path, '*.jpg'))
    frames_path=natsort.natsorted(frames_path)

    flow = []
    prev = cv2.imread(frames_path[0])
    hsv_stage0 = np.zeros_like(prev)
    # 转换为8bit灰度图
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frames_name = []
    for i, frame_curr in enumerate(frames_path):
        curr = cv2.imread(frame_curr)
        hsv_staget = np.zeros_like(curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        tmp_flow = compute_farneback_bcakgroudwhite(prev, curr, hsv_stage0)
        flow.append(tmp_flow)

        prev = curr
        hsv_stage0 = hsv_staget

        # 提出每帧图片名字
        frames_name.append(os.path.basename(frame_curr))

    return flow, frames_name


def save_flow(flows, flow_path, frames_name):
    for i, flow in enumerate(flows):
        cv2.imwrite(os.path.join(flow_path, frames_name[i]), flow)


def extract_flow(video_paths):
    # video_paths = "/Users/wusilin/Documents/myproject/pycharm_project/frames2flow/input_frames"
    #光流默认保存路径
    flow_paths = "/Users/wusilin/Documents/myproject/GitHub/lightweight-human-pose-estimation.pytorch/output_flow"

    # 提出可视化光流图片
    flow, frames_name = cal_for_frames(video_paths)

    save_flow(flow, flow_paths, frames_name)