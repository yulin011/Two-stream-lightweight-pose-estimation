import os
import cv2
from glob import glob
import natsort

# outputpath精确到具体想要生成的mp4文件
def jpg2mp4(inputpath,outputpath):

    frames_path = glob(os.path.join(inputpath, '*.jpg'))
    frames_path=natsort.natsorted(frames_path)

    pic1 = cv2.imread(frames_path[0])  # 读取第一张图片
    fps = 20
    imgInfo = pic1.shape
    size = (imgInfo[1], imgInfo[0])
    video = cv2.VideoWriter(outputpath, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i,framepath in enumerate(frames_path):
        # print(framepath)
        img = cv2.imread(framepath)  # 读取图片
        video.write(img)   #写入视频
    video.release()
if __name__ == '__main__':
    jpg2mp4('input_frames',"output_video/source.mp4")