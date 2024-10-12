import sys
sys.path.append('/home/llll/codes/TSN/')

import os
from tqdm import tqdm
from toolkit.visualization.bbox_utils import get_axis_aligned_bbox
from toolkit.datasets import DatasetFactory
import math
import numpy as np
import argparse
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

import PIL
import imageio
from io import BytesIO
import PIL.Image as Image


parser = argparse.ArgumentParser(description='Draw gifs for Single Object Tracking')
parser.add_argument('--video', default='Animal2', type=str, help='eval one special video')
parser.add_argument('--dataset', type=str, default='DTB70', help='dataset name')
parser.add_argument('--dataset_root', default='test_datasets_demo', type=str, help='test datasets root')
parser.add_argument('--tracker_result_dir', default='test_results', type=str, help='tracker result root')
parser.add_argument('--trackers', default=["TSN"], nargs='+')
parser.add_argument('--save_dir', default='vis_results', type=str, help='Save path')
parser.add_argument('--gt_draw', dest='gt_draw', default=False, action="store_true")
args = parser.parse_args()


color = ["red",  #GT
         "blue"]  #TSN
linewidth = 1.5


args.dataset_root = os.path.join(args.dataset_root, args.dataset)
args.tracker_result_dir = os.path.join(args.tracker_result_dir, args.dataset)
args.save_dir = os.path.join(args.save_dir, args.dataset)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=args.dataset_root, load_img=False)


for v_idx, video in tqdm(enumerate(dataset), leave=True):
    if args.video != '':
        if video.name != args.video:
            continue
        
    bboxes = []
    names = []
    gif_images = []

    for P in args.trackers:
        try:
            bboxes.append(pd.read_csv(os.path.join(args.tracker_result_dir, P, str(video.name) + ".txt"),
                                      sep='\t|,| ',
                                      header=None,
                                      names=['xmin', 'ymin', 'width', 'height'],
                                      engine='python'))
            names.append(P.split('/')[-1])
        except:
            try:
                name = str(video.name)
                bboxes.append(
                        pd.read_csv(os.path.join(args.tracker_result_dir, P, str(video.name)[:-2] + "_" + str(video.name)[-1] + ".txt"),
                                    sep='\t|,| ',
                                    header=None,
                                    names=['xmin', 'ymin', 'width', 'height'],
                                    engine='python'))
                names.append(P.split('/')[-1])
            except:
                print("Please check path")
                print("path1:{} or path2:{}".format(os.path.join(args.tracker_result_dir, P, str(video.name) + ".txt"),
                                                    os.path.join(args.tracker_result_dir, P, str(video.name)[:-2] + "_" + str(video.name)[-1] + ".txt")))
                exit()

    for idx, (img, gt_bbox) in tqdm(enumerate(video), leave=False):
        img = img[..., ::-1]
        plt.clf()
        plt.imshow(img)
        ax = plt.gca()
        
        for (n, bbox) in enumerate(bboxes):
            try:
                bbox = list(map(int, bbox.iloc[idx].values))
                if not any(map(math.isnan, bbox)):
                    ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                                color=color[n+1],
                                                fill=False,
                                                linewidth=linewidth))
            except:
                continue
            
        if len(gt_bbox) == 4:
            gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                       gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                       gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]

        if args.gt_draw and (not any(map(math.isnan, gt_bbox))):
            try:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                x1, y1 = int(cx - w / 2 + 1), int(cy - h / 2 + 1)
                x2, y2 = int(cx + w / 2 - 1), int(cy + h / 2 - 1)
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1 + 1, y2 - y1 + 1,
                                           color=color[0],
                                           fill=False,
                                           linewidth=linewidth,
                                           linestyle='solid'))
            except:
                continue
            
        if os.path.exists(args.save_dir) is False:
            os.makedirs(args.save_dir)
        # image_dir = os.path.join(args.save_dir, str(idx) + '.' + args.format)
        plt.axis('off')

        
        #申请缓冲地址
        #using buffer,great way!
        with BytesIO() as buffer_:
            #保存在内存中，而不是在本地磁盘，注意这个默认认为你要保存的就是plt中的内容
            # plt.savefig(buffer_, bbox_inches = 'tight',pad_inches = 0, dpi=300)
            plt.savefig(buffer_, bbox_inches = 'tight', pad_inches = 0, dpi=150)
            buffer_.seek(0)
            #用imageio从内存中读取
            image_data = imageio.imread(buffer_)
            gif_images.append(image_data)
            
        plt.cla()

    kwargs_write = {'fps':25}
    imageio.mimsave(f"{args.save_dir}/{video.name}.mp4",gif_images,  **kwargs_write)
    