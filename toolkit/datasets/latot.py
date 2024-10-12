import os
import json
import numpy as np

from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video

# 只考虑了test
def loaddata():
    
    path='./test_datasets_demo/LaTOT'
    test_txt = 'test_datasets_demo/LaTOT/dataset-split/test.txt'
    with open(test_txt,'r') as f:
        test_name_list = f.read().splitlines()
        
    
    name_list=test_name_list
    name_list.sort()
    attrs = ["Scale Variation",
             "Fast Motion",
             "Out-of-View",
             "Illumination Variation",
             "Camera Motion",
             "Motion Blur",
             "Background Clutter",
             "Similar Object",
             "Partial Occlusion",
             "Full Occlusion",
             "Abrupt Motion",
             "Low Illumination"]
    b=[]
    for i in range(len(name_list)):
        b.append(name_list[i])
    c=[]
    
    for jj in range(len(name_list)):
        imgs=path+'/'+str(name_list[jj])+'/'+'img'
        txt=path+'/annos/'+str(name_list[jj])+'.txt'
        att_txt=path+'/annos/attr/'+str(name_list[jj])+'.txt'
        bbox=[]
        f = open(txt)               # 返回一个文件对象
        ff = open(att_txt)               # 返回一个文件对象
        file= f.readlines()
        file_att= ff.read().splitlines()[0]
        li=os.listdir(imgs)
        li.sort()
        for ii in range(len(file)):
            li[ii]=name_list[jj]+'/img/'+li[ii]
    
            line = file[ii].strip('\n').split(',')
            
            try:
                line[0]=int(line[0])
            except:
                line[0]=float(line[0])
            try:
                line[1]=int(line[1])
            except:
                line[1]=float(line[1])
            try:
                line[2]=int(line[2])
            except:
                line[2]=float(line[2])
            try:
                line[3]=int(line[3])
            except:
                line[3]=float(line[3])
            bbox.append(line)
            
        # print(len(bbox))
        if len(bbox)!=len(li):
            print (jj, name_list[jj], len(bbox), len(li))
        f.close()
        ff.close()
        attr_mask = list(map(int, file_att.split(',')))
        video_attr = [attrs[i] for i in range(len(attr_mask)) if attr_mask[i]]
        # print(attr_mask)
        # print(video_attr)
        c.append({'attr':video_attr,'gt_rect':bbox,'img_names':li,'init_rect':bbox[0],'video_dir':name_list[jj]})
        
    d=dict(zip(b,c))

    return d

class LaTOTVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False):
        super(LaTOTVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)
        # self.absent = np.array(absent, np.int8)

class LaTOTDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'LaTOT'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(LaTOTDataset, self).__init__(name, dataset_root)
        meta_data = loaddata()

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = LaTOTVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          meta_data[video]['attr'])

        # set attr
        attr = []
        for x in self.videos.values():
            attr += x.attr
        attr = set(attr)
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
        for x in attr:
            self.attr[x] = []
        for k, v in self.videos.items():
            for attr_ in v.attr:
                self.attr[attr_].append(k)