import os
import sys
import json
import time
import argparse
import functools
sys.path.append("./")

from collections import OrderedDict
from glob import glob
from tqdm import tqdm
import pandas
from multiprocessing import Pool
from toolkit.datasets import UAV10Dataset,UAV20Dataset,DTBDataset,VisDrone2019Dataset,LaTOTfullDataset,LaTOTDataset,Small90Dataset,Small112Dataset
from toolkit.evaluation import OPEBenchmark
from toolkit.visualization import draw_success_precision
from toolkit.visualization import draw_extract_metrics

# 有时候不加vis会报错: 段错误,核心已转储
# python tools/eval.py --dataset DTB70 --trackers TSN --vis

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='Single Object Tracking Evaluation')
    parser.add_argument('--dataset_dir', default='test_datasets_demo',type=str, help='dataset root directory')
    parser.add_argument('--dataset', default='UAV10fps',type=str, help='dataset name')
    parser.add_argument('--tracker_result_dir',default='test_results', type=str, help='tracker result root')
    parser.add_argument('--trackers',default='', nargs='+')
    parser.add_argument('--vis', default='',dest='vis', action='store_true')
    parser.add_argument('--metrics', default=False,dest='metrics', action='store_true')
    parser.add_argument('--show_video_level', default=False,dest='show_video_level', action='store_true')
    parser.add_argument('--num', default=1, type=int, help='number of processes to eval')
    args = parser.parse_args()

    # tracker_dir = os.path.join(args.tracker_path, args.dataset)
    # trackers = glob(os.path.join(args.tracker_path,
    #                               args.dataset,
    #                               args.tracker_prefix+'*'))
    # trackers = [x.split('/')[-1] for x in trackers]
    # root = os.path.realpath(os.path.join(os.path.dirname(__file__),
    #                          '../testing_datasets'))
    # root = os.path.join(root, args.dataset)
    # trackers=args.tracker_prefix

    tracker_dir = os.path.join(args.tracker_result_dir, args.dataset)
    if args.trackers:
        trackers = args.trackers
    else:
        trackers = os.listdir(tracker_dir)
        trackers = [tracker for tracker in trackers if os.path.isdir(os.path.join(tracker_dir, tracker))]
        trackers = [x.split('/')[-1] for x in trackers]
        # DEBUG
        # startswith 'checkpoint'
        # trackers = [tracker for tracker in trackers if tracker.startswith('checkpoint')]
    print(trackers)
    root = os.path.join(args.dataset_dir, args.dataset)
    
    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    if 'UAV10fps' in args.dataset:
        dataset = UAV10Dataset(args.dataset, root)
    elif 'UAV20l' in args.dataset:
        dataset = UAV20Dataset(args.dataset, root)
    elif 'DTB70' in args.dataset:
        dataset = DTBDataset(args.dataset, root)
    elif 'VisDrone2019' in args.dataset:
        dataset = VisDrone2019Dataset(args.dataset, root)
    elif 'LaTOTfull' in args.dataset:
        dataset = LaTOTfullDataset(args.dataset, root)
    elif 'LaTOT' in args.dataset:
        dataset = LaTOTDataset(args.dataset, root)
    elif 'small90' in args.dataset:
        dataset = Small90Dataset(args.dataset, root)
    elif 'small112' == args.dataset:
        dataset = Small112Dataset(args.dataset, root)

    ## compute metrics
    dataset.set_tracker(tracker_dir, trackers)
    benchmark = OPEBenchmark(dataset)
    success_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
            trackers), desc='eval success', total=len(trackers), ncols=18):
            success_ret.update(ret)
    precision_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
            trackers), desc='eval precision', total=len(trackers), ncols=18):
            precision_ret.update(ret)
    norm_precision_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
            trackers), desc='eval norm precision', total=len(trackers), ncols=100):
            norm_precision_ret.update(ret)
    benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
            show_video_level=args.show_video_level)
    
    ## draw extract metrics
    if args.metrics:
        name = dataset.name.replace('_','@')
        draw_plots_dir = os.path.join('draw_jsons', name)
        if not os.path.isdir(draw_plots_dir):
            os.makedirs(draw_plots_dir)
        msg = OrderedDict({k:OrderedDict() for k in dataset.attr.keys()})
        for attr, videos in dataset.attr.items():
            draw_extract_metrics(success_ret,
                        videos=videos,
                        attr=attr,
                        precision_ret=precision_ret,
                        norm_precision_ret=norm_precision_ret,
                        msg=msg)
        filename = os.path.join(draw_plots_dir, f'{name}_evaluation_results.json')
        with open (filename, 'w') as f:
            json.dump(msg, f, indent=4)
    ##
    
    ## draw plots
    if args.vis:
        name = dataset.name
        if 'VisDrone2019' in args.dataset:
            name = 'VisDrone2020'
        for attr, videos in dataset.attr.items():
            draw_success_precision(success_ret,
                        name=name,
                        videos=videos,
                        attr=attr,
                        precision_ret=precision_ret,
                        norm_precision_ret=norm_precision_ret)
    ##