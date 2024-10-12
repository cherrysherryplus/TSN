from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import sys
sys.path.append('/home/llll/codes/TSN/')

import cv2
import torch
import numpy as np


from pysot.core.config import cfg
from pysot.models.model_builder_test_only import ModelBuilder
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.datasets import DTBDataset,UAV10Dataset,UAV20Dataset,VisDrone2019Dataset,LaTOTDataset,Small90Dataset,Small112Dataset
from toolkit.evaluation import OPEBenchmark
from pysot.tracker.siamcar_tracker import SiamCARTracker

import optuna
import logging


def rename(tune_dir,dataset,video,model_name,result):
    oldname = os.path.join(tune_dir, dataset, video, model_name)
    newname = os.path.join(tune_dir, dataset, video, "{:.3f}".format(result) + model_name)
    os.rename(oldname, newname)

def eval(dataset, tracker_name):
    # root = os.path.realpath(os.path.join(os.path.dirname(__file__),
    #                                      '../testing_dataset'))
    # root = os.path.join(root, dataset)
    tracker_dir = "./"
    trackers = [tracker_name]
    dataset.set_tracker(tracker_dir, trackers)
    benchmark = OPEBenchmark(dataset)
    eval_auc = benchmark.eval_success(tracker_name)
    auc = np.mean(list(eval_auc[tracker_name].values()))
    return auc

# fitness function
def objective(trial):
    # different params
    WINDOW_INFLUENCE = trial.suggest_uniform('window_influence', 0.000, 1.000)
    PENALTY_K = trial.suggest_uniform('penalty_k', 0.000, 1.000)
    LR = trial.suggest_uniform('scale_lr', 0.000, 1.000)
    hp = {'lr': LR, 'penalty_k': PENALTY_K, 'window_lr': WINDOW_INFLUENCE}
    # rebuild tracker
    tracker = SiamCARTracker(model, cfg)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    tracker_name = os.path.join('tune_results_siamcar_cam', args.dataset, model_name, model_name + \
                    '_wi-{:.3f}'.format(WINDOW_INFLUENCE) + \
                    '_pk-{:.3f}'.format(PENALTY_K) + \
                    '_lr-{:.3f}'.format(LR))
    # OPE tracking
    for v_idx, video in enumerate(dataset):
        toc = 0
        pred_bboxes = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append(pred_bbox)
            else:
                try:
                    outputs = tracker.track(img, hp)
                    pred_bbox = outputs['bbox']
                except Exception as e:
                    pred_bbox = pred_bboxes[-1]
                pred_bboxes.append(pred_bbox)
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            # if idx == 0:
            #     cv2.destroyAllWindows()
        toc /= cv2.getTickFrequency()
        # save results
        if not os.path.isdir(tracker_name):
            os.makedirs(tracker_name)
        result_path = os.path.join(tracker_name, '{}.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x]) + '\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx + 1, video.name, toc, idx / toc))
    auc = eval(dataset=dataset_eval, tracker_name=tracker_name)
    info = "{:s} window_influence: {:1.17f}, penalty_k: {:1.17f}, scale_lr: {:1.17f}, AUC: {:1.3f}".format(
        model_name, WINDOW_INFLUENCE, PENALTY_K, LR, auc)
    logging.getLogger().info(info)
    print(info)
    rename('tune_results_siamcar_cam', args.dataset, model_name, model_name + \
            '_wi-{:.3f}'.format(WINDOW_INFLUENCE) + \
            '_pk-{:.3f}'.format(PENALTY_K) + \
            '_lr-{:.3f}'.format(LR), auc)
    return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tuning for SiamCAR')
    parser.add_argument('--dataset_root', default='test_datasets_demo', type=str, help='datasetdir')
    parser.add_argument('--tracker', default='TSN', type=str, help='tracker name')
    parser.add_argument('--dataset', default='UAV20l', type=str, help='dataset')
    parser.add_argument('--config', default='experiments/siamcar_r50/config.yaml', type=str, help='config file')
    parser.add_argument('--snapshot', default='tools/snapshot/TSN_uav20l.pth', type=str, help='snapshot of models to eval')
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu id")
    args = parser.parse_args()

    torch.set_num_threads(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # load config
    cfg.merge_from_file(args.config)

    if args.dataset == "GOT-10k":
        root = os.path.join(args.dataset_root, "GOT-10k", "test")
    else:
        root = os.path.join(args.dataset_root, args.dataset)
    # create model
    model = ModelBuilder()
    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=root,
                                            load_img=False)
    
    if 'DTB' in args.dataset:
        dataset_eval = DTBDataset(args.dataset, root)
    elif 'UAV10fps' in args.dataset:
        dataset_eval = UAV10Dataset(args.dataset, root)
    elif 'UAV20l' in args.dataset:
        dataset_eval = UAV20Dataset(args.dataset, root)
    # elif 'UAV' in args.dataset:
    #     dataset_eval = UAVDataset(args.dataset, root)
    elif 'VisDrone2019' in args.dataset:
        dataset_eval = VisDrone2019Dataset(args.dataset, root)
    elif 'LaTOT' in args.dataset:
        dataset_eval = LaTOTDataset(args.dataset, root)
    elif 'small90' == args.dataset:
        dataset_eval = Small90Dataset(args.dataset, root)
    elif 'small112' == args.dataset:
        dataset_eval = Small112Dataset(args.dataset, root)
    
    tune_result = os.path.join('tune_results_siamcar_cam', args.dataset)
    if not os.path.isdir(tune_result):
        os.makedirs(tune_result)
    log_path = os.path.join(tune_result, (args.snapshot).split('/')[-1].split('.')[0] + '.log')
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.FileHandler(log_path))
    optuna.logging.enable_propagation()

    study = optuna.create_study(study_name=f'{args.dataset}_{args.tracker}',
                                direction='maximize',
                                storage='sqlite:///{}_{}.db'.format(args.dataset, args.tracker),
                                load_if_exists=True)
    study.optimize(objective, n_trials=500)
    print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))