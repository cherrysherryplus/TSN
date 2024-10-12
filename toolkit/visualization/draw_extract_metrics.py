import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import shutil


def draw_extract_metrics(success_ret, videos, attr, precision_ret=None,
        norm_precision_ret=None, msg=None):
       
    # success plot
    success = {}
    for tracker_name in success_ret.keys():
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        success[tracker_name] = np.mean(value)
        msg[attr][tracker_name] = {}
    for idx, (tracker_name, auc) in  \
            enumerate(sorted(success.items(), key=lambda x:x[1], reverse=True)):
        msg[attr][tracker_name]['auc'] = "%.3f"%(auc)

    # precision plot
    if precision_ret:
        precision = {}
        for tracker_name in precision_ret.keys():
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in  \
                enumerate(sorted(precision.items(), key=lambda x:x[1], reverse=True)):
            msg[attr][tracker_name]['pre'] = "%.3f"%(pre)

    # norm precision plot
    if norm_precision_ret:
        norm_precision = {}
        for tracker_name in precision_ret.keys():
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            norm_precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in  \
                enumerate(sorted(norm_precision.items(), key=lambda x:x[1], reverse=True)):
            msg[attr][tracker_name]['norm_pre'] = "%.3f"%(pre)
