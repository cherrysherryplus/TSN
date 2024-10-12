import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import shutil

from .draw_utils import COLOR, LINE_STYLE


def draw_success_precision(success_ret, name, videos, attr, precision_ret=None,
        norm_precision_ret=None, bold_name='TSN', axis=[0, 1]):
    if '_' in name:
        name = name.replace('_','@')
    draw_plots_dir = osp.join('draw_plots',name)
    if not osp.isdir(draw_plots_dir):
        os.makedirs(draw_plots_dir)
        # shutil.rmtree(draw_plots_dir)
    
    # success plot
    plt.rc('text', usetex=True)
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots()
    ax.grid(b=True)
    ax.set_aspect(1)
    plt.xlabel('Overlap threshold', fontsize=14)
    plt.ylabel('Success rate', fontsize=14)
    if attr == 'ALL':
        img_title = f'Success plots of OPE on {name}'
        plt.title(r'\textbf{Success plots of OPE on %s}' % (name), fontsize=12, loc='center')
    else:
        img_title = f'Success plots of OPE - {attr}'
        plt.title(r'\textbf{Success plots of OPE - %s}' % (attr), fontsize=12, loc='center')
    plt.axis([0, 1]+axis)
    success = {}
    thresholds = np.arange(0, 1.05, 0.05)
    for tracker_name in success_ret.keys():
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        success[tracker_name] = np.mean(value)
    for idx, (tracker_name, auc) in  \
            enumerate(sorted(success.items(), key=lambda x:x[1], reverse=True)):
        if tracker_name == bold_name:
            label = r"\textbf{[%.3f] %s}" % (auc, tracker_name)
        else:
            label = "[%.3f] " % (auc) + tracker_name
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        # plt.plot(thresholds, np.mean(value, axis=0),
        #         color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
        plt.plot(thresholds, np.mean(value, axis=0), 
                color=COLOR[idx % 10], linestyle=LINE_STYLE[divmod(idx, 10)[0]], label=label, linewidth=2)
    # ax.legend(loc='lower left', labelspacing=0.2)
    ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0, fontsize=11, frameon=False)
    ax.autoscale(enable=True, axis='both', tight=True)
    xmin, xmax, ymin, ymax = plt.axis()
    ax.autoscale(enable=False)
    ymax += 0.03
    ymin = np.around(ymin, decimals=2)
    ymax = np.around(ymax, decimals=2)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xticks(np.arange(xmin, xmax+0.01, 0.1), fontsize=14)
    plt.yticks(np.arange(ymin, ymax,0.1), fontsize=14)
    ax.set_aspect((xmax - xmin)/(ymax-ymin))
    # plt.show()
    plt.tight_layout()
    fig.savefig(f'{draw_plots_dir}/{img_title}.png',dpi=300)

    if precision_ret:
        # norm precision plot
        plt.rc('text', usetex=True)
        plt.rc('font', family='Times New Roman')
        fig, ax = plt.subplots()
        ax.grid(b=True)
        ax.set_aspect(50)
        plt.xlabel('Location error threshold', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        if attr == 'ALL':
            img_title = f'Precision plots of OPE on {name}'
            plt.title(r'\textbf{Precision plots of OPE on %s}' % (name), fontsize=12, loc='center')
        else:
            img_title = f'Precision plots of OPE on {attr}'
            plt.title(r'\textbf{Precision plots of OPE - %s}' % (attr), fontsize=12, loc='center')
        plt.axis([0, 50]+axis)
        precision = {}
        thresholds = np.arange(0, 51, 1)
        for tracker_name in precision_ret.keys():
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            # plt.plot(thresholds, np.mean(value, axis=0),
            #         color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
            plt.plot(thresholds, np.mean(value, axis=0), 
                    color=COLOR[idx % 10], linestyle=LINE_STYLE[divmod(idx, 10)[0]], label=label, linewidth=2)
        # ax.legend(loc='lower right', labelspacing=0.2)
        ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0, fontsize=11, frameon=False)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        ymin = np.around(ymin, decimals=2)
        ymax = np.around(ymax, decimals=2)
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 5), fontsize=14)
        plt.yticks(np.arange(ymin, ymax, 0.1), fontsize=14)
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        # plt.show()
        plt.tight_layout()
        fig.savefig(f'{draw_plots_dir}/{img_title}.png',dpi=300)

    # norm precision plot
    if norm_precision_ret:
        plt.rc('text', usetex=True)
        plt.rc('font', family='Times New Roman')
        fig, ax = plt.subplots()
        ax.grid(b=True)
        plt.xlabel('Location error threshold', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        if attr == 'ALL':
            img_title = f'Normalized Precision plots of OPE on {name}'
            plt.title(r'\textbf{Normalized Precision plots of OPE on %s}' % (name), fontsize=12, loc='center')
        else:
            img_title = f'Normalized Precision plots of OPE - {attr}'
            plt.title(r'\textbf{Normalized Precision plots of OPE - %s}' % (attr), fontsize=12, loc='center')
        norm_precision = {}
        thresholds = np.arange(0, 51, 1) / 100
        for tracker_name in precision_ret.keys():
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            norm_precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(norm_precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name.replace('_','-'))
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            # plt.plot(thresholds, np.mean(value, axis=0),
            #         color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
            plt.plot(thresholds, np.mean(value, axis=0), 
                    color=COLOR[idx % 10], linestyle=LINE_STYLE[divmod(idx, 10)[0]], label=label, linewidth=2)
        # ax.legend(loc='lower right', labelspacing=0.2)
        ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0, fontsize=11, frameon=False)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        ymin = np.around(ymin, decimals=2)
        ymax = np.around(ymax, decimals=2)
        plt.axis([xmin, xmax, ymin, ymax])
        # plt.xticks(np.arange(xmin, xmax+0.01, 0.05), fontsize=14)
        plt.xticks(np.arange(xmin, xmax+0.01, 0.1), fontsize=14)
        plt.yticks(np.arange(ymin, ymax, 0.1), fontsize=14)
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        # plt.show()
        plt.tight_layout()
        fig.savefig(f'{draw_plots_dir}/{img_title}.png',dpi=300)
        
