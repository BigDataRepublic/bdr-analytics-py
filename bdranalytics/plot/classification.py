import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

primary_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

default_names = ('negative', 'positive')


def compute_parallel_metric(metric, y_true, y_pred):

    thresholds = np.arange(0, 1, .02)

    return Parallel(n_jobs=-1)(
        delayed(metric)(
            y_true,
            y_pred > threshold
        )
        for threshold in thresholds
    ), thresholds


def plot_accuracy(y_true, y_pred):
    acc, thresholds = compute_parallel_metric(accuracy_score, y_true, y_pred)

    lower_baseline = sum(y_true) / len(y_true)
    upper_baseline = 1 - lower_baseline

    plt.plot([0, 1], [lower_baseline, lower_baseline], 'k--')
    plt.plot([0, 1], [upper_baseline, upper_baseline], 'k--')
    plt.plot(thresholds, acc)
    plt.title('Accuracy across thresholds')
    plt.xlabel('classifier threshold')
    plt.ylabel('accuracy')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])


def plot_f1_score(y_true, y_pred):
    f1s, thresholds = compute_parallel_metric(f1_score, y_true, y_pred)

    plt.plot(thresholds, f1s)
    plt.title('F1 score across thresholds')
    plt.xlabel('classifier threshold')
    plt.ylabel('F1 score')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])


def plot_confusion_matrix(
    y_true, y_pred_bin, target_names=default_names, normalize=False):

    c = confusion_matrix(y_true, y_pred_bin)

    if normalize:
        c = c / c.sum()
        fmt = '.3f'
    else:
        fmt = 'd'

    confusion = pd.DataFrame(c, index=target_names, columns=target_names)
    sns.heatmap(confusion, annot=True, fmt=fmt)
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title('Confusion matrix')
    plt.show()


def plot_roc_curve(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1,
                                     drop_intermediate=True)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label="ROC curve (area = {:.2f})".format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver-operating characteristic')
    plt.legend(loc="lower right")


def plot_pr_curve(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred,
                                                           pos_label=1)

    average_precision = average_precision_score(
        y_true, y_pred, average="micro")

    baseline = sum(y_true) / len(y_true)

    plt.plot(recall, precision,
             label="PR curve (area = {:.2f})".format(average_precision))
    plt.plot([0, 1], [baseline, baseline], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Precision-recall curve')
    plt.legend(loc="lower right")


def plot_benefits(y_true, y_pred, benefit_func=None, recalibrate=False,
                  ax=None):
    if benefit_func is None:
        def net_benefit(tpr, fpr):
            cost_fp, benefit_tp = (1, 1)  # equal weights
            n_positives = sum(y_true)
            n_tp = tpr * n_positives  # number of true positives (benefits)
            n_fp = fpr * len(
                y_true) - n_positives  # number of false positives (costs)
            fp_costs = n_fp * cost_fp
            tp_benefits = n_tp * benefit_tp
            return tp_benefits - fp_costs

        benefit_func = net_benefit

    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1,
                                     drop_intermediate=True)

    benefits = np.zeros_like(thresholds)
    for i, _ in enumerate(thresholds):
        benefits[i] = benefit_func(tpr[i], fpr[i])

    i_max = np.argmax(benefits)
    print(
        "max benefits: {:.0f} units on {:,} samples, "
        "tpr: {:.3f}, fpr: {:.3f}, threshold: {:.3f}"
        .format(
            benefits[i_max], len(y_true),
            tpr[i_max], fpr[i_max], thresholds[i_max]
        )
    )

    if ax is not None:
        ax1 = ax
    else:
        _, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax2.vlines(thresholds[i_max], 0, 1, linestyles='dashed')
    ax1.set_xlim([0, 1])
    ax1.plot(thresholds, benefits, c=primary_color)
    ax1.set_ylim([0, np.max(benefits)])
    ax2.plot(thresholds, tpr, 'g-')
    ax2.plot(thresholds, fpr, 'r-')
    ax2.set_ylim([0, 1])
    ax1.set_xlabel('classifier threshold')
    ax1.set_ylabel('units')
    ax2.set_ylabel('rate')
    ax2.legend(labels=['TP', 'FP'], loc="upper right")
    ax1.set_title('Benefits across thresholds')
    ax1.legend(labels=['benefit'], loc="center right")
    ax1.grid(1)
    ax2.grid(0)

    if recalibrate:
        y_pred_bin = (y_pred > thresholds[i_max]) * 1.
        return y_pred_bin


def subplot_evaluation_curves(y_true, y_pred, benefit_func=None,
                              figsize=(12, 12)):

    fig, axarr = plt.subplots(3, 2, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    plt.sca(axarr[0, 0])
    plot_roc_curve(y_true, y_pred)

    plt.sca(axarr[0, 1])
    plot_pr_curve(y_true, y_pred)

    plt.sca(axarr[1, 0])
    plot_accuracy(y_true, y_pred)

    plt.sca(axarr[1, 1])
    plot_f1_score(y_true, y_pred)

    plot_benefits(y_true, y_pred, ax=axarr[2, 0], benefit_func=benefit_func)
    plt.show()
