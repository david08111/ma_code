from matplotlib import pyplot as plt


def plot_pr_curve(precision, recall):
    sorted_precision_list = sorted(precision.items())
    thresholds, precision = zip(*sorted_precision_list)

    sorted_recall_list = sorted(recall.items())
    thresholds, recall = zip(*sorted_recall_list)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title("Precision/Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.plot(recall, precision)
    return fig


def plot_roc_curve(true_pos_rate, false_pos_rate):
    sorted_tpr_list = sorted(true_pos_rate.items())
    thresholds, true_pos_rate = zip(*sorted_tpr_list)

    sorted_fpr_list = sorted(false_pos_rate.items())
    thresholds, false_pos_rate = zip(*sorted_fpr_list)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title("ROC Curve")
    ax.set_xlabel("False_pos_rate")
    ax.set_ylabel("True_pos_rate")
    ax.plot(false_pos_rate, true_pos_rate)
    return fig

