import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_test,
    y_pred,
    class_names,
    figsize=(10, 7),
    fontsize=14,
    normalize=False,
    save_path=None,
    fmt=None,
    map=None,
    xlabel="Predicted label",
    ylabel="True label",
):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix.

    Arguments
    ---------
    y_test: list
        List of test labels
    y_pred: list
        List of predicted labels
    class_names: list
       An ordered list of class names, in the order corrsponding to the index of y_test and y_pred.
    figsize: tuple
       A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
       the second determining the vertical size. Defaults to (10,7).
    fontsize: int
       Font size for axes labels. Defaults to 14.
    normalize: boolean
        Normalize confusion matrix. Default to False
    save_path: string
        If provided will save the figure to the path of the string. Defaults to not saving
    fmt: string
        format string of numbers in tiles with sensible defaults
    map: array
        Used for permutations of order of classes, e.g. [0, 2, 1] will swap class 1 and 2.
        Length should equal number classes.
    xlabel: string
        x label of heatmap
    ylabel: string
        y label of heatmap

    Returns
    -------
    array, shape = [n_classes, n_classes]
       The resulting confusion matrix
    """
    if map is not None:
        map = np.array(map)
        y_test = map[y_test]
        y_pred = map[y_pred]
        class_names = [class_names[m] for m in map]

    c_matrix = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(c_matrix, index=class_names, columns=class_names)
    plt.figure(figsize=figsize)

    if normalize:
        df_cm = df_cm / df_cm.sum(axis=1)[:, None]
        fmt_ = ".0%"
    elif np.issubdtype(c_matrix.dtype, np.integer):
        fmt_ = "d"
    elif np.issubdtype(c_matrix.dtype, np.floating):
        fmt_ = ".2f"
    else:
        raise ValueError("Confusion matrix values must be subtype of np.integer or np.floating")

    fmt = fmt if fmt is not None else fmt_
    heatmap = sns.heatmap(
        df_cm, annot=True, fmt=fmt, cmap="Blues", cbar=False, annot_kws={"size": 12}
    )

    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontsize
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=fontsize
    )
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    return c_matrix
