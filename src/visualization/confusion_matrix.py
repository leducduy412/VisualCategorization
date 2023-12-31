import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from ..configs import config as cfg

def plotting_confusion_matrix(cm, classes,
                              normalize=False,
                              title=cfg.CONF_MATRIX_TITLE,
                              cmap=cfg.CONF_MATRIX_CMAP):
    # If 'normalize' is True, convert the confusion matrix to proportions instead of raw counts.
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Visualize the confusion matrix using a color-coded heatmap.
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    # Set the ticks and labels for the confusion matrix.
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Determine the text format inside the confusion matrix cells.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Label each cell in the confusion matrix with its corresponding count/proportion.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # Adjust layout for better readability and add axis labels.
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
