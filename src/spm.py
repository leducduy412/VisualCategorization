import math
import numpy as np
from ..configs import config as cfg
from features_extraction import extract_features_one
from histogram_extraction import build_vocab_one

def extract_vocab_SPM(img, L, kmeans):
    # Obtain the dimensions of the image.
    h, w, _ = img.shape

    # Initialize variables to store the histograms for each level.
    word_hist = []
    code_level_0, code_level_1, code_level_2 = [], [], []

    # Loop through each level of the pyramid.
    for level in range(cfg.SPM_L + 1):

        # Calculate the step size for the current level. The image is divided into 2^level parts on each axis.
        w_step = math.floor(w / (2 ** level))
        h_step = math.floor(h / (2 ** level))

        # Initialize the starting points for the grid.
        m, n = 0, 0
        for i in range(1, 2 ** level + 1):
            m = 0
            for j in range(1, 2 ** level + 1):
                # Extract features from the specific part of the image.
                des = extract_features_one(img[n:n + h_step, m:m + w_step])
                if des is None:
                    # If no descriptors are found, append a zero histogram.
                    word_hist.append([0 for i in range(cfg.HIST_BINS)])
                    continue
                # Build a histogram (visual word occurrences) for the descriptors.
                hist = build_vocab_one(des, kmeans)
                word_hist.append(hist)
                # Move to the next part on the same level.
                m = m + w_step
            # Move to the next row of parts on the same level.
            n = n + h_step

    # Calculate the weighted histograms for each level. Weights are defined in the configuration.
    # These weights are used to balance the contribution of each level to the final feature vector.
    code_level_0 = cfg.LEVEL_0_WEIGHT * np.asarray(word_hist[0]).flatten()
    code_level_1 = cfg.LEVEL_1_WEIGHT * np.asarray(word_hist[1:5]).flatten()
    code_level_2 = cfg.LEVEL_2_WEIGHT * np.asarray(word_hist[5:]).flatten()

    # Concatenate the weighted histograms from each level to form the final feature vector.
    return np.concatenate((code_level_0, code_level_1, code_level_2))

