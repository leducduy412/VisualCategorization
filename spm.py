import math
import numpy as np

from features_extraction import extract_features_one
from histogram_extraction import build_vocab_one


def extract_vocab_SPM(img, L, kmeans):
    h, w, _ = img.shape

    word_hist = []

    code_level_0, code_level_1, code_level_2 = [], [], []

    for level in range(L + 1):

        # Initialize step
        w_step = math.floor(w / (2 ** level))
        h_step = math.floor(h / (2 ** level))

        m, n = 0, 0
        for i in range(1, 2 ** level + 1):
            m = 0
            for j in range(1, 2 ** level + 1):
                des = extract_features_one(img[n:n + h_step, m:m + w_step])
                if des is None:
                    word_hist.append([0 for i in range(1000)])
                    continue
                hist = build_vocab_one(des, kmeans)
                word_hist.append(hist)
                m = m + w_step
            n = n + h_step

    # word_hist = np.array(word_hist)
    code_level_0 = 0.25 * np.asarray(word_hist[0]).flatten()
    code_level_1 = 0.25 * np.asarray(word_hist[1:5]).flatten()
    code_level_2 = 0.5 * np.asarray(word_hist[5:]).flatten()

    return np.concatenate((code_level_0, code_level_1, code_level_2))
