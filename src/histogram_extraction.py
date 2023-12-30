import numpy as np
from scipy.cluster.vq import kmeans, vq
from ..configs import config as cfg

def build_vocab_one(descriptor, codebook):
    # Vector quantization
    code, _ = vq(descriptor, codebook)
    word_hist, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1), normed=cfg.HIST_NORMED)

    return word_hist


def build_vocab(descriptors, codebook):
    word_hists = []
    # Vector quantization
    for descriptor in descriptors:
        code, _ = vq(descriptor, codebook)
        word_hist, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
        word_hists.append(word_hist)

    return word_hists
