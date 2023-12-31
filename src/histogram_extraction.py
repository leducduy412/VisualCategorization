import numpy as np
from scipy.cluster.vq import kmeans, vq
from ..configs import config as cfg

def build_vocab_one(descriptor, codebook):
    # Perform vector quantization of the descriptors using the codebook.
    # Each feature is assigned to the closest code (visual word) in the codebook.
    code, _ = vq(descriptor, codebook)

    # Create a histogram of the code occurrences. The histogram bins correspond to the visual words.
    # The 'normed' parameter is used to normalize the histogram if specified in the configuration.
    word_hist, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1), normed=cfg.HIST_NORMED)

    # Return the histogram which represents the image's visual word occurrences.
    return word_hist

def build_vocab(descriptors, codebook):
    word_hists = []  # Initialize an empty list to store histograms of all images.

    # Iterate through each descriptor (from all images) and perform vector quantization.
    for descriptor in descriptors:
        code, _ = vq(descriptor, codebook)

        # Generate and store the histogram for each image's descriptors.
        word_hist, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
        word_hists.append(word_hist)

    # Return the list of histograms corresponding to all images.
    return word_hists

