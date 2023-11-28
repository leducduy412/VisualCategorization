import numpy as np
import torch
import kmeans_gpu


def build_codebook(descriptors, k=10):
    features = np.vstack((descriptor for descriptor in descriptors)).astype(np.float32)

    # K-means clustering with k = 1000, trong bài báo sử dụng k = 1000
    dataset = torch.from_numpy(features).to(torch.device('cuda'))
    print('Starting clustering')
    centers, codes = kmeans_gpu.cluster(dataset, k)
    return centers.cpu()


