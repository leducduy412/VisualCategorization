import numpy as np
import torch
import kmeans_gpu
from ..configs import config as cfg

def build_codebook(descriptors, k=cfg.KMEANS_CLUSTERS):
    # Stack all the descriptors vertically into a numpy array for efficient processing
    features = np.vstack((descriptor for descriptor in descriptors)).astype(np.float32)

    # Convert the features to a PyTorch tensor and transfer it to GPU for faster computation
    dataset = torch.from_numpy(features).to(torch.device('cuda'))
    print('Starting clustering')
    
    # Perform K-means clustering to create 'k' clusters; 'k' is defined in the config file. 
    # This is used to construct the visual vocabulary for image categorization.
    centers, codes = kmeans_gpu.cluster(dataset, k)
    
    # Return the computed centers (visual words) back to CPU memory.
    return centers.cpu()


