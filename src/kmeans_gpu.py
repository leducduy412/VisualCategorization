import torch
import random
import sys
from ..configs import config as cfg

# Set devices for tensor operations; GPUs are used for faster computation.
device_gpu = cfg.DEVICE_GPU
device_cpu = cfg.DEVICE_CPU
chunk_size = cfg.CHUNK_SIZE  # Size of data chunks for processing on the GPU.

def random_init(dataset, num_centers):
    # Initialize the cluster centers randomly by selecting 'num_centers' data points from the dataset.
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    used = torch.zeros(num_points, dtype=torch.long)
    indices = torch.zeros(num_centers, dtype=torch.long)
    for i in range(num_centers):
        while True:
            cur_id = random.randint(0, num_points - 1)
            if used[cur_id] > 0:  # Ensure the same point is not selected twice.
                continue
            used[cur_id] = 1
            indices[i] = cur_id
            break
    indices = indices.to(device_gpu)
    centers = torch.gather(dataset, 0, indices.view(-1, 1).expand(-1, dimension))
    return centers  # Return the initial centers.

def compute_codes(dataset, centers):
    # Assign each data point to the nearest cluster center.
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)
    chunk_size = int(5e8 / num_centers)  # Define chunk size based on GPU memory availability.
    codes = torch.zeros(num_points, dtype=torch.long, device=device_gpu)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers ** 2, dim=1).view(1, -1)
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
    return codes  # Return the index of the nearest center for each data point.

def update_centers(dataset, codes, num_centers):
    # Update cluster centers as the mean of all points assigned to that cluster.
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    centers = torch.zeros(num_centers, dimension, dtype=torch.float, device=device_gpu)
    cnt = torch.zeros(num_centers, dtype=torch.float, device=device_gpu)
    centers.scatter_add_(0, codes.view(-1, 1).expand(-1, dimension), dataset)
    cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float, device=device_gpu))
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(num_centers, dtype=torch.float, device=device_gpu))
    centers /= cnt.view(-1, 1)
    return centers  # Return the updated centers.

def cluster(dataset, num_centers=cfg.NUM_CENTERS):
    # Perform K-means clustering on the dataset.
    centers = random_init(dataset, num_centers)
    codes = compute_codes(dataset, centers)
    num_iterations = 0
    while True:
        sys.stdout.write('.')
        sys.stdout.flush()
        num_iterations += 1
        centers = update_centers(dataset, codes, num_centers)
        new_codes = compute_codes(dataset, centers)
        if torch.equal(codes, new_codes) or num_iterations >= cfg.MAX_ITERATIONS:
            sys.stdout.write('\n')
            print('Converged in %d iterations' % num_iterations)
            break
        codes = new_codes
    return centers, codes  # Return the final centers and code assignments.

