# config.py

# Codebook Extraction
KMEANS_CLUSTERS = 1000  # Number of clusters used in K-means

# Confusion Matrix Plotting
CONF_MATRIX_TITLE = 'Confusion matrix'  # Default title for confusion matrix
CONF_MATRIX_CMAP = plt.cm.Blues  # Default color map

# Features Extraction
HARRIS_BLOCKSIZE = 2
HARRIS_KSIZE = 3
HARRIS_K = 0.04
NORM_ALPHA = 0
NORM_BETA = 255
THRESHOLD_RATIO = 0.01

# Histogram Extraction
HIST_BINS = None
HIST_NORMED = True

# K-means Clustering
NUM_CENTERS = 10  
CHUNK_SIZE = int(5e8 / NUM_CENTERS)  
MAX_ITERATIONS = 300  
DEVICE_GPU = torch.device('cuda')
DEVICE_CPU = torch.device('cpu')

# Spatial Pyramid Matching
SPM_L = 2  # Number of levels in Spatial Pyramid
LEVEL_0_WEIGHT = 0.25
LEVEL_1_WEIGHT = 0.25
LEVEL_2_WEIGHT = 0.5
HIST_BINS = 1000  # Number of bins in the histogram

# SVM Configurations
LAMBDUH = 1
MAX_ITER = 1000
CLASSIFICATION_STRATEGY = 'ovr'
N_FOLDS = 3
LAMBDA_VALS = [10 ** i for i in range(-3, 4)]
USE_OPTIMAL_LAMBDA = False
DISPLAY_PLOTS = False
LOGGING = False

# Kernel Configurations
KERNEL_SIGMA = 1  # Use for RBF
KERNEL_DEGREE = 3  # Used for polynomial kernel
ALPHAPARAM = 0.5
BETAPARAM = 0.8

# Data Preparation Configurations
BASE_PATH = 'D:/PyCharmProjects/BoK_VisualCategorization/'
DATASET_ROOT = BASE_PATH + 'dataset/'
TRAIN_DATASET_PATH = 'dataset1/'
VAL_DATASET_PATH = 'dataset1/'
TEST_DATASET_PATH = 'dataset1/'
TRAIN_SPLIT_FILE = DATASET_ROOT + 'dataset1/split_zhou_Caltech101.json'
VAL_SPLIT_FILE = DATASET_ROOT + 'dataset1/split_zhou_Caltech101.json'
TEST_SPLIT_FILE = DATASET_ROOT + 'dataset1/split_zhou_Caltech101.json'
TARGET_IMAGE_SIZE = (256, 256)

# SVM Configurations
CODEBOOK_PATH = 'D:/PyCharmProjects/BoK_VisualCategorization/dataset1/codebook.pkl'
C_VALUES_RANGE = (0.0001, 0.1, 0.00198)  # (start, stop, step)

# Codebook and Feature Extraction Configurations
K_VALUES_RANGE = (1800, 2500, 100) 

# SVM and SPM Configurations
SPM_L = 2  # Levels in the spatial pyramid
SVM_KERNEL = 'linear'  # Kernel type for SVM
SVM_LAMBDUH = 0.0001  # Regularization parameter for SVM

# Data paths for Dataset2
TRAIN_PATH_DATASET2 = BASE_PATH + 'dataset/dataset2/train/'
TEST_PATH_DATASET2 = BASE_PATH + 'dataset/dataset2/test/'

# Codebook and Feature Extraction Configurations for Dataset2
CODEBOOK_K_DATASET2 = 1000  # Number of visual words for codebook
SPM_L_DATASET2 = 4  # Levels in the spatial pyramid for Dataset2
