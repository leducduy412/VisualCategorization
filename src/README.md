## Directory Structure

Below is the directory structure of the `src/` folder, explaining the purpose of each sub-directory and file within it:

### `configs/`
Contains the central configuration file for the entire project.
- `config.py`: Defines configuration variables and parameters used across various scripts.

### `data/`
Contains scripts related to data handling, including preparation and testing.
- `prepare/`: Contains scripts for data preparation.
  - `data_preparation.py`: Prepares data for training, testing, and validation.
- `test/`: Contains scripts for testing the models on datasets.
  - `test_data1.py`: Tests the first dataset using SPM and SVM.
  - `test_data2.py`: Tests the second dataset using BoK and SPM with SVM.

### `models/`
Contains the machine learning models used in the project.
- `svm/`: Related to the Support Vector Machine model.
  - `svm_gpu.py`: Defines and trains an SVM model using GPU acceleration.
- `nb/`: Related to Naive Bayes models.
  - `find_C.py`: Finds the best value for the parameter C.
  - `find_k.py`: Determines the best number of visual words (k).

### `processing/`
Contains scripts for data processing, including feature extraction and matching.
- `codebook/`: Related to codebook creation from features.
  - `codebook_extraction.py`: Constructs a codebook using k-means clustering.
- `features/`: Related to feature extraction and histogram building.
  - `features_extraction.py`: Extracts features from images.
  - `histogram_extraction.py`: Builds feature histograms from the codebook.
- `clustering/`: Contains clustering methods like k-means.
  - `kmeans_gpu.py`: Performs k-means clustering using GPU.
- `matching/`: Contains matching methods such as SPM.
  - `spm.py`: Performs Spatial Pyramid Matching.

### `visualization/`
Contains scripts for visualization and result presentation.
- `confusion_matrix.py`: Visualizes confusion matrices to evaluate classification results.

