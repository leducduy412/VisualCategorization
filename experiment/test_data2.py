import glob

import cv2
import numpy as np
import cupy as xp
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from ..configs import config as cfg
from codebook_extraction import build_codebook
from confusion_matrix import plotting_confusion_matrix
from features_extraction import extract_features
from histogram_extraction import build_vocab
from spm import extract_vocab_SPM
from svm_gpu import SVM

# Class names for the second dataset.
class_name = ['city', 'face', 'green', 'house_building', 'house_indoor', 'office', 'sea']
dic_class_name = {'city': 1, 'face': 2, 'green': 3, 'house_building': 4, 'house_indoor': 5, 'office': 6, 'sea': 7}

path_train = cfg.TRAIN_PATH_DATASET2
path_test = cfg.TEST_PATH_DATASET2


def get_key(dic_class_name, val):
    """Helper function to get class name from its numerical label."""
    for key, value in dic_class_name.items():
        if val == value:
            return key

def data_extraction():
    """
    Extract and preprocess images and labels from the dataset.
    Reads images from disk, resizes them, and compiles them into training and testing sets.
    """
    x_train, x_test, y_train, y_test = [], [], [], []
    # Loop through each class and load images.
    for c in class_name:
        # Extract training data.
        path_class = path_train + c
        paths = glob.glob(path_class + "/*")
        for p in paths:
            img = cv2.imread(p)
            if img.shape[:2] != (256, 256):
                img = cv2.resize(img, (256, 256))
            x_train.append(img)
            y_train.append(c)

        # Extract testing data.
        path_class = path_test + c
        paths = glob.glob(path_class + "/*")
        for p in paths:
            img = cv2.imread(p)
            if img.shape[:2] != (256, 256):
                img = cv2.resize(img, (256, 256))
            x_test.append(img)
            y_test.append(c)
    return x_train, x_test, y_train, y_test

def BoK_data2():
    """
    Test the second dataset using the Bag of Keypoints (BoK) approach and SVM classifier.
    This involves extracting features, creating a codebook, building histograms, and classifying with SVM.
    """
    # Data extraction and feature processing.
    x_train, x_test, y_train, y_test = data_extraction()
    _, x_train_ef = extract_features(x_train)
    _, x_test_ef = extract_features(x_test)
    codebook = build_codebook(x_train_ef, k=cfg.CODEBOOK_K_DATASET2)
    x_train_final = build_vocab(x_train_ef, codebook)
    x_test_final = build_vocab(x_test_ef, codebook)

    # Initialize and train the SVM model.
    svm_model = SVM(kernel='linear', kernel_params={})
    x_train_final = xp.asarray(x_train_final)
    x_test_final = xp.asarray(x_test_final)
    y_tr, y_te = [dic_class_name[y] for y in y_train], [dic_class_name[y] for y in y_test]
    svm_model.fit(x_train_final, xp.asarray(y_tr))
    y_pred = svm_model.predict(x_test_final)

    # Calculate and print accuracy.
    accuracy = np.mean(y_pred == xp.asarray(y_te))
    print(accuracy)

    # Generate and visualize the confusion matrix.
    y_pred_array = [get_key(dic_class_name, int(y)) for y in y_pred]
    y_te_array = [get_key(dic_class_name, int(y)) for y in y_te]
    cnf_matrix = confusion_matrix(np.array([y_te_array]).T, y_pred_array, labels=class_name)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(30, 10))
    plotting_confusion_matrix(cnf_matrix, classes=class_name, title='Confusion matrix, without normalization')

def SPM_data2():
    """
    Test the second dataset using the Spatial Pyramid Matching (SPM) approach and SVM classifier.
    This involves extracting features, creating a codebook, building histograms with SPM, and classifying with SVM.
    """
    # Data extraction and feature processing.
    x_train, x_test, y_train, y_test = data_extraction()
    _, x_train_ef = extract_features(x_train)
    codebook = build_codebook(x_train_ef, k=cfg.SPM_K_DATASET2)

    # Apply SPM to both training and testing images.
    x_train_spm = [extract_vocab_SPM(img, L=cfg.SPM_L_DATASET2, kmeans=codebook) for img in x_train]
    x_test_spm = [extract_vocab_SPM(img, L=cfg.SPM_L_DATASET2, kmeans=codebook) for img in x_test]

    # Initialize and train the SVM model.
    svm_model_spm = SVM(kernel='linear', kernel_params={})
    y_tr, y_te = [dic_class_name[y] for y in y_train], [dic_class_name[y] for y in y_test]
    svm_model_spm.fit(xp.asarray(x_train_spm), xp.asarray(y_tr))
    y_pred_spm = svm_model_spm.predict(xp.asarray(x_test_spm))

    # Calculate and print accuracy.
    accuracy_spm = np.mean(y_pred_spm == xp.asarray(y_te))
    print(accuracy_spm)

    # Generate and visualize the confusion matrix.
    y_pred_spm_array = [get_key(dic_class_name, int(y)) for y in y_pred_spm]
    y_te_spm_array = [get_key(dic_class_name, int(y)) for y in y_te]
    cnf_matrix = confusion_matrix(np.array([y_te_spm_array]).T, y_pred_spm_array, labels=class_name)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(30, 10))
    plotting_confusion_matrix(cnf_matrix, classes=class_name, title='Confusion matrix, without normalization')

if __name__ == "__main__":
    SPM_data2()
