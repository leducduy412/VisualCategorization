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

class_name = ['city', 'face', 'green', 'house_building', 'house_indoor', 'office', 'sea']
dic_class_name = {'city': 1, 'face': 2, 'green': 3, 'house_building': 4, 'house_indoor': 5, 'office': 6, 'sea': 7}

path_train = cfg.TRAIN_PATH_DATASET2
path_test = cfg.TEST_PATH_DATASET2


def get_key(dic_class_name, val):
    for key, value in dic_class_name.items():
        if val == value:
            return key


def data_extraction():
    x_train, x_test, y_train, y_test = [], [], [], []
    for c in class_name:
        path_class = path_train + c
        paths = glob.glob(path_class + "/*")
        for p in paths:
            img = cv2.imread(p)
            if img.shape[:2] != (256, 256):
                img = cv2.resize(img, (256, 256))
            x_train.append(img)
            y_train.append(c)

    for c in class_name:
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
    x_train, x_test, y_train, y_test = data_extraction()
    _, x_train_ef = extract_features(x_train)
    _, x_test_ef = extract_features(x_test)
    codebook = build_codebook(x_train_ef, k=cfg.CODEBOOK_K_DATASET2)
    x_train_final = build_vocab(x_train_ef, codebook)
    x_test_final = build_vocab(x_test_ef, codebook)

    # Train model
    svm_model = SVM(kernel='linear', kernel_params={})

    x_train_final = xp.asarray(x_train_final)
    x_test_final = xp.asarray(x_test_final)
    y_tr, y_te = [], []
    for y in y_train:
        y_tr.append(dic_class_name[y])
    y_tr = xp.asarray(y_tr)
    for y in y_test:
        y_te.append(dic_class_name[y])
    y_te = xp.asarray(y_te)
    svm_model.fit(x_train_final, y_tr)
    y_pred = svm_model.predict(x_test_final)
    count = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] == y_te[i]:
            count = count + 1
    accuracy = count / y_pred.shape[0]
    print(accuracy)

    # Confusion matrix
    y_pred_array = []

    for y in y_pred:
        y_pred_array.append(get_key(dic_class_name, int(y)))

    y_te_array = []
    for y in y_te:
        y_te_array.append(get_key(dic_class_name, int(y)))

    cnf_matrix = confusion_matrix(np.array([y_te_array]).T, y_pred_array, labels=class_name)
    np.set_printoptions(precision=2)

    plt.figure(figsize=(30, 10))
    plotting_confusion_matrix(cnf_matrix, classes=class_name,
                              title='Confusion matrix, without normalization')


def SPM_data2():
    x_train, x_test, y_train, y_test = data_extraction()

    x_train_spm = []
    x_test_spm = []

    _, x_train_ef = extract_features(x_train)
    codebook = build_codebook(x_train_ef, k=1000)

    for img in x_train:
        x_train_spm.append(extract_vocab_SPM(img, L=cfg.SPM_L_DATASET2, kmeans=codebook))

    for img in x_test:
        x_test_spm.append(extract_vocab_SPM(img, L=cfg.SPM_L_DATASET2, kmeans=codebook))

    x_train_spm = xp.asarray(x_train_spm)
    x_test_spm = xp.asarray(x_test_spm)
    y_tr, y_te = [], []
    for y in y_train:
        y_tr.append(dic_class_name[y])
    y_tr = xp.asarray(y_tr)
    for y in y_test:
        y_te.append(dic_class_name[y])
    y_te = xp.asarray(y_te)
    # Train model
    svm_model_spm = SVM(kernel='linear', kernel_params={})
    svm_model_spm.fit(x_train_spm, y_tr)
    y_pred_spm = svm_model_spm.predict(x_test_spm)
    count_spm = 0
    for i in range(y_pred_spm.shape[0]):
        if y_pred_spm[i] == y_te[i]:
            count_spm = count_spm + 1
    accuracy_spm = count_spm / y_pred_spm.shape[0]
    print(accuracy_spm)

    # Confusion matrix
    y_pred_spm_array = []

    for y in y_pred_spm:
        y_pred_spm_array.append(get_key(dic_class_name, int(y)))

    y_te_spm_array = []
    for y in y_te:
        y_te_spm_array.append(get_key(dic_class_name, int(y)))

    cnf_matrix = confusion_matrix(np.array([y_te_spm_array]).T, y_pred_spm_array, labels=class_name)
    np.set_printoptions(precision=2)

    plt.figure(figsize=(30, 10))
    plotting_confusion_matrix(cnf_matrix, classes=class_name,
                              title='Confusion matrix, without normalization')


if __name__ == "__main__":
    SPM_data2()
