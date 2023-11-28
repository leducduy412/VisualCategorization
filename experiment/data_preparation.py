import cv2
import json

p = open('path.txt', 'r').read()


def prepare_data_train(path='dataset1/',
                       split='dataset1/split_zhou_Caltech101.json'):
    path = p + path
    with open(split, 'r') as d:
        data = json.load(d)
        train = data['train']

    x_train, y_train = [], []

    for tr in train:
        y_train.append(tr[2])
        train_path = path + tr[0]
        train_img = cv2.imread(train_path)
        if train_img.shape[:2] != (256, 256):
            train_img = cv2.resize(train_img, (256, 256))
        x_train.append(train_img)

    return [x_train, y_train]


def prepare_data_val(path='dataset1/',
                     split='dataset1/split_zhou_Caltech101.json'):
    path = p + path
    with open(split, 'r') as d:
        data = json.load(d)
        val = data['val']

    x_val, y_val = [], []

    for va in val:
        y_val.append(va[2])
        val_path = path + va[0]
        val_img = cv2.imread(val_path)
        if val_img.shape[:2] != (256, 256):
            val_img = cv2.resize(val_img, (256, 256))
        x_val.append(val_img)

    return [x_val, y_val]


def prepare_data_test(path='dataset1/',
                      split='dataset1/split_zhou_Caltech101.json'):
    path = p + path
    with open(split, 'r') as d:
        data = json.load(d)
        test = data['experiment']

    x_test, y_test = [], []

    for te in test:
        y_test.append(te[2])
        test_path = path + te[0]
        test_img = cv2.imread(test_path)
        if test_img.shape[:2] != (256, 256):
            test_img = cv2.resize(test_img, (256, 256))
        x_test.append(test_img)

    return [x_test, y_test]
