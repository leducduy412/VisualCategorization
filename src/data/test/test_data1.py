import pickle
import cupy as xp
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from ..configs import config as cfg
from confusion_matrix import plotting_confusion_matrix
from experiment.data_preparation import prepare_data_train, prepare_data_test, prepare_data_val
from spm import extract_vocab_SPM
from svm_gpu import SVM

# Dictionary mapping class names to numerical labels.
class_name = ['accordion', 'airplanes', 'anchor', 'ant', 'barrel', 'bass',
              'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus',
              'buddha', 'butterfly', 'camera', 'cannon', 'car_side',
              'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body',
              'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head',
              'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly',
              'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'face',
              'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk',
              'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog',
              'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo',
              'ketch', 'lamp', 'laptop', 'leopard', 'llama', 'lobster', 'lotus',
              'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'motorbike',
              'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza',
              'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone',
              'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball',
              'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry',
              'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly',
              'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']

dic_class_name = {'accordion': 1, 'airplane': 2, 'anchor': 3, 'ant': 4, 'barrel': 5, 'bass': 6,
                  'beaver': 7, 'binocular': 8, 'bonsai': 9, 'brain': 10, 'brontosaurus': 11,
                  'buddha': 12, 'butterfly': 13, 'camera': 14, 'cannon': 15, 'car_side': 16,
                  'ceiling_fan': 17, 'cellphone': 18, 'chair': 19, 'chandelier': 20, 'cougar_body': 21,
                  'cougar_face': 22, 'crab': 23, 'crayfish': 24, 'crocodile': 25, 'crocodile_head': 26,
                  'cup': 27, 'dalmatian': 28, 'dollar_bill': 29, 'dolphin': 30, 'dragonfly': 31,
                  'electric_guitar': 32, 'elephant': 33, 'emu': 34, 'euphonium': 35, 'ewer': 36, 'face': 37,
                  'ferry': 38, 'flamingo': 39, 'flamingo_head': 40, 'garfield': 41, 'gerenuk': 42,
                  'gramophone': 43, 'grand_piano': 44, 'hawksbill': 45, 'headphone': 46, 'hedgehog': 47,
                  'helicopter': 48, 'ibis': 49, 'inline_skate': 50, 'joshua_tree': 51, 'kangaroo': 52,
                  'ketch': 53, 'lamp': 54, 'laptop': 55, 'leopard': 56, 'llama': 57, 'lobster': 58, 'lotus': 59,
                  'mandolin': 60, 'mayfly': 61, 'menorah': 62, 'metronome': 63, 'minaret': 64, 'motorbike': 65,
                  'nautilus': 66, 'octopus': 67, 'okapi': 68, 'pagoda': 69, 'panda': 70, 'pigeon': 71, 'pizza': 72,
                  'platypus': 73, 'pyramid': 74, 'revolver': 75, 'rhino': 76, 'rooster': 77, 'saxophone': 78,
                  'schooner': 79, 'scissors': 80, 'scorpion': 81, 'sea_horse': 82, 'snoopy': 83, 'soccer_ball': 84,
                  'stapler': 85, 'starfish': 86, 'stegosaurus': 87, 'stop_sign': 88, 'strawberry': 89,
                  'sunflower': 90, 'tick': 91, 'trilobite': 92, 'umbrella': 93, 'watch': 94, 'water_lilly': 95,
                  'wheelchair': 96, 'wild_cat': 97, 'windsor_chair': 98, 'wrench': 99, 'yin_yang': 100}


def SPM_data1():
    """
    Test the first dataset using Spatial Pyramid Matching (SPM) and a Support Vector Machine (SVM).
    The process involves training and testing the SVM on features extracted via SPM.
    """
    # Prepare training, testing, and validation data.
    x_train, y_train = prepare_data_train()
    x_test, y_test = prepare_data_test()
    x_val, y_val = prepare_data_val()

    # Combine training and validation sets for a more comprehensive training dataset.
    x_train = x_train + x_val
    y_train = y_train + y_val

    # Load the codebook generated from the training data.
    with open(cfg.CODEBOOK_PATH, 'rb') as fp:
        codebook = pickle.load(fp)

    # Extract features from the training and test data using Spatial Pyramid Matching (SPM).
    x_train_spm = extract_vocab_SPM(x_train, L=cfg.SPM_L, kmeans=codebook)
    x_test_spm = extract_vocab_SPM(x_test, L=cfg.SPM_L, kmeans=codebook)

    # Initialize the SVM model with the specified kernel and parameters.
    svm_model = SVM(kernel=cfg.SVM_KERNEL, kernel_params={}, lambduh=cfg.SVM_LAMBDUH)

    # Convert the feature matrices to GPU arrays for faster computation.
    x_train_spm = xp.asarray(x_train_spm)
    x_test_spm = xp.asarray(x_test_spm)

    # Convert class labels from string to numerical format.
    y_tr, y_te = [], []
    for y in y_train:
        y_tr.append(dic_class_name[y])
    y_tr = xp.asarray(y_tr)
    for y in y_test:
        y_te.append(dic_class_name[y])
    y_te = xp.asarray(y_te)

    # Train the SVM model using the training data.
    svm_model.fit(x_train_spm, y_tr)

    # Predict the class labels for the test data.
    y_pred = svm_model.predict(x_test_spm)

    # Calculate and print the accuracy of the model on the test data.
    count = sum(y_pred == y_te)
    print(f'Accuracy: {count / y_pred.shape[0]}')

    # Convert numerical labels back to the original class names for better interpretability.
    y_pred_array = [get_key(dic_class_name, int(y)) for y in y_pred]
    y_te_array = [get_key(dic_class_name, int(y)) for y in y_te]

    # Print the true and predicted class names.
    print(y_te_array)
    print(y_pred_array)

    # Generate and visualize the confusion matrix to understand the model's performance in detail.
    cnf_matrix = confusion_matrix(np.array([y_te_array]).T, y_pred_array, labels=class_name)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(90, 30))
    plotting_confusion_matrix(cnf_matrix, classes=class_name, title='Confusion matrix, without normalization')

def get_key(dic_class_name, val):
    """
    Helper function to get the class name from its numerical label.
    Args:
        dic_class_name: Dictionary mapping class names to numerical labels.
        val: The numerical label for which the class name is needed.
    Returns:
        The class name corresponding to the given numerical label.
    """
    for key, value in dic_class_name.items():
        if val == value:
            return key

if __name__ == "__main__":
    SPM_data1()
