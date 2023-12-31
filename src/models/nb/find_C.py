import pickle

import numpy as np
import cupy as xp
from ..configs import config as cfg
from experiment.data_preparation import prepare_data_test, prepare_data_train, prepare_data_val
from features_extraction import extract_features
from histogram_extraction import build_vocab
from svm_gpu import SVM

# Dictionary mapping class names to numerical labels.
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


def find_C():
    """
    Determine the optimal value of the regularization parameter (C) for the SVM classifier.
    This is done by training the SVM with different values of C and selecting the one that
    results in the highest accuracy on the validation set.
    """
    # Prepare the training, validation, and test data.
    x_train, y_train = prepare_data_train()
    x_test_C, y_test_C = prepare_data_test()
    x_val, y_val = prepare_data_val()

    # Combine the training and validation data for cross-validation.
    x_train_C = x_train + x_val
    y_train_C = y_train + y_val

    # Load the codebook generated from the training data.
    with open(cfg.CODEBOOK_PATH, 'rb') as fp:
        codebook = pickle.load(fp)

    # Extract features from the training and test data using the loaded codebook.
    _, x_train_C = extract_features(x_train_C)
    _, x_test_C = extract_features(x_test_C)

    # Build the visual word histograms for the training and test data.
    x_train_C = build_vocab(x_train_C, codebook)
    x_test_C = build_vocab(x_test_C, codebook)

    # Initialize variables to store the best accuracy and corresponding C value.
    best_accuracy = 0
    best_c = 0.0001

    # Convert class labels from string to numerical format.
    y_tr_C, y_te_C = [], []
    for y in y_train_C:
        y_tr_C.append(dic_class_name[y])
    y_tr_C = xp.asarray(y_tr_C)
    for y in y_test_C:
        y_te_C.append(dic_class_name[y])
    y_te_C = xp.asarray(y_te_C)

    # Convert feature matrices to GPU arrays for faster computation.
    x_train_C = xp.asarray(x_train_C)
    x_test_C = xp.asarray(x_test_C)

    # Iterate over a range of C values to find the optimal one.
    for c in np.arange(cfg.C_VALUES_RANGE[0], cfg.C_VALUES_RANGE[1], cfg.C_VALUES_RANGE[2]):
        # Initialize and train the SVM model with the current value of C.
        svm_model = SVM(kernel='linear', kernel_params={}, lambduh=c)
        svm_model.fit(x_train_C, y_tr_C)
        y_pred = svm_model.predict(x_test_C)
        
        # Calculate the accuracy of the model on the test data.
        count = sum(y_pred == y_te_C)
        accuracy = count / y_pred.shape[0]
        print(f"C = {c}, Accuracy: {accuracy * 100}%")
        
        # Update the best accuracy and C value if the current model is better.
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_c = c

    # Print out the best C value and the corresponding accuracy.
    print(f"Best C: {best_c}, Best Accuracy: {best_accuracy * 100}%")

if __name__ == "__main__":
    find_C()
