from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import pickle
from ..configs import config as cfg
from codebook_extraction import build_codebook
from experiment.data_preparation import prepare_data_train
from features_extraction import extract_features
from histogram_extraction import build_vocab


def find_k():
    model = GaussianNB()

    x_train, y_train = prepare_data_train()

    _, x_train = extract_features(x_train)

    for k in range(cfg.K_VALUES_RANGE[0], cfg.K_VALUES_RANGE[1], cfg.K_VALUES_RANGE[2]):
        start = time.time()  # Capture start time
        codebook = build_codebook(x_train, k=k)
        x_find_k_temp = build_vocab(x_train, codebook)
        model.fit(x_find_k_temp, y_train)
        y_pred = model.predict(x_find_k_temp)
        end = time.time()  # Capture end time
        accuracy = accuracy_score(y_train, y_pred)
        error = 1 - accuracy
        print(y_pred)
        print('k = ', k, ' error rate = ', error, ' time = ', end - start)


if __name__ == "__main__":
    find_k()
