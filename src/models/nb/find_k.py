from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import time
import pickle
from ..configs import config as cfg
from codebook_extraction import build_codebook
from experiment.data_preparation import prepare_data_train
from features_extraction import extract_features
from histogram_extraction import build_vocab

def find_k():
    """
    Determine the optimal value of 'k' for the codebook used in image classification.
    The optimal 'k' is the one that maximizes the accuracy of the classifier.
    This is done by training the Gaussian Naive Bayes model with different 'k' values.
    """
    # Initialize the Gaussian Naive Bayes model.
    model = GaussianNB()

    # Prepare the training data.
    x_train, y_train = prepare_data_train()

    # Extract features from the training data.
    _, x_train = extract_features(x_train)

    # Iterate over a range of k values to find the optimal one.
    for k in range(cfg.K_VALUES_RANGE[0], cfg.K_VALUES_RANGE[1], cfg.K_VALUES_RANGE[2]):
        start = time.time()  # Record the start time for performance evaluation.

        # Build the codebook with the current value of 'k' (number of visual words).
        codebook = build_codebook(x_train, k=k)

        # Convert the features into a visual word histogram using the generated codebook.
        x_find_k_temp = build_vocab(x_train, codebook)

        # Train the model and predict using the same training data.
        model.fit(x_find_k_temp, y_train)
        y_pred = model.predict(x_find_k_temp)

        end = time.time()  # Record the end time for performance evaluation.

        # Calculate the accuracy and error rate for the current value of 'k'.
        accuracy = accuracy_score(y_train, y_pred)
        error = 1 - accuracy

        # Print the prediction, 'k' value, error rate, and time taken for the current iteration.
        print(y_pred)
        print('k = ', k, ' error rate = ', error, ' time = ', end - start)

if __name__ == "__main__":
    find_k()

