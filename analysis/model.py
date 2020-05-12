#!/usr/bin/env python
"""Run our model against the data contained in input.csv.
"""

import sys
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.wrappers.scikit_learn as ksl
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score


INPUT_FILE = "input.csv"
SEX_FEATURE = 'Sex'
ALL_ATTRIBUTES = ['Patient_ID',
                  'admitted_itu',
                  'died',
                  'chd_code',
                  SEX_FEATURE,
                  'Age',
                  'smoking_status']
ALL_FEATURES = ['admitted_itu',
                'chd_code',
                SEX_FEATURE,
                'Age',
                'smoking_status']
ALL_LABELS = ['died']
TRANSLATE_SEX = {
    'F': 1,
    'M': 0
}


def main(_):
    """Run the model."""
    train_model('.')


def read_data(relative_path):
    """Produce a pandas frame from the INPUT_FILE.

Read data from the sub folder at RELATIVE_PATH."""
    data = pd.read_csv("%s/%s" % (relative_path, INPUT_FILE))
    features = data[ALL_FEATURES]
    features[SEX_FEATURE] = features[SEX_FEATURE].map(TRANSLATE_SEX)
    normalised = normalise_data(features)
    return (normalised, data[ALL_LABELS])


def normalise_data(data):
    """Normalise Pandas frame, DATA, producing values in the range [0, 1]."""
    return data / data.max(axis=0)


def create_neural_network():
    """Create a Keras neural network tuned to our problem."""
    hidden_layer_1_size = 64
    hidden_layer_2_size = 32
    hidden_layer_3_size = 16
    model = Sequential()
    model.add(Dense(hidden_layer_1_size, activation='relu',
                    input_shape=((len(ALL_FEATURES),))))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_layer_2_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_layer_3_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    return model


def create_neural_network_model():
    """Produce a Sci Kit Learn wrapper for a Keras neural network.

Names have significance here.  The fact that a Sci Kit Learn model is
produced means that it *should* be compatible with Sci Kit Learn
methods of verificaiton and training.  The fact that it's wrapping a
Keras-based Neural Network (NN) means that it misbehaves in a few
ways.  Beware YMMV!"""
    epochs = 100
    return ksl.KerasClassifier(create_neural_network, epochs=epochs, verbose=0)


def train_and_test_neural_net(untrained_model, X, y, verbose=False):
    """Produce a trained version of UNTRAINED_MODEL.

As a side effect, also print the error rate and the k_folds error
rates when VERBOSE is True."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    if verbose:
        print("> Performing cross validation...")
        print("> Cross validation score: %s" %
              str(cross_val_score(untrained_model, X, to_categorical(y), cv=5)
                  .mean()))
    print("> Training against training data...")
    trained_model = untrained_model.fit(X_train, to_categorical(y_train)).model
    if verbose:
        print("> Score against test data: %s" %
              str(accuracy_score(
                  [x.argmax() for x in trained_model.predict(X_test)],
                  y_test)))
    return trained_model


def train_model(relative_path):
    """Run our model on the data in 'input.csv'."""
    (X, y) = read_data(relative_path)
    return train_and_test_neural_net(
        create_neural_network_model(),
        X.copy(),
        y.copy(),
        verbose=True)


if __name__ == "__main__":
    main(sys.argv)
