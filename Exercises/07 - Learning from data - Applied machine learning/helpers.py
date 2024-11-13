from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd



def preprocess_data(
    df: pd.DataFrame,
    label: str,
    train_size: float = 0.6,
    seed: Optional[int] = None):
    """Transforms data into numpy arrays and splits it into a train and test set

    Args:
        df: Data to split
        label: name of the training label, values of column should be numerical
        train_size: proportion of the data used for training
        val_size: proportion of the data used for validation
        seed: random seed
        categorical_label: whether the label is categorical or not

    Returns:
        object: Tuple containing the training features, training label,
            test features, test label and names of the features
    """

    df = df.sort_values(by=label)

    df = df.sample(frac=1, random_state=seed)
    train, test = (df[: int(len(df) * train_size)], df[int(len(df) * train_size) :])

    X_train = train.drop(columns=label)
    X_test = test.drop(columns=label)

    y_train = train[label]
    y_test = test[label]

    return X_train, y_train, X_test, y_test