import numpy as np


def test_accuracy(predicted_cat, labeled_cat):
    predicted_cat_arr = np.array(predicted_cat)
    labeled_cat_arr = np.array(labeled_cat)
    diff = predicted_cat_arr - labeled_cat_arr
    false_count = np.count_nonzero(diff)
    accuracy = 1 - false_count / predicted_cat_arr.shape[0]
    return accuracy
