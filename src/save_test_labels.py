import pandas as pd

def save_test_labels(y_test):
    """
    Save actual test labels.
    """
    y_test.to_csv("test_labels.csv", index=False)

    print("Test labels saved successfully as test_labels.csv")