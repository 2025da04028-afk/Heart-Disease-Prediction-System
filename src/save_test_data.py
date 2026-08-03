import pandas as pd


def save_test_data(X_test):
    """
    Save test dataset for Streamlit application.
    """

    df = pd.DataFrame(X_test)

    df.to_csv("test_data.csv", index=False)

    print("Test data saved successfully as test_data.csv")