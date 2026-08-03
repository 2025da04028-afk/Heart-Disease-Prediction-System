import pandas as pd


def save_results(results):
    """
    Save all model evaluation results to a CSV file.
    """

    df = pd.DataFrame(results)

    df.to_csv("results.csv", index=False)

    print("\nResults saved successfully as results.csv")