import os
import joblib


def save_model(model, filename):
    """
    Save a trained model as a .pkl file.
    """

    # Create models folder if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save model
    filepath = os.path.join("models", filename)
    joblib.dump(model, filepath)

    print(f"Model saved: {filepath}")