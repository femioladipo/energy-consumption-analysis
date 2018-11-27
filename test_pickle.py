import pickle
import pandas as pd

PICKLE_FILE = "model.pickle"
DATA_FILE = "data/test.zip"

if __name__ == "__main__":
    with open(PICKLE_FILE, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(DATA_FILE)

    X = model.preprocess_unseen_data(df)
    y_pred = model.predict(X)

    print("--- YOUR PREDICTIONS ---")
    print(y_pred)