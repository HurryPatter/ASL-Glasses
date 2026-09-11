"""
Trains the landmark-based letter classifier on landmark_data.csv
(produced by collect_data.py).

Uses scikit-learn instead of TensorFlow/Keras -- a 42-input MLP doesn't
need a deep learning framework, and this avoids TensorFlow's native DLL
entirely (which fails to load under some Windows/Python setups, notably
the Microsoft Store Python distribution).

Input:  landmark_data.csv    -- label + 42 normalized landmark floats per row
Output: landmark_model.joblib -- trained scikit-learn model
        landmark_labels.json  -- the letter order the model's output
                                  indices correspond to (index i -> letter)
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = "landmark_data.csv"
MODEL_OUT = "landmark_model.joblib"
LABELS_OUT = "landmark_labels.json"


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows across {df['label'].nunique()} letters.")
    print(df["label"].value_counts().sort_index())

    letters = sorted(df["label"].unique())  # index -> letter mapping the model will use
    label_to_idx = {l: i for i, l in enumerate(letters)}

    X = df.drop(columns=["label"]).values.astype("float32")
    y = df["label"].map(label_to_idx).values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        max_iter=500,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ── Evaluation ───────────────────────────────────────────────────────
    val_pred = model.predict(X_val)
    val_acc = (val_pred == y_val).mean()
    print(f"\nValidation accuracy: {val_acc:.3f}\n")
    print(classification_report(y_val, val_pred, target_names=letters, zero_division=0))

    print("Most-confused pairs (true -> predicted, count):")
    cm = confusion_matrix(y_val, val_pred)
    confusions = []
    for i in range(len(letters)):
        for j in range(len(letters)):
            if i != j and cm[i][j] > 0:
                confusions.append((cm[i][j], letters[i], letters[j]))
    for count, true_l, pred_l in sorted(confusions, reverse=True)[:10]:
        print(f"  {true_l} -> {pred_l}: {count}")

    # ── Save ─────────────────────────────────────────────────────────────
    joblib.dump(model, MODEL_OUT)
    with open(LABELS_OUT, "w") as f:
        json.dump(letters, f)
    print(f"\nSaved model to {MODEL_OUT} and label order to {LABELS_OUT}")


if __name__ == "__main__":
    main()