"""
Trains the landmark-based letter classifier on landmark_data.csv
(produced by collect_data.py).

Input:  landmark_data.csv  -- label + 42 normalized landmark floats per row
Output: landmark_model.h5  -- trained Keras model
        landmark_labels.json -- the letter order the model's output
                                 indices correspond to (index i -> letter)

Run this after collect_data.py, before wiring the model into main.py.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

DATA_PATH = "landmark_data.csv"
MODEL_OUT = "landmark_model.h5"
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
    y_train_cat = to_categorical(y_train, num_classes=len(letters))
    y_val_cat = to_categorical(y_val, num_classes=len(letters))

    model = Sequential([
        Input(shape=(42,)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(len(letters), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    early_stop = EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True)
    model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=150,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    # ── Evaluation ───────────────────────────────────────────────────────
    val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print("\nPer-letter results:")
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
    model.save(MODEL_OUT)
    with open(LABELS_OUT, "w") as f:
        json.dump(letters, f)
    print(f"\nSaved model to {MODEL_OUT} and label order to {LABELS_OUT}")


if __name__ == "__main__":
    main()
