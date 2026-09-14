"""
Trains the landmark-based letter classifier on landmark_data.csv
(produced by collect_data.py).

Uses scikit-learn instead of TensorFlow/Keras -- a 42-input MLP doesn't
need a deep learning framework, and this avoids TensorFlow's native DLL
entirely (which fails to load under some Windows/Python setups, notably
the Microsoft Store Python distribution).

Accuracy is reported by HOLDING OUT A WHOLE PERSON, not by a random split of
rows. Frames inside one recording burst are near-duplicates of a single held
pose, so a shuffled split trains on frame 200 and tests on frame 201: it
measures memorisation. On the first three signers that split reported 98.6%
while the same model scored 81.7% on a signer it had never seen. The second
number is the one a stranger at a demo experiences, and the one to quote.

Input:  landmark_data.csv     -- label + person + 42 normalized landmark floats
Output: landmark_model.joblib -- trained scikit-learn model
        landmark_labels.json  -- the label order the model's output
                                 indices correspond to (index i -> label)

    python train_classifier.py            # cross-person report, then train
    python train_classifier.py --quick    # skip the report, just train
"""
import json
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

import dataset

DATA_PATH = "landmark_data.csv"
MODEL_OUT = "landmark_model.joblib"
LABELS_OUT = "landmark_labels.json"


def build_model():
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        max_iter=500,
        random_state=42,
    )


def load():
    df = pd.read_csv(DATA_PATH)
    if "person" not in df.columns:
        sys.exit(f"{DATA_PATH} has no `person` column. Run "
                 f"`python backfill_person.py` first.")
    # Select features explicitly rather than dropping known columns, so a new
    # metadata column can never silently become a 43rd input feature.
    X = df[dataset.LANDMARK_COLUMNS].values.astype("float32")
    labels = sorted(df["label"].unique())
    y = df["label"].map({l: i for i, l in enumerate(labels)}).values
    groups = df["person"].values
    return df, X, y, groups, labels


def cross_person_report(X, y, groups, labels):
    """Leave-one-person-out: train on everyone else, test on the held-out signer.

    Rows whose person is `unknown` (the word signs, recorded by all three
    signers in one sitting with no boundary in the file marking who is who)
    are used for training on every fold and never tested on. Assigning them to
    a name we cannot verify would corrupt exactly the grouping this measures.
    """
    unknown = groups == dataset.UNKNOWN_PERSON
    people = sorted(set(groups[~unknown]))
    if len(people) < 2:
        print("Only one person in the dataset -- cross-person accuracy cannot be "
              "measured. Collect from someone else before quoting any number.")
        return

    # A label only one person ever signed cannot be tested cross-person: on the
    # fold holding that person out it is absent from training entirely, so it
    # would score 0 and drag the headline down for the wrong reason.
    per_label_people = {l: set() for l in labels}
    for label_idx, person in zip(y[~unknown], groups[~unknown]):
        per_label_people[labels[label_idx]].add(person)
    single = sorted(l for l, ps in per_label_people.items() if len(ps) < 2)

    print(f"\n{'='*66}\nCROSS-PERSON ACCURACY (leave-one-person-out)\n{'='*66}")
    print(f"people: {', '.join(people)}")
    if unknown.any():
        unknown_labels = sorted({labels[i] for i in y[unknown]})
        print(f"\n  {int(unknown.sum())} rows have no known signer "
              f"({', '.join(unknown_labels)}): trained on in every fold, never "
              f"tested on.")
    if single:
        print(f"\n  NOT VALIDATED -- no two known signers: {', '.join(single)}")
        print("  Excluded from the headline number. Re-collect these with the")
        print("  signer recorded, so they can be validated across people.")

    testable = np.array([labels[i] not in single for i in y])
    y_true_all, y_pred_all, keep_all = [], [], []

    known_idx = np.where(~unknown)[0]
    unknown_idx = np.where(unknown)[0]
    for tr, te in LeaveOneGroupOut().split(known_idx, y[known_idx],
                                           groups[known_idx]):
        train_idx = np.concatenate([known_idx[tr], unknown_idx])
        test_idx = known_idx[te]
        held = groups[test_idx][0]
        model = build_model()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        keep = testable[test_idx]
        acc = (pred[keep] == y[test_idx][keep]).mean()
        print(f"  held out {held:10} -> {acc:.3f}  ({keep.sum()} rows)")
        y_true_all.append(y[test_idx]); y_pred_all.append(pred); keep_all.append(keep)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    keep = np.concatenate(keep_all)
    print(f"\n  MEAN CROSS-PERSON ACCURACY: {(y_pred[keep] == y_true[keep]).mean():.3f}")
    print("  (quote this one, not the in-sample number below)")

    print("\nPer-label, pooled across folds:")
    print(classification_report(y_true[keep], y_pred[keep],
                                labels=range(len(labels)), target_names=labels,
                                zero_division=0))

    print("Most-confused pairs across people (true -> predicted, count):")
    cm = confusion_matrix(y_true[keep], y_pred[keep], labels=range(len(labels)))
    confusions = [(cm[i][j], labels[i], labels[j])
                  for i in range(len(labels)) for j in range(len(labels))
                  if i != j and cm[i][j] > 0]
    for count, true_l, pred_l in sorted(confusions, reverse=True)[:10]:
        print(f"  {true_l} -> {pred_l}: {count}")


def main():
    quick = "--quick" in sys.argv
    df, X, y, groups, labels = load()

    print(f"Loaded {len(df)} rows, {len(labels)} labels, "
          f"{df['person'].nunique()} people.")
    print(df.groupby(["person"])["label"].value_counts().unstack(fill_value=0)
          .T.to_string())

    if not quick:
        cross_person_report(X, y, groups, labels)

    # The shipped model trains on everything -- holding a person out is for
    # measurement, not for the artifact that main.py loads.
    print(f"\n{'='*66}\nTRAINING SHIPPED MODEL ON ALL {len(X)} ROWS\n{'='*66}")
    model = build_model()
    model.fit(X, y)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    in_sample = (build_model().fit(X_tr, y_tr).predict(X_val) == y_val).mean()
    print(f"In-sample (random split) accuracy: {in_sample:.3f}  <- INFLATED, "
          f"do not quote:\n  frames within a recording are near-duplicates, so "
          f"this mostly measures memorisation.")

    joblib.dump(model, MODEL_OUT)
    with open(LABELS_OUT, "w") as f:
        json.dump(labels, f)
    print(f"\nSaved model to {MODEL_OUT} and label order to {LABELS_OUT}")


if __name__ == "__main__":
    main()
