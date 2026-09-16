"""
Trains the Veronica sign classifier on veronica_signs.csv
(produced by collect_signs.py / rebuild_signs.py).

Deliberately the same shape as train_classifier.py, because its methodology is
the strongest thing in this repository: **leave-one-person-out is the headline**
and the random split is printed only as an explicitly inflated comparison. On
the letter data those two numbers are 84.0% and 98.6%.

All of the decisions about what goes in which fold live in folds.py, which is
standard-library only and tested in CI. The sklearn call is the easy part to
get right; choosing the folds is where a silent mistake produces a number that
looks perfectly reasonable.

Input:  veronica_signs.csv       -- metadata + 371 sign features per clip
Output: veronica_model.joblib    -- trained pipeline
        veronica_labels.json     -- label order (index i -> label)

    python train_signs.py            # cross-person report, then train
    python train_signs.py --quick    # skip the report, just train

Expect a lower number than the letters' 84%. More classes, harder classes, and
a genuinely harder task -- a lower number on real ASL is a better result than a
higher one on fingerspelling, and it should be reported as such.
"""
import json
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import folds
import sequence
import signset

DATA_PATH = signset.SIGNS_PATH
MODEL_OUT = "veronica_model.joblib"
LABELS_OUT = "veronica_labels.json"


def build_model():
    """Scaled, then a slightly wider MLP than the letter model's (64, 32).

    The scaler is new, and it is not decoration. The letter model's 42 inputs
    were all the same kind of quantity in the same range, so scaling them
    changed nothing. These 371 are not: normalized landmark coordinates sit
    near +/-3, clamped relational distances run to 6, orientation cosines to 1,
    a duration to 3, a coverage fraction to 1. Un-scaled, the widest block
    dominates the first layer purely because of its units.

    Wider layers because the input is 371 floats rather than 42 and the class
    count is ~68 rather than 27 -- but only slightly, since the clip count is
    the binding constraint and this still has to run on the glasses.
    """
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            max_iter=600,
            random_state=42,
        ),
    )


def load():
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in signset.HEADER if c not in df.columns]
    if missing:
        sys.exit(f"{DATA_PATH} is missing {len(missing)} expected column(s), "
                 f"starting with {missing[:3]}. Re-derive it with "
                 f"`python rebuild_signs.py`.")

    # Select features explicitly rather than dropping known metadata, so a new
    # metadata column can never silently become a 372nd input -- and `person`
    # in particular would let the model identify the signer instead of the
    # sign, inflating exactly the number this script exists to measure.
    X = df[sequence.SIGN_COLUMNS].values.astype("float32")
    labels = sorted(df["label"].unique())
    y = df["label"].map({l: i for i, l in enumerate(labels)}).values
    people = df["person"].tolist()
    return df, X, y, people, labels


def accuracy(pred, truth, mask):
    mask = np.asarray(mask, dtype=bool)
    return float("nan") if not mask.any() else (pred[mask] == truth[mask]).mean()


def cross_person_report(X, y, people, labels):
    """Train on everyone else, test on the held-out signer."""
    label_names = [labels[i] for i in y]

    print(f"\n{'='*70}\nBEFORE THE NUMBER\n{'='*70}")
    for note in folds.problems(label_names, people):
        print(f"  - {note}")
    if len(set(people)) < 2:
        return

    testable = np.array(folds.testable_mask(label_names, people))
    is_sign = np.array(folds.sign_mask(label_names))

    print(f"\n{'='*70}\nCROSS-PERSON ACCURACY (leave-one-person-out)\n{'='*70}")

    truth_all, pred_all, keep_all, sign_all = [], [], [], []
    for held, train_idx, test_idx in folds.folds(label_names, people):
        model = build_model()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        keep = testable[test_idx]
        signs_only = keep & is_sign[test_idx]
        truth = y[test_idx]

        print(f"  held out {held:12} signs {accuracy(pred, truth, signs_only):.3f}"
              f"   with {signset.REST_LABEL} {accuracy(pred, truth, keep):.3f}"
              f"   ({int(signs_only.sum())} clips)")

        truth_all.append(truth); pred_all.append(pred)
        keep_all.append(keep); sign_all.append(signs_only)

    truth = np.concatenate(truth_all)
    pred = np.concatenate(pred_all)
    keep = np.concatenate(keep_all)
    signs_only = np.concatenate(sign_all)

    print(f"\n  MEAN CROSS-PERSON ACCURACY (signs only): "
          f"{accuracy(pred, truth, signs_only):.3f}")
    print(f"  (quote this one -- not the figure below, and not the one "
          f"including {signset.REST_LABEL})")

    print("\nPer-label, pooled across folds:")
    print(classification_report(truth[signs_only], pred[signs_only],
                                labels=range(len(labels)), target_names=labels,
                                zero_division=0))

    print("Most-confused pairs across people (true -> predicted, count):")
    matrix = confusion_matrix(truth[signs_only], pred[signs_only],
                              labels=range(len(labels)))
    confusions = [(matrix[i][j], labels[i], labels[j])
                  for i in range(len(labels)) for j in range(len(labels))
                  if i != j and matrix[i][j] > 0]
    for count, true_l, pred_l in sorted(confusions, reverse=True)[:12]:
        print(f"  {true_l} -> {pred_l}: {count}")

    # The pairs to read first. A confusion between two signs that differ in
    # only ONE parameter says that parameter is not reaching the classifier --
    # which is a representation problem to fix in hands/location/sequence, not
    # something more data will cure. G/Q on the letter model was exactly this.
    print("\n(Confusions between signs differing in only one parameter -- "
          "location, orientation or movement -- point at the feature layer, "
          "not at the dataset. More clips will not fix those.)")


def main():
    quick = "--quick" in sys.argv
    df, X, y, people, labels = load()

    print(f"Loaded {len(df)} clips, {len(labels)} labels, "
          f"{df['person'].nunique()} people.")
    print(df.groupby(["person"])["label"].value_counts().unstack(fill_value=0)
          .T.to_string())

    if not quick:
        cross_person_report(X, y, people, labels)

    # The shipped model trains on everything -- holding a person out is for
    # measurement, not for the artifact the live pipeline loads.
    print(f"\n{'='*70}\nTRAINING SHIPPED MODEL ON ALL {len(X)} CLIPS\n{'='*70}")
    model = build_model()
    model.fit(X, y)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    in_sample = (build_model().fit(X_tr, y_tr).predict(X_val) == y_val).mean()
    print(f"In-sample (random split) accuracy: {in_sample:.3f}  <- INFLATED, "
          f"do not quote:\n  clips of one sign by one person in one sitting are "
          f"far more like each other\n  than like anyone else's, so this mostly "
          f"measures memorisation.")

    joblib.dump(model, MODEL_OUT)
    with open(LABELS_OUT, "w") as f:
        json.dump(labels, f)
    print(f"\nSaved model to {MODEL_OUT} and label order to {LABELS_OUT}")


if __name__ == "__main__":
    main()
