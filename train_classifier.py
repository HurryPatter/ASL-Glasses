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

Views (see dataset.py): every row says whether the camera faced the signer
(`front`) or sat on their head (`firstperson`). --train-view and --test-view
choose which rows train and which are tested, so the glasses chapter's three
measurements are one command each:

    # 1. transfer: front-view model, tested on first-person signing
    python train_classifier.py --train-view front --test-view firstperson
    # 2. first-person only
    python train_classifier.py --train-view firstperson --test-view firstperson
    # 3. both views together, tested on first-person
    python train_classifier.py --train-view all --test-view firstperson

The held-out person is removed from training in BOTH views, so no fold ever
trains on the signer it tests. --seeds 0,42,123 averages each fold over
several training seeds: one seed moved a fold by up to 5.4 points, so quote
the averaged figure. The shipped model trains on --train-view; --out names
it, so a first-person model can sit beside the front-view one:

    python train_classifier.py --train-view firstperson --quick \
        --out landmark_model_firstperson.joblib
    python main.py --model landmark_model_firstperson.joblib
"""
import argparse
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


def build_model(seed=42):
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        max_iter=500,
        random_state=seed,
    )


def load():
    df = pd.read_csv(DATA_PATH)
    if "person" not in df.columns:
        sys.exit(f"{DATA_PATH} has no `person` column. Run "
                 f"`python backfill_person.py` first.")
    if "view" not in df.columns:      # collected before the glasses: all front
        df["view"] = dataset.FRONT
    # Select features explicitly rather than dropping known columns, so a new
    # metadata column can never silently become a 43rd input feature.
    X = df[dataset.LANDMARK_COLUMNS].values.astype("float32")
    labels = sorted(df["label"].unique())
    y = df["label"].map({l: i for i, l in enumerate(labels)}).values
    groups = df["person"].values
    views = df["view"].values
    return df, X, y, groups, labels, views


def view_mask(views, which):
    return np.ones(len(views), bool) if which == "all" else views == which


def cross_person_report(X, y, groups, labels, views=None,
                        train_view="all", test_view="all", seeds=(42,)):
    """Leave-one-person-out: train on everyone else, test on the held-out signer.

    Rows whose person is `unknown` (the word signs, recorded by all three
    signers in one sitting with no boundary in the file marking who is who)
    are used for training on every fold and never tested on. Assigning them to
    a name we cannot verify would corrupt exactly the grouping this measures.

    Training rows come from train_view, test rows from test_view; the held-out
    person is excluded from training in every view. A test row whose label
    the fold's training set lacks cannot be scored fairly -- it is absent
    from training, not misread -- so it is left out of the headline and
    listed. With seeds, each fold is trained once per seed and the
    predictions pooled, so the reported accuracy is the seed average.
    """
    views = np.full(len(y), dataset.FRONT) if views is None else views
    unknown = groups == dataset.UNKNOWN_PERSON
    trainable = view_mask(views, train_view)
    testable = view_mask(views, test_view) & ~unknown
    people = sorted(set(groups[testable]))

    title = "CROSS-PERSON ACCURACY (leave-one-person-out)"
    if (train_view, test_view) != ("all", "all"):
        title += f"\ntrain view: {train_view}   test view: {test_view}"
    print(f"\n{'='*66}\n{title}\n{'='*66}")
    if not people:
        print(f"No rows with a known signer in view '{test_view}'.")
        return None
    others = sorted(set(groups[trainable & ~unknown]))
    if len(set(others) | set(people)) < 2:
        print("Only one person -- cross-person accuracy cannot be measured. "
              "Collect from someone else before quoting any number.")
        return None
    print(f"tested people: {', '.join(people)}")
    if (trainable & unknown).any():
        unknown_labels = sorted({labels[i] for i in y[trainable & unknown]})
        print(f"\n  {int((trainable & unknown).sum())} rows have no known signer "
              f"({', '.join(unknown_labels)}): trained on in every fold, never "
              f"tested on.")
    if len(seeds) > 1:
        print(f"  averaged over seeds {', '.join(map(str, seeds))}")

    y_true_all, y_pred_all = [], []
    per_person = []
    untrainable = {}
    for held in people:
        train_idx = np.where(trainable & (groups != held))[0]
        test_idx = np.where(testable & (groups == held))[0]
        present = set(y[train_idx])
        keep = np.array([label in present for label in y[test_idx]])
        for label in set(y[test_idx][~keep]):
            untrainable.setdefault(labels[label], []).append(held)
        test_idx = test_idx[keep]
        if not len(test_idx):
            continue
        accs = []
        for seed in seeds:
            model = build_model(seed)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            accs.append((pred == y[test_idx]).mean())
            y_true_all.append(y[test_idx]); y_pred_all.append(pred)
        per_person.append(np.mean(accs))
        spread = f"  (seeds {min(accs):.3f}-{max(accs):.3f})" if len(seeds) > 1 else ""
        print(f"  held out {held:10} -> {np.mean(accs):.3f}  "
              f"({len(test_idx)} rows){spread}")

    if untrainable:
        print("\n  NOT SCORED -- label missing from that fold's training set:")
        for label, who in sorted(untrainable.items()):
            print(f"    {label}: when holding out {', '.join(who)}")
        print("  Collect these from a second signer (in the training view) so")
        print("  they can be validated across people.")
    if not y_true_all:
        return None

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    keep = np.ones(len(y_true), bool)
    # The headline weights every signer equally, as the README's table does;
    # pooling over rows would let whoever recorded the most rows dominate.
    mean = float(np.mean(per_person))
    print(f"\n  MEAN CROSS-PERSON ACCURACY: {mean:.3f}  (average over signers)")
    print(f"  pooled over all test rows:  {(y_pred == y_true).mean():.3f}")
    print("  (quote the first, not the in-sample number below)")

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
    return mean


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the landmark classifier.")
    p.add_argument("--quick", action="store_true",
                   help="skip the cross-person report, just train")
    p.add_argument("--train-view", default="all", choices=("all",) + dataset.VIEWS,
                   help="rows to train on (and to ship the model from)")
    p.add_argument("--test-view", default="all", choices=("all",) + dataset.VIEWS,
                   help="rows the cross-person report tests on")
    p.add_argument("--seeds", default="42",
                   help="comma-separated training seeds to average the report over")
    p.add_argument("--out", default=MODEL_OUT,
                   help=f"model file to write (default {MODEL_OUT})")
    a = p.parse_args(argv)
    try:
        a.seeds = tuple(int(x) for x in a.seeds.split(","))
    except ValueError:
        p.error("--seeds must be integers separated by commas, e.g. 0,42,123")
    return a


def main():
    args = parse_args()
    df, X, y, groups, labels, views = load()

    print(f"Loaded {len(df)} rows, {len(labels)} labels, "
          f"{df['person'].nunique()} people.")
    print(df.groupby(["view", "person"])["label"].value_counts()
          .unstack(fill_value=0).T.to_string())

    if not args.quick:
        cross_person_report(X, y, groups, labels, views,
                            args.train_view, args.test_view, args.seeds)

    # The shipped model trains on all rows of the training view -- holding a
    # person out is for measurement, not for the artifact main.py loads.
    rows = view_mask(views, args.train_view)
    Xs, ys = X[rows], y[rows]
    print(f"\n{'='*66}\nTRAINING SHIPPED MODEL ON {len(Xs)} ROWS "
          f"(view: {args.train_view})\n{'='*66}")
    model = build_model()
    model.fit(Xs, ys)
    missing = sorted(set(labels) - {labels[i] for i in set(ys)})
    if missing:
        print(f"  this view has no rows for: {', '.join(missing)} -- the model "
              f"can never output them")

    X_tr, X_val, y_tr, y_val = train_test_split(
        Xs, ys, test_size=0.15, stratify=ys, random_state=42)
    in_sample = (build_model().fit(X_tr, y_tr).predict(X_val) == y_val).mean()
    print(f"In-sample (random split) accuracy: {in_sample:.3f}  <- INFLATED, "
          f"do not quote:\n  frames within a recording are near-duplicates, so "
          f"this mostly measures memorisation.")

    # The label order travels inside the model file, so a second model (say,
    # first-person) can never be paired with the wrong landmark_labels.json.
    # The JSON is still written for the default model, for older checkouts.
    model.label_names_ = list(labels)
    model.trained_view_ = args.train_view
    joblib.dump(model, args.out)
    print(f"\nSaved model to {args.out}")
    if args.out == MODEL_OUT:
        with open(LABELS_OUT, "w") as f:
            json.dump(labels, f)
        print(f"and label order to {LABELS_OUT}")


if __name__ == "__main__":
    main()
