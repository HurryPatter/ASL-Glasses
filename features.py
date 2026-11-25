"""The hand-frame features every model in this project is trained and run on.

One definition, imported by collect_data.py (which writes training rows),
main.py and evaluate.py (which classify live frames). It used to be copied
into all three, and the README had to warn that changing one copy without
the others would make the trained model and live inference disagree
silently. With first-person data about to be collected on a new camera, that
is exactly the kind of drift that would not show up until the accuracy did.

The hand frame: origin at the wrist, unit length = wrist -> middle knuckle,
y axis along that line, and mirrored if needed so the thumb is on +x. That
removes position in the image, distance from the camera, in-plane rotation
and left vs right hand, exactly (to ~1e-16). What survives is out-of-plane
tilt and the handshape itself, which is what the classifier should see.
"""
import numpy as np

WRIST, THUMB_MCP, MIDDLE_MCP = 0, 2, 9


def normalize_landmarks(landmarks, frame_w, frame_h):
    """21 MediaPipe landmarks -> 42 floats [p0_x, p0_y, ..., p20_x, p20_y].

    Landmarks arrive in normalised image coordinates; scaling by the frame
    size first keeps the geometry true when the frame is not square.
    """
    pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=float)
    pts[:, 0] *= frame_w
    pts[:, 1] *= frame_h

    v = pts[MIDDLE_MCP] - pts[WRIST]
    size = np.linalg.norm(v) or 1e-6
    y_hat = v / size
    x_hat = np.array([y_hat[1], -y_hat[0]])
    local = ((pts - pts[WRIST]) @ np.stack([x_hat, y_hat], axis=1)) / size

    if local[THUMB_MCP][0] < 0:     # thumb landed on -x: mirror it onto +x
        local[:, 0] *= -1

    return local.flatten()


def output_names(classes, label_names):
    """The label for each column of predict_proba, in column order.

    A scikit-learn classifier only has columns for the classes it saw in
    training (model.classes_), and main.py / evaluate.py pick the name by
    column position. So the names must line up with classes_, not with the
    full label list: a model trained on a view that lacks some labels (say,
    first-person data without the word signs) has fewer columns than there
    are labels, and pairing it with the full list shifted every name after
    the first gap -- I read as HELLO, L as IHATEYOU.

    classes are the integer indices train_classifier.py trains on, so each
    one picks its own name out of label_names. Also accepts label_names that
    are already one per column (as saved since this fix).
    """
    classes = [int(c) for c in classes]
    names = list(label_names)
    if len(names) == len(classes) and classes == list(range(len(classes))):
        return names                       # every label trained: same thing
    if len(names) == len(classes) and len(names) != max(classes) + 1:
        return names                       # already one name per column
    if classes and max(classes) < len(names):
        return [names[c] for c in classes]
    raise ValueError(f"model has {len(classes)} outputs (classes {classes}) "
                     f"but {len(names)} label names: they do not belong together")


def load_classifier(model_path="landmark_model.joblib",
                    labels_path="landmark_labels.json"):
    """(model, labels): the classifier and the label for each of its outputs.

    Models trained since the view column carry their label names inside
    (label_names_), so a first-person model cannot be paired with the wrong
    labels file; older ones fall back to landmark_labels.json. Either way the
    names are aligned to the model's outputs (see output_names), and a model
    and a labels file that cannot belong together are refused rather than
    allowed to speak the wrong words.
    """
    import json
    import joblib
    model = joblib.load(model_path)
    labels = getattr(model, "label_names_", None)
    if labels is None:
        with open(labels_path) as fh:
            labels = json.load(fh)
    return model, output_names(model.classes_, labels)
