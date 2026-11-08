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


def load_classifier(model_path="landmark_model.joblib",
                    labels_path="landmark_labels.json"):
    """(model, labels): the classifier and its output index -> label order.

    Models trained since the view column carry their label order inside
    (label_names_), so a first-person model cannot be paired with the wrong
    labels file; older ones fall back to landmark_labels.json.
    """
    import json
    import joblib
    model = joblib.load(model_path)
    labels = getattr(model, "label_names_", None)
    if labels is None:
        with open(labels_path) as fh:
            labels = json.load(fh)
    return model, list(labels)
