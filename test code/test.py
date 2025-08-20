import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

def _resolve_asset(filename: str) -> str:
    """
    Look for the .task model in a few sensible places:
      - ./pos_tracking/models/<filename>            (next to this file, in a models/ folder)
      - ./pos_tracking/<filename>                   (next to this file)
      - project root models/<filename>              (two dirs up, models/)
      - env var MP_HAND_MODEL (absolute path)
    """
    here = os.path.dirname(__file__)
    candidates = [
        os.environ.get("MP_HAND_MODEL") or "",
        os.path.join(here, "models", filename),
        os.path.join(here, filename),
        os.path.join(here, "..", filename),
        os.path.join(here, "..", "models", filename),
        os.path.join(here, "..", "..", "models", filename),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Could not find {filename}. Checked:\n" + "\n".join("  - " + os.path.abspath(c) for c in candidates)
    )

# In __init__ (or wherever you create the landmarker):
model_path = _resolve_asset("hand_landmarker.task")  # or "hand_landmarker_full.task" / "hand_landmarker_lite.task"
base_opts = mp_python.BaseOptions(
    model_asset_path=model_path,
    delegate=mp_python.BaseOptions.Delegate.GPU
)
hl_opts = mp_vision.HandLandmarkerOptions(
    base_options=base_opts,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
self.hands = mp_vision.HandLandmarker.create_from_options(hl_opts)
print(f"[mediapipe] using model: {model_path}")
