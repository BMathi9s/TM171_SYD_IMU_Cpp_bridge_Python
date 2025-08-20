# dexsuite/models/manipulators/franka/franka.py
from __future__ import annotations

from pathlib import Path
import numpy as np

from dexsuite.core.components import ModularManipulatorModel
from dexsuite.core.registry import register_manipulator

ASSETS = Path(__file__).parent / "assets"


@register_manipulator("franka")
class FrankaArm(ModularManipulatorModel):
    mjcf_path = str(ASSETS / "panda_nohand.xml")
    dof = 7

    root_link = "link0"
    end_link = "attachment"

    home_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 3.1415, 2.356], dtype=np.float32)



    # Example gains / limits (optional)
    kp = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000], dtype=np.float32)
    kv = np.array([450, 450, 350, 350, 200, 200, 200], dtype=np.float32)
    force_range = np.array([150, 150, 150, 150, 150, 12, 12], dtype=np.float32)

