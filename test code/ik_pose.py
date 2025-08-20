"""Absolute operational space pose controller (IK-based, simple)."""
from __future__ import annotations
import numpy as np
import torch
import genesis as gs
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
from .base_controller import Controller
from dexsuite.core.registry import register_controller

@register_controller("ik_pose")
class IKPoseController(Controller):
    """
    Absolute Cartesian pose controller.
    Action layout (6 floats in [-1, 1]):
    0–2 → x, y, z [m] (absolute around home)
    3–5 → roll, pitch, yaw [rad] (absolute)
    """
    
    def __init__(self, *, entity, model, **kwargs) -> None:
        super().__init__(entity=entity, model=model, **kwargs)
        self.link_name = getattr(model, "end_link", None)
        if self.link_name is None:
            raise ValueError("IK_POSE requires the ArmModel to define `end_link`.")
        self.link = self.entity.get_link(self.link_name)
        self.home_q = getattr(model, "home_q", np.zeros(self.model.dof))
        self.home_pos = None  # set in post_build()
        self.initial_R = None  # store initial rotation matrix on first call
        self.first_call = True  # flag to track first call
        
        # Simple scales (keep it minimal)
        self.LIN_SCALE = 0.30  # ±30 cm
        self.ANG_SCALE = np.pi  # ±π rad
    
    # called only after scene.build()
    def post_build(self) -> None:
        super().post_build()  # install kp/kv/force ranges
        q_home = torch.as_tensor(self.home_q, dtype=torch.float32, device=gs.device)
        pos0, quat0 = self.entity.forward_kinematics(
            qpos=q_home, links_idx_local=[self.link.idx_local]
        )
        self.home_pos = pos0.squeeze(0)  # (3,)
        # my quat is w x y z , make to handle this
        self.home_quat = quat0.squeeze(0)  # (4,) in [w, x, y, z] order
    
    @torch.no_grad()
    def _apply_cmd(self, vec) -> None:
        vec = torch.clamp(torch.as_tensor(vec, dtype=torch.float32), -1.0, 1.0).flatten()
        if vec.numel() != 6:
            raise ValueError("IK pose controller expects six values")
        
        # Get current pose
        qpos = self.entity.get_dofs_position()
        pos, quat = self.entity.forward_kinematics(  # quat = [w,x,y,z] (torch)
            qpos=qpos,
            links_idx_local=[self.link.idx_local],
        )
        
        # Handle batch dimension
        pos = pos.squeeze(0)    # Remove batch dimension
        quat = quat.squeeze(0)  # Remove batch dimension, shape (4,)
        
        # On first call, capture initial quaternion and convert to rotation matrix
        if self.first_call:
            # Convert quat [w,x,y,z] to [x,y,z,w] for scipy
            self.initial_R = R.from_quat([quat[1].item(), quat[2].item(), 
                                        quat[3].item(), quat[0].item()])
            initial_rpy = self.initial_R.as_euler('xyz')  # Get roll, pitch, yaw
            print(f"Initial quaternion [w,x,y,z]: [{quat[0].item():.4f}, {quat[1].item():.4f}, {quat[2].item():.4f}, {quat[3].item():.4f}]")
            print(f"Initial RPY [roll,pitch,yaw]: [{initial_rpy[0]:.4f}, {initial_rpy[1]:.4f}, {initial_rpy[2]:.4f}]")
            self.first_call = False
        
        # Scale normalised action → absolute pose (around home)
        dp = vec[:3] * self.LIN_SCALE   # [m]
        rpy = vec[3:] * self.ANG_SCALE  # [rad]
        
        new_pos = self.home_pos + dp
        
        # Apply rotation offset to the initial orientation (not current)
        offset_R = R.from_euler("xyz", rpy.cpu().numpy())
        new_R = offset_R * self.initial_R  # Always relative to initial orientation
        
        x, y, z, w = new_R.as_quat()
        new_quat = torch.tensor([w, x, y, z], dtype=torch.float32, device=gs.device)
        
        # IK + position control
        q_target = self.entity.inverse_kinematics(
            link=self.link,
            pos=new_pos,
            quat=new_quat,
            init_qpos=self.home_q,
            max_solver_iters=100,
            damping=0.05,
            max_step_size=0.05,
        )
        self.entity.control_dofs_position(q_target, self.dofs_idx)
    
    def action_space(self) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)