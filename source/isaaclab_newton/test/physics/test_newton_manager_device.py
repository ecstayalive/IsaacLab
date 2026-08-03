# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for Newton CUDA graph device ownership."""

from types import SimpleNamespace

from isaaclab_newton.physics import newton_manager as newton_manager_module
from isaaclab_newton.physics.newton_manager import NewtonManager

from isaaclab.physics import PhysicsManager


def test_cuda_graph_capture_uses_simulation_device(monkeypatch):
    """CUDA graph capture must use the simulation device, not Warp's default."""
    captured_devices = []
    captured_graph = object()

    class FakeSolverCfg:
        use_mujoco_contacts = True

        def to_dict(self):
            return {"solver_type": "mujoco_warp", "use_mujoco_contacts": True}

    class FakeSolverMuJoCo:
        def __init__(self, model, use_mujoco_contacts=True):
            self.model = model

    class FakeScopedCapture:
        def __init__(self, device=None):
            captured_devices.append(device)
            self.graph = captured_graph

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    cfg = SimpleNamespace(
        num_substeps=1,
        solver_cfg=FakeSolverCfg(),
        use_cuda_graph=True,
    )
    monkeypatch.setattr(PhysicsManager, "_cfg", cfg, raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:2", raising=False)
    monkeypatch.setattr(NewtonManager, "_model", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_initialize_contacts", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "_simulate", classmethod(lambda cls: None))
    monkeypatch.setattr(newton_manager_module, "SolverMuJoCo", FakeSolverMuJoCo)
    monkeypatch.setattr(newton_manager_module.wp, "ScopedCapture", FakeScopedCapture)

    NewtonManager.initialize_solver()

    assert captured_devices == ["cuda:2"]
    assert NewtonManager._graph is captured_graph
