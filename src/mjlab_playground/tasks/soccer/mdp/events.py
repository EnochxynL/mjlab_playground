from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from mjlab.utils.lab_api import math as math_utils  # MJLab: isaaclab.utils.math → mjlab.utils.lab_api.math
from mjlab.entity import Entity  # MJLab: isaaclab.assets Articulation → mjlab.entity Entity
# MJLab: _randomize_prop_by_op defined locally below — isaaclab.envs.mdp.events not available
from mjlab.managers.scene_entity_config import SceneEntityCfg  # MJLab: isaaclab.managers → mjlab.managers.scene_entity_config

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv  # MJLab: isaaclab.envs ManagerBasedEnv → mjlab.envs ManagerBasedRlEnv


# MJLab: local implementation — isaaclab.envs.mdp.events._randomize_prop_by_op not available in MJLab
def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_params: tuple[float, float],
    env_ids: torch.Tensor,
    dim_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
) -> torch.Tensor:
    """Randomize tensor values by operation with given distribution."""  # MJLab: local implementation replacing isaaclab.envs.mdp.events._randomize_prop_by_op
    # resolve distribution function
    if distribution == "uniform":
        rand_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        rand_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        rand_fn = math_utils.sample_gaussian
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # generate random values
    low, high = distribution_params
    rand_values = rand_fn(low, high, data[env_ids][:, dim_ids].shape, data.device)

    # apply operation
    if operation == "add":
        data[env_ids][:, dim_ids] += rand_values
    elif operation == "scale":
        data[env_ids][:, dim_ids] *= rand_values
    elif operation == "abs":
        data[env_ids][:, dim_ids] = rand_values
    else:
        raise ValueError(f"Unknown operation: {operation}")

    return data


def randomize_joint_default_pos(
    env: ManagerBasedRlEnv,  # MJLab: ManagerBasedEnv → ManagerBasedRlEnv
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]  # MJLab: Articulation → Entity

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedRlEnv,  # MJLab: ManagerBasedEnv → ManagerBasedRlEnv
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]  # MJLab: Articulation → Entity
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    # MJLab: root_physx_view.get_coms() not available in MuJoCo — use body_ipos model field instead
    # MJLab: IsaacLab modifies runtime PhysX COM directly; MJLab modifies body_ipos model field
    coms = env.sim.model.body_ipos.clone()  # MJLab: asset.root_physx_view.get_coms() → env.sim.model.body_ipos

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    # MJLab: root_physx_view.set_coms() not available in MuJoCo — assign directly to body_ipos model field
    env.sim.model.body_ipos[:] = coms  # MJLab: asset.root_physx_view.set_coms(coms, env_ids) → direct model field assignment


def randomize_rigid_body_material(
    env: ManagerBasedRlEnv,  # MJLab: ManagerBasedEnv → ManagerBasedRlEnv
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    num_buckets: int,
    asset_cfg: SceneEntityCfg,
    make_consistent: bool = False,
):
    """Randomize the physics materials on all geometries of the asset.

    MJLab port: uses dr.geom_friction to randomize tangential friction.
    Restitution is not directly available as a MuJoCo model field (solref/solimp
    control contact dynamics instead) — the restitution_range parameter is accepted
    but currently unused.
    """
    # MJLab: IsaacLab version samples material buckets and assigns via PhysX
    # root_physx_view.get_material_properties/set_material_properties.
    # MuJoCo uses geom_friction (tangential/torsional/rolling) and solref/solimp.
    # We delegate to dr.geom_friction for the friction component.
    # MJLab note: num_buckets is accepted but continuous uniform distribution is used
    # instead of discrete buckets. MuJoCo's geom_friction applies per-geom randomization
    # from a continuous range, which provides strictly more diversity than bucket sampling.
    # make_consistent is also not applicable — each geom gets independent randomization.
    from mjlab.envs.mdp.dr.geom import geom_friction  # MJLab: lazy import to avoid circular deps

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")

    # Use the average of static/dynamic friction as the tangential friction range.
    # MJLab: geom_friction operates on axes [0] (tangential) by default
    friction_min = (static_friction_range[0] + dynamic_friction_range[0]) / 2
    friction_max = (static_friction_range[1] + dynamic_friction_range[1]) / 2
    geom_friction(
        env,
        env_ids,
        ranges=(friction_min, friction_max),
        asset_cfg=asset_cfg,
        distribution="uniform",
        operation="abs",
        axes=[0],  # MJLab: tangential friction only (restitution via solref/solimp)
    )

    # MJLab: material RGBA color randomization (optional, matches IsaacLab material variation)
    from mjlab.envs.mdp.dr.material import mat_rgba  # MJLab: lazy import

    if asset_cfg.material_ids is not None and len(asset_cfg.material_ids) > 0:
        mat_rgba(
            env,
            env_ids,
            ranges=(0.5, 1.5),
            asset_cfg=asset_cfg,
            distribution="uniform",
            operation="scale",
        )
