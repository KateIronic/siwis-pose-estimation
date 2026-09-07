"""
Loss functions exactly as defined in §3.2.3.
"""

from __future__ import annotations

import config
import torch


def mask_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha1: float = config.MASK_BCE_WEIGHT_ALPHA1,
    alpha2: float = config.MASK_DICE_WEIGHT_ALPHA2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Eqn. (12):
      L_mask = alpha1 * sum_i [y_i log x_i + (1-y_i) log(1-x_i)]
             + alpha2 * (1 - 2*sum(x_i*y_i) / (sum(x_i^2) + sum(y_i^2)))
    x_i = predicted value (pred, in (0,1)), y_i = ground-truth label (target).
    """
    x = torch.clamp(pred, eps, 1 - eps)
    y = target

    bce = -(y * torch.log(x) + (1 - y) * torch.log(1 - x)).sum()

    intersection = (x * y).sum()
    dice = 1 - (2 * intersection) / (x.pow(2).sum() + y.pow(2).sum() + eps)

    return alpha1 * bce + alpha2 * dice


def weighted_mse_heatmap_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    First term of Eqn. (13): (y + 1) (*) ||y_hat - y||_2^2 -- elementwise
    product of (ground-truth heatmap + 1) with the squared error, giving
    higher weight near keypoints (where y is large) than background
    (where y ~ 0), as motivated in §3.2.3.
    """
    weight = target + 1.0
    return (weight * (pred - target).pow(2)).mean()


def associative_embedding_loss(
    tags: torch.Tensor,
    keypoint_coords: torch.Tensor,
    keypoint_visible: torch.Tensor,
) -> torch.Tensor:
    """
    L_group (§3.2.3: "following the approach outlined in [33], we use
    L_group for keypoint grouping ... grouping keypoints with smaller
    l2 distances between their tags"). Implements the pull/push
    associative-embedding loss of Newell et al. 2017 [33]:
      - pull: tags of joints belonging to the same person are pulled
        towards that person's mean tag.
      - push: mean tags of different people are pushed apart.

    tags: (B, K, H, W) predicted tag maps, one channel per keypoint type.
    keypoint_coords: (B, N_people, K, 2) integer (row, col) location of
        each joint in the tag map.
    keypoint_visible: (B, N_people, K) bool visibility mask.
    """
    B, N, K, _ = keypoint_coords.shape
    pull_loss = tags.new_tensor(0.0)
    push_loss = tags.new_tensor(0.0)
    n_valid_people = 0

    for b in range(B):
        person_mean_tags = []
        for n in range(N):
            vis = keypoint_visible[b, n]                # (K,) bool
            k_idx = vis.nonzero(as_tuple=True)[0]         # visible joint channel indices
            if len(k_idx) == 0:
                continue
            rows = keypoint_coords[b, n, k_idx, 0]
            cols = keypoint_coords[b, n, k_idx, 1]
            joint_tags = tags[b, k_idx, rows, cols]        # (num_visible,)
            mean_tag = joint_tags.mean()
            pull_loss = pull_loss + ((joint_tags - mean_tag) ** 2).mean()
            person_mean_tags.append(mean_tag)
            n_valid_people += 1

        if len(person_mean_tags) > 1:
            means = torch.stack(person_mean_tags)
            n_people = len(means)
            diff = means.unsqueeze(0) - means.unsqueeze(1)
            off_diag = ~torch.eye(n_people, dtype=torch.bool, device=means.device)
            push_loss = push_loss + torch.exp(-diff.pow(2))[off_diag].sum() / (n_people * (n_people - 1))

    denom = max(n_valid_people, 1)
    return pull_loss / denom + push_loss / max(B, 1)


def pose_loss(
    pred_heatmaps: torch.Tensor,
    target_heatmaps: torch.Tensor,
    pred_tags: torch.Tensor,
    keypoint_coords: torch.Tensor,
    keypoint_visible: torch.Tensor,
) -> torch.Tensor:
    """Eqn. (13): L_pose = (y+1) (*) ||y_hat - y||_2^2 + L_group."""
    mse_term = weighted_mse_heatmap_loss(pred_heatmaps, target_heatmaps)
    group_term = associative_embedding_loss(pred_tags, keypoint_coords, keypoint_visible)
    # §4.1.2: "the associative embedding loss is weighted by a factor
    # of 1e-3 relative to the MSE loss of the keypoint detection heatmaps"
    return mse_term + config.ASSOC_EMBED_LOSS_WEIGHT * group_term


def total_loss(
    mask_pred,
    mask_target,
    pose_heatmap_pred,
    pose_heatmap_target,
    pose_tags_pred,
    keypoint_coords,
    keypoint_visible,
    lam: float = config.POSE_LOSS_LAMBDA,
):
    """Eqn. (14): L = L_mask + lambda * L_pose."""
    l_mask = mask_loss(mask_pred, mask_target)
    l_pose = pose_loss(
        pose_heatmap_pred, pose_heatmap_target, pose_tags_pred, keypoint_coords, keypoint_visible
    )
    return l_mask + lam * l_pose, {"mask_loss": l_mask.item(), "pose_loss": l_pose.item()}