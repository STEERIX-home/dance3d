"""Small rigid-geometry helpers (NumPy) shared by the alignment and scene tools."""
import numpy as np


def kabsch(source, target):
    """Rigid transform with target ~ R @ source + t for point sets (N, 3). Returns R, t, rms."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    H = (source - source_mean).T @ (target - target_mean)
    U, _, Vt = np.linalg.svd(H)
    sign = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, sign]) @ U.T
    t = target_mean - R @ source_mean
    rms = float(np.sqrt(((source @ R.T + t - target) ** 2).sum(-1).mean()))
    return R, t, rms


def rot_y(angle):
    """Rotation(s) about the vertical axis y; angle may be an array."""
    angle = np.asarray(angle, dtype=np.float64)
    c, s = np.cos(angle), np.sin(angle)
    R = np.zeros(angle.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 2], R[..., 1, 1], R[..., 2, 0], R[..., 2, 2] = c, s, 1.0, -s, c
    return R


def yaw_of(R):
    """The rotation about y closest to R (least squares): its angle."""
    return np.arctan2(R[..., 0, 2] - R[..., 2, 0], R[..., 0, 0] + R[..., 2, 2])


def unwrap(angles):
    return np.unwrap(np.asarray(angles, dtype=np.float64))


def smooth_weighted(values, weights, sigma):
    """Gaussian smoothing along time of values (L, ...) with per-frame weights (L,): frames with
    weight 0 do not contribute and are filled from their neighbours (normalized convolution)."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    radius = int(np.ceil(sigma * 4))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    flat = values.reshape(len(values), -1)
    padded_w = np.pad(weights, radius, mode='edge')
    padded_v = np.pad(flat * weights[:, None], ((radius, radius), (0, 0)), mode='edge')
    weight_sum = np.convolve(padded_w, kernel, mode='valid')
    out = np.stack([np.convolve(padded_v[:, i], kernel, mode='valid') for i in range(flat.shape[1])], axis=1)
    out = out / np.maximum(weight_sum, 1e-9)[:, None]
    # where no weighted frame lies inside the window, interpolate between the valid frames
    valid = weight_sum > 1e-6
    if not valid.all():
        index = np.arange(len(values))
        for i in range(out.shape[1]):
            out[~valid, i] = np.interp(index[~valid], index[valid], out[valid, i])
    return out.reshape(values.shape)


def axis_angle_to_matrix(vectors):
    """Rodrigues: axis-angle (..., 3) -> rotation matrices (..., 3, 3)."""
    vectors = np.asarray(vectors, dtype=np.float64)
    angle = np.linalg.norm(vectors, axis=-1, keepdims=True)
    axis = vectors / np.maximum(angle, 1e-12)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    K = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=-1).reshape(vectors.shape[:-1] + (3, 3))
    angle = angle[..., None]
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def matrix_to_axis_angle(R):
    """Rotation matrices (..., 3, 3) -> axis-angle (..., 3)."""
    R = np.asarray(R, dtype=np.float64)
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    angle = np.arccos(cos)
    axis = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    small = norm[..., 0] < 1e-8
    axis = axis / np.maximum(norm, 1e-12)
    out = axis * angle[..., None]
    if np.any(small & (cos < 0)):  # rotation by pi: take the axis from the symmetric part
        for index in zip(*np.nonzero(small & (cos < 0))):
            M = (R[index] + np.eye(3)) / 2
            k = int(np.argmax(np.diag(M)))
            v = M[:, k] / np.sqrt(max(M[k, k], 1e-12))
            out[index] = v * np.pi
    out[small & (cos >= 0)] = 0.0
    return out
