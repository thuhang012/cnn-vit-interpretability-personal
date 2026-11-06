"""Heatmap extraction and crop geometry; model/data loading stays in src."""

from contextlib import contextmanager
import numpy as np

BLOCKS = (2, 5, 8, 11)


@contextmanager
def _attention_capture(model, rollout):
    """Scoped MHA hooks; retain selected CLS maps and only one rollout matrix."""
    import torch
    handles, state = [], {"maps": {}, "shapes": {}, "row_sum_error": 0.}
    grid = model.image_size // model.patch_size
    tokens = grid * grid + 1

    def request(module, args, kwargs):
        return args, {**kwargs, "need_weights": True, "average_attn_weights": False}

    def collect(module, args, output, index):
        weights = output[1].detach()
        if tuple(weights.shape[1:]) != (12, tokens, tokens):
            raise ValueError(f"Unexpected attention shape: {tuple(weights.shape)}")
        if not weights.isfinite().all() or (weights < 0).any():
            raise ValueError("Nonfinite/negative attention")
        error = (weights.sum(-1) - 1).abs().max().item()
        if error > 1e-4:
            raise ValueError("Attention rows do not sum to one")
        state["row_sum_error"] = max(state["row_sum_error"], error)
        mean = weights.mean(1)
        if index in BLOCKS:
            key = f"block_{index + 1}"
            state["maps"][key] = mean[:, 0, 1:].reshape(-1, grid, grid).cpu().numpy()
            state["shapes"][key] = list(weights.shape)
        if rollout:
            matrix = mean + torch.eye(tokens, device=mean.device)[None]
            matrix = matrix / matrix.sum(-1, keepdim=True)
            state["joint"] = matrix if index == 0 else matrix @ state["joint"]

    try:
        for index, block in enumerate(model.encoder.layers):
            if rollout or index in BLOCKS:
                handles.append(block.self_attention.register_forward_pre_hook(request, with_kwargs=True))
                handles.append(block.self_attention.register_forward_hook(
                    lambda m, a, o, i=index: collect(m, a, o, i)))
        yield state
    finally:
        for handle in handles:
            handle.remove()


def attention(model, inputs, rollout=False):
    import torch
    if model.training:
        raise ValueError("Attention extraction requires model.eval()")
    with _attention_capture(model, rollout) as state, torch.no_grad():
        logits = model(inputs).cpu()
    if not logits.isfinite().all():
        raise ValueError("Nonfinite logits")
    if rollout:
        grid = model.image_size // model.patch_size
        state["maps"]["rollout"] = state.pop("joint")[:, 0, 1:].reshape(-1, grid, grid).cpu().numpy()
    return {**state, "logits": logits}


def preprocess(image, transform):
    """Official tensor plus exact resize/crop geometry in original edge coordinates."""
    from torchvision.transforms import functional as TF
    resized = TF.resize(image, transform.resize_size, interpolation=transform.interpolation,
                        antialias=transform.antialias)
    rw, rh = resized.size
    crop = transform.crop_size
    ch, cw = (crop[0], crop[0]) if len(crop) == 1 else crop
    if rw < cw or rh < ch:
        raise ValueError("Unexpected padding in official transform")
    left, top = int(round((rw - cw) / 2)), int(round((rh - ch) / 2))
    w, h = image.size
    return transform(image), {
        "original_size": [w, h], "resized_size": [rw, rh], "crop_offset": [left, top],
        "crop_size": [cw, ch],
        "box": [left * w / rw, top * h / rh, (left + cw) * w / rw, (top + ch) * h / rh],
    }


def sample_grid(cam, geometry, size=7, region=None):
    """Bilinear sample at region cell centers, equivalent to align_corners=False.

    Border replication handles the native grid boundary; unseen image regions are
    rejected rather than filled with zero. Geometry uses original pixel edges.
    """
    values = np.asarray(cam, dtype=np.float64)
    if values.ndim != 2 or not values.size or size < 1:
        raise ValueError("Expected a nonempty 2D map and positive grid size")
    own = geometry["box"]
    region = own if region is None else region
    if (region[0] < own[0] - 1e-7 or region[1] < own[1] - 1e-7
            or region[2] > own[2] + 1e-7 or region[3] > own[3] + 1e-7
            or region[2] <= region[0] or region[3] <= region[1]):
        raise ValueError("Invalid region or outside model crop")
    if np.isfinite(values).all() and values.max() > 0:
        values = values / values.max()
    h, w = values.shape
    x = region[0] + (np.arange(size) + .5) / size * (region[2] - region[0])
    y = region[1] + (np.arange(size) + .5) / size * (region[3] - region[1])
    x = np.clip((x - own[0]) / (own[2] - own[0]) * w - .5, 0, w - 1)
    y = np.clip((y - own[1]) / (own[3] - own[1]) * h - .5, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    dx, dy = (x - x0)[None, :], (y - y0)[:, None]
    return ((1 - dy) * ((1 - dx) * values[y0[:, None], x0] + dx * values[y0[:, None], x1])
            + dy * ((1 - dx) * values[y1[:, None], x0] + dx * values[y1[:, None], x1]))


def map_status(cam):
    values = np.asarray(cam, dtype=float)
    if values.ndim != 2 or not values.size:
        return "invalid_shape"
    if not np.isfinite(values).all():
        return "nonfinite"
    if values.min() < 0:
        return "negative"
    if values.max() == 0:
        return "all_zero"
    return "near_constant" if np.ptp(values / values.max()) <= 1e-6 else "ok"
