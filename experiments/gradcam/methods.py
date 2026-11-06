"""Heatmap extraction and crop geometry; model/data loading stays in src."""

import numpy as np

STAGES = ("early", "middle", "late", "near_head")
BLOCKS = (2, 5, 8, 11)


def cam_layers(model, name, layerwise=False):
    if name == "resnet50":
        layers = {s: getattr(model, f"layer{i}")[-1] for i, s in enumerate(STAGES, 1)}
    elif name == "vit_b_16":
        layers = {s: model.encoder.layers[i].ln_1 for s, i in zip(STAGES, BLOCKS)}
    else:
        raise ValueError(f"Unknown model: {name}")
    return layers if layerwise else {"near_head": layers["near_head"]}


def gradcam(model, inputs, labels, name, layerwise=False):
    """Raw native-grid CAMs from ground-truth logits; one backward per batch.

    Model must be in eval mode. Samples are independent, so differentiating the
    sum of selected logits preserves each image's individual target gradient.
    ViT ln_1 is BEFORE attention, not the output of the final encoder block.
    """
    import torch
    if model.training:
        raise ValueError("Grad-CAM requires model.eval()")
    layers, features, handles = cam_layers(model, name, layerwise), {}, []
    try:
        for key, layer in layers.items():
            def capture(module, args, output, key=key):
                features[key] = output
            handles.append(layer.register_forward_hook(capture))
        with torch.enable_grad():
            logits = model(inputs.detach().requires_grad_(True))
            if not logits.isfinite().all():
                raise ValueError("Nonfinite logits")
            gradients = torch.autograd.grad(logits.gather(1, labels[:, None]).sum(),
                                            tuple(features[key] for key in layers))
        maps, diagnostics = {}, {}
        for key, gradient in zip(layers, gradients):
            activation, gradient = features[key].detach(), gradient.detach()
            if name == "vit_b_16":
                grid = model.image_size // model.patch_size
                if activation.shape[1] != grid * grid + 1:
                    raise ValueError("Unexpected CLS/patch token count")
                activation = activation[:, 1:].transpose(1, 2).reshape(len(inputs), -1, grid, grid)
                gradient = gradient[:, 1:].transpose(1, 2).reshape_as(activation)
            cam = (activation * gradient.mean((2, 3), keepdim=True)).sum(1).relu()
            maps[key] = cam.cpu().numpy()  # no normalization hiding zero CAMs
            diagnostics[key] = {
                "activation_shape": list(activation.shape[1:]),
                "gradient_shape": list(gradient.shape[1:]),
                "gradient_abs_mean": gradient.abs().mean((1, 2, 3)).cpu().tolist(),
                "finite": (activation.isfinite().flatten(1).all(1)
                           & gradient.isfinite().flatten(1).all(1)).cpu().tolist(),
            }
        return {"logits": logits.detach().cpu(), "maps": maps, "diagnostics": diagnostics}
    finally:
        for handle in handles:
            handle.remove()


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


def crop_intersection(first, second):
    if first["original_size"] != second["original_size"]:
        raise ValueError("Maps refer to different original image sizes")
    a, b = first["box"], second["box"]
    box = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("Crops do not overlap")
    return box


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
