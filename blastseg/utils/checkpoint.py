import torch


def read_checkpoint(checkpoint_path, device, weights_only=False):
    load_kwargs = {"map_location": device}
    try:
        return torch.load(checkpoint_path, weights_only=weights_only, **load_kwargs)
    except TypeError:
        return torch.load(checkpoint_path, **load_kwargs)


def selected_weight_kind(checkpoint):
    if not isinstance(checkpoint, dict):
        return "raw"
    kind = checkpoint.get("selection_weight_kind")
    if kind is None:
        # Older training checkpoints used EMA for validation without recording it.
        ema_state = checkpoint.get("ema_state")
        kind = "ema" if isinstance(ema_state, dict) and isinstance(ema_state.get("ema"), dict) else "raw"
    if kind not in {"raw", "ema"}:
        raise ValueError(f"Unsupported selection_weight_kind: {kind}")
    return kind


def extract_state_dict(checkpoint, weights="selected"):
    if weights not in {"selected", "raw", "ema"}:
        raise ValueError(f"Unsupported weights selection: {weights}")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        kind = selected_weight_kind(checkpoint) if weights == "selected" else weights
        if kind == "ema":
            ema_state = checkpoint.get("ema_state")
            if not isinstance(ema_state, dict) or not isinstance(ema_state.get("ema"), dict):
                raise ValueError("EMA weights were requested but are missing from the checkpoint")
            return ema_state["ema"]
        return checkpoint["state_dict"]
    if weights == "ema":
        raise ValueError("EMA weights were requested from a raw state dict")
    return checkpoint


def normalize_checkpoint(checkpoint, default_model_name=None):
    if isinstance(checkpoint, dict):
        return checkpoint

    normalized = {"state_dict": checkpoint}
    if default_model_name is not None:
        normalized["model_name"] = default_model_name
    return normalized


def get_checkpoint_model_identity(checkpoint, *, default_model_name=None, default_deploy=False):
    checkpoint = normalize_checkpoint(checkpoint, default_model_name=default_model_name)
    model_name = checkpoint.get("model_name", default_model_name)
    model_deploy = bool(checkpoint.get("model_deploy", default_deploy))
    return model_name, model_deploy


def load_state_dict_into_model(model, checkpoint, strict=True, weights="selected"):
    state_dict = extract_state_dict(checkpoint, weights=weights)
    model.load_state_dict(state_dict, strict=strict)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    return model, metadata


def load_checkpoint_into_model(model, checkpoint_path, device, weights_only=False, strict=True, weights="selected"):
    checkpoint = read_checkpoint(checkpoint_path, device=device, weights_only=weights_only)
    return load_state_dict_into_model(model, checkpoint, strict=strict, weights=weights)
