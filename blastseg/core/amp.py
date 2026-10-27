import platform

import torch


def get_gpu_memory_summary(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return "N/A"

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return f"alloc={allocated:.2f}G, reserved={reserved:.2f}G, max_alloc={max_allocated:.2f}G"


def should_enable_amp(device, amp_mode="auto"):
    if amp_mode not in {"auto", "on", "off"}:
        raise ValueError(f"Unsupported amp_mode: {amp_mode}")

    if amp_mode == "off":
        return False

    if device.type != "cuda":
        if amp_mode == "on":
            print("AMP was requested, but CUDA is unavailable. Falling back to full precision.")
        return False

    return True


def maybe_compile_module(module, device, compile_mode="auto"):
    if compile_mode not in {"auto", "on", "off"}:
        raise ValueError(f"Unsupported compile_mode: {compile_mode}")

    if compile_mode == "off":
        return module, False, "torch.compile disabled by configuration."

    if not hasattr(torch, "compile"):
        reason = "Current PyTorch build does not provide torch.compile."
        if compile_mode == "on":
            print(f"torch.compile requested but unavailable. {reason}")
        return module, False, reason

    if device.type != "cuda":
        reason = "torch.compile is only enabled for CUDA runs in this project."
        if compile_mode == "on":
            print(f"torch.compile requested but skipped. {reason}")
        return module, False, reason

    if platform.system() == "Windows":
        reason = "torch.compile is skipped on Windows in this project for stability and backend compatibility."
        if compile_mode == "on":
            print(f"torch.compile requested but skipped. {reason}")
        return module, False, reason

    try:
        compiled_module = torch.compile(module)
        return compiled_module, True, "torch.compile enabled."
    except Exception as exc:
        reason = f"torch.compile failed: {exc}"
        if compile_mode == "on":
            print(f"torch.compile requested but failed. {reason}")
        return module, False, reason


def unwrap_model(module):
    return getattr(module, "_orig_mod", module)
