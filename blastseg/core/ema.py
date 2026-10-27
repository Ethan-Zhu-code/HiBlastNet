import copy
import math

from .amp import unwrap_model


class ModelEMA:
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = copy.deepcopy(unwrap_model(model)).eval()
        self.updates = updates
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay_fn(self.updates)
        msd = unwrap_model(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * d + (1 - d) * msd[k].detach())

    def state_dict(self):
        return {"ema": self.ema.state_dict(), "updates": self.updates}

    def load_state_dict(self, state):
        self.ema.load_state_dict(state["ema"])
        self.updates = state["updates"]
