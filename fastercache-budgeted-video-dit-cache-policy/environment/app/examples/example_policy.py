class CachePolicy:
    def __init__(self, num_steps, budget, device, dtype):
        self.num_steps = num_steps
        self.budget = budget
        self.device = device
        self.dtype = dtype
        self.last_cond = None
        self.last_uncond = None

    def decide(self, step, state):
        return "full" if state["units_remaining"] >= 2 else "skip"

    def reconstruct(self, step, mode, computed, state):
        if "cond" in computed:
            self.last_cond = computed["cond"]
        if "uncond" in computed:
            self.last_uncond = computed["uncond"]
        return self.last_cond, self.last_uncond
