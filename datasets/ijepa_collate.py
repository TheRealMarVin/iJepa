import torch

from helpers.patch_helpers import generate_context_and_targets


class JepaCollate:
    def __init__(self, config):
        if "nb_patches" not in config:
            raise KeyError("config must contain 'nb_patches'")
        if "context" not in config or "target" not in config:
            raise KeyError("config must contain 'context' and 'target'")

        self.nb_patches_w, self.nb_patches_h = config["nb_patches"]

        context = config["context"]
        target = config["target"]

        self.min_context_h = context["min_h"]
        self.max_context_h = context["max_h"]
        self.min_context_w = context["min_w"]
        self.max_context_w = context["max_w"]

        self.nb_targets = target["nb_targets"]
        self.min_target_h = target["min_h"]
        self.max_target_h = target["max_h"]
        self.min_target_w = target["min_w"]
        self.max_target_w = target["max_w"]
        self.max_tries_per_block = target.get("max_tries", 20)

        self._check_min_max("context_h", self.min_context_h, self.max_context_h)
        self._check_min_max("context_w", self.min_context_w, self.max_context_w)
        self._check_min_max("target_h", self.min_target_h, self.max_target_h)
        self._check_min_max("target_w", self.min_target_w, self.max_target_w)

    def __call__(self, batch):
        images = torch.stack(batch, dim=0)

        context_indices, target_indices_list = generate_context_and_targets(
            nb_patches=(self.nb_patches_w, self.nb_patches_h),
            context_min_height=self.min_context_h,
            context_max_height=self.max_context_h,
            context_min_width=self.min_context_w,
            context_max_width=self.max_context_w,
            nb_targets=self.nb_targets,
            target_min_height=self.min_target_h,
            target_max_height=self.max_target_h,
            target_min_width=self.min_target_w,
            target_max_width=self.max_target_w,
            max_tries_per_block=self.max_tries_per_block)

        context_indices = torch.as_tensor(context_indices, dtype=torch.long).repeat(len(batch), 1)
        target_indices_list = [torch.as_tensor(t, dtype=torch.long).unsqueeze(0).repeat(len(batch), 1) for t in target_indices_list]

        return images, context_indices, target_indices_list

    @staticmethod
    def _check_min_max(name, min_v, max_v):
        if min_v > max_v:
            raise ValueError(f"{name}: min ({min_v}) > max ({max_v})")
