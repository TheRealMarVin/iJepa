import torch
from torch.utils.data import Dataset

from helpers.patch_helpers import generate_context_and_targets


class IJEPADatasetWrapper(Dataset):
    def __init__(self, base_dataset, config):
        self.base = base_dataset

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

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]

        context_indices, target_indices_list = generate_context_and_targets(
            nb_patches=(self.nb_patches_w, self.nb_patches_h),
            min_context_height=self.min_context_h,
            max_context_height=self.max_context_h,
            min_context_width=self.min_context_w,
            max_context_width=self.max_context_w,
            nb_targets=self.nb_targets,
            min_target_height=self.min_target_h,
            max_target_height=self.max_target_h,
            min_target_width=self.min_target_w,
            max_target_width=self.max_target_w,
            max_tries_per_block=self.max_tries_per_block,
        )

        context_indices = torch.as_tensor(context_indices, dtype=torch.long)
        target_indices_list = [torch.as_tensor(t, dtype=torch.long) for t in target_indices_list]

        return img, context_indices, target_indices_list

    @staticmethod
    def _check_min_max(name, min_v, max_v):
        if min_v > max_v:
            raise ValueError(f"{name}: min ({min_v}) > max ({max_v})")
import torch
from torch.utils.data import Dataset


class IJEPADatasetWrapper(Dataset):
    def __init__(self, base_dataset, config):
        self.base = base_dataset

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

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]

        context_indices, target_indices_list = generate_context_and_targets(
            nb_patches=(self.nb_patches_w, self.nb_patches_h),
            min_context_height=self.min_context_h,
            max_context_height=self.max_context_h,
            min_context_width=self.min_context_w,
            max_context_width=self.max_context_w,
            nb_targets=self.nb_targets,
            min_target_height=self.min_target_h,
            max_target_height=self.max_target_h,
            min_target_width=self.min_target_w,
            max_target_width=self.max_target_w,
            max_tries_per_block=self.max_tries_per_block,
        )

        context_indices = torch.as_tensor(context_indices, dtype=torch.long)
        target_indices_list = [torch.as_tensor(t, dtype=torch.long) for t in target_indices_list]

        return img, context_indices, target_indices_list

    @staticmethod
    def _check_min_max(name, min_v, max_v):
        if min_v > max_v:
            raise ValueError(f"{name}: min ({min_v}) > max ({max_v})")
