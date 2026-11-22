import torch
import torch.nn as nn
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils.torch_utils import unwrap_model
class FrequencyLRTrainer(SegmentationTrainer):
    main_lr = 5e-4
    freq_lr = 7e-4
    bifpn_lr = 7e-4
    
    def build_optimizer(
        self,
        model,
        name="AdamW",
        lr=0.001,
        momentum=0.9,
        decay=1e-5,
        iterations=1e5,
    ):
        model = unwrap_model(model)
        if str(name).lower() != "adamw":
            raise ValueError(
                "FrequencyLRTrainer supports only AdamW"
            )
        groups = {
            "main_decay": [],
            "main_no_decay": [],
            "freq_decay": [],
            "freq_no_decay": [],
            "bifpn_decay": [],
            "bifpn_no_decay": [],
        }
        norm_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.LayerNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        )
        assigned_ids = set()
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                full_name = (f"{module_name}.{param_name}" if module_name else param_name)
                if (".freq." in full_name or full_name.endswith(".freq_scale")):
                    prefix = "freq"
                else:
                    prefix = "main"
                no_decay = (param_name == "bias" or isinstance(module, norm_types) or param.ndim <= 1)
                suffix = "no_decay" if no_decay else "decay"
                groups[f"{prefix}_{suffix}"].append(param)
                assigned_ids.add(id(param))
        expected_ids = {id(p) for p in model.parameters() if p.requires_grad}
        if assigned_ids != expected_ids:
            raise RuntimeError("Not all trainable parameters assigned")
        if not (groups["freq_decay"]or groups["freq_no_decay"]):
            raise RuntimeError("Frequency parameters not found")
        param_groups = []
        for group_name, params in groups.items():
            if not params:
                continue
            if group_name.startswith("freq"):
                group_lr = self.freq_lr
            elif group_name.startswith("bifpn"):
                group_lr = self.bifpn_lr
            else:
                group_lr = self.main_lr
            group_decay = (decay if group_name.endswith("_decay") else 0.0)
            param_groups.append({
                "params": params,
                "lr": group_lr,
                "weight_decay": group_decay,
                "group_name": group_name,
                "param_group": (
                    "bias" if group_name.endswith("no_decay")
                    and False else "other"
                ),
            })
            print(
                f"{group_name}: "
                f"params={sum(p.numel() for p in params):,}, "
                f"lr={group_lr:.6f}, "
                f"weight_decay={group_decay:.6f}")
        optimizer = torch.optim.AdamW(param_groups,lr=self.main_lr,betas=(momentum, 0.999),)
        return optimizer
