import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.tasks import SegmentationModel
from src.frequency_trainer import FrequencyLRTrainer
class BoundaryHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, x):
        return self.net(x)
def make_edge_target(masks, batch_idx, batch_size, size):
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks [B,H,W] or [N,H,W], got {tuple(masks.shape)}")
    if masks.shape[0] == batch_size:
        binary = (masks > 0).float().unsqueeze(1)
    else:
        binary = torch.zeros(
            (batch_size, 1, *masks.shape[-2:]),
            device=masks.device, dtype=torch.float32,
        )
        idx = batch_idx.reshape(-1).long()
        if masks.shape[0] != idx.numel():
            raise RuntimeError(
                "Mask count does not match batch_idx. Set overlap_mask=True for this experiment."
            )
        for i in range(batch_size):
            selected = masks[idx == i]
            if selected.numel():
                binary[i, 0] = selected.amax(dim=0).float().clamp(0, 1)
    binary = F.interpolate(binary, size=size, mode="nearest")
    dilated = F.max_pool2d(binary, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-binary, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0, 1)


class BoundarySegmentationModel(SegmentationModel):
    def __init__(self, cfg, ch=3, nc=None, verbose=True,
                 edge_layer=15, edge_weight=0.1, edge_dice_weight=1.0):
        self._edge_logits = None
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.edge_layer = int(edge_layer)
        self.edge_weight = float(edge_weight)
        self.edge_dice_weight = float(edge_dice_weight)
        if self.edge_weight < 0 or self.edge_dice_weight < 0:
            raise ValueError("Edge loss coefficients must be nonnegative")

        p3 = self.model[self.edge_layer]
        try:
            in_channels = p3.cv2.conv.out_channels 
        except AttributeError as exc:
            raise RuntimeError(
                f"Layer {self.edge_layer} is not the expected P3 C2f. "
                "Inspect model.model and change edge_layer."
            ) from exc
        self.boundary_head = BoundaryHead(in_channels)

    def _capture_p3(self, module, inputs, output):
        self._edge_logits = self.boundary_head(output)

    def predict(self, x, profile=False, augment=False, embed=None):
        capture = (
            hasattr(self, "boundary_head")
            and self.training
            and torch.is_grad_enabled()
        )
        if not capture:
            return super().predict(x, profile=profile, augment=augment, embed=embed)

        self._edge_logits = None
        handle = self.model[self.edge_layer].register_forward_hook(self._capture_p3)
        try:
            return super().predict(x, profile=profile, augment=augment, embed=embed)
        finally:
            handle.remove()

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if not self.training or not torch.is_grad_enabled():
            if preds is None:
                preds = super().predict(batch["img"])
            return self.criterion(preds, batch)

        if preds is None:
            self._edge_logits = None
            hook = self.model[self.edge_layer].register_forward_hook(self._capture_p3)
            try:
                preds = super().predict(batch["img"])
            finally:
                hook.remove()
        if self._edge_logits is None:
            hook = self.model[self.edge_layer].register_forward_hook(self._capture_p3)
            try:
                _ = super().predict(batch["img"])
            finally:
                hook.remove()
        if self._edge_logits is None:
            raise RuntimeError(
                "Boundary Head did not receive P3 features. "
                f"Check edge_layer={self.edge_layer} and model hooks."
            )
        yolo_loss, items = self.criterion(preds, batch)
        logits = self._edge_logits.float()
        self._edge_logits = None  
        batch_size = int(logits.shape[0])
        target = make_edge_target(
            batch["masks"], batch["batch_idx"], batch_size, logits.shape[-2:]
        )
        n_pos = target.sum().detach()
        n_neg = torch.tensor(target.numel(), device=target.device) - n_pos
        pos_weight = (n_neg / n_pos.clamp_min(1)).clamp(1, 20)
        edge_bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight
        )
        prob = logits.sigmoid()
        intersection = (prob * target).sum(dim=(1, 2, 3))
        denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        edge_dice = (1 - (2 * intersection + 1) / (denom + 1)).mean()
        edge_loss = edge_bce + self.edge_dice_weight * edge_dice
        total = yolo_loss + self.edge_weight * edge_loss * batch_size
        self.last_edge_loss = edge_loss.detach()
        return total, items


class BoundaryFrequencyTrainer(FrequencyLRTrainer):
    edge_weight = 0.1
    edge_dice_weight = 1.0
    edge_layer = 15
    main_lr = 5e-4
    freq_lr = 1e-3
    bifpn_lr = 7e-4
    def get_model(self, cfg=None, weights=None, verbose=True):
        net = BoundarySegmentationModel(
            cfg=cfg,
            nc=self.data["nc"],
            verbose=verbose,
            edge_layer=self.edge_layer,
            edge_weight=self.edge_weight,
            edge_dice_weight=self.edge_dice_weight,
        )
        if weights is not None:
            net.load(weights)
        print(
            f"[Aux Boundary Head] P3 layer={self.edge_layer}, "
            f"edge_weight={self.edge_weight}, "
            f"edge_dice_weight={self.edge_dice_weight}"
        )
        return net
