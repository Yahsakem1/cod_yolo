import torch
from torch import nn
from ultralytics.nn.modules import C2f
class FrequencyBranch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction: int = 4,
        dropout: float = 0.1,
        cutoff: float = 0.20,
        mode: str = "fft",
    ):
        super().__init__()
        if mode not in {"fft", "conv_only", "high_only", "low_only"}:
            raise ValueError(f"Unsupported frequency mode: {mode}")
        if not 0.0 < cutoff < 0.5:
            raise ValueError("cutoff must be between 0 and 0.5")
        inner = max(8, in_channels // reduction)
        self.cutoff = float(cutoff)
        self.mode = mode
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.SiLU(),
        )
        self.high_gain = nn.Parameter(torch.full((1, inner, 1, 1), 0.25))
        self.low_gain = nn.Parameter(torch.zeros(1, inner, 1, 1))
        self.expand = nn.Sequential(
            nn.Conv2d(inner, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        z = self.reduce(x)
        mode = getattr(self, "mode", "fft") 
        if mode == "conv_only":
            filtered = z 
        else:
            h, w = z.shape[-2:]
            with torch.autocast(device_type=z.device.type, enabled=False):
                spectrum = torch.fft.rfft2(z.float(), norm="ortho")
                fy = torch.fft.fftfreq(h, device=z.device, dtype=torch.float32)
                fx = torch.fft.rfftfreq(w, device=z.device, dtype=torch.float32)
                radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
                high_mask = torch.sigmoid(30.0 * (radius - self.cutoff))[None, None]
                low_mask = 1.0 - high_mask
                if mode == "fft":
                    response = (1.0 + self.high_gain.float() * high_mask
                                + self.low_gain.float() * low_mask)
                elif mode == "high_only":
                    response = 1.0 + self.high_gain.float() * high_mask
                elif mode == "low_only":
                    response = 1.0 + self.low_gain.float() * low_mask
                else:
                    raise ValueError(f"Unexpected mode={mode}")
                filtered = torch.fft.irfft2(
                    spectrum * response, s=(h, w), norm="ortho"
                )
        out = self.expand(filtered.to(input_dtype))
        dropout = getattr(self, "dropout", None)
        return dropout(out) if dropout is not None else out
class FreqC2f(C2f):
    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5,
        dropout=0.1, mode="fft", cutoff=0.20,
    ):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.freq = FrequencyBranch(
            c1, c2, reduction=4, dropout=dropout, cutoff=cutoff, mode=mode
        )
        self.freq_scale = nn.Parameter(torch.full((1, c2, 1, 1), 0.01))
        print(f"FreqC2f c1={c1}, c2={c2}, n={n}, mode={mode}, "
              f"cutoff={cutoff}, dropout={dropout}")

    def forward(self, x):
        spatial = super().forward(x)
        frequency = self.freq(x)
        return spatial + self.freq_scale * frequency
