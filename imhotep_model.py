import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP

FEATURES = ["T2M", "PRECTOTCORR", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN"]
TARGETS = ["dry_hot", "heavy_rain", "fire_weather"]
SEQ_LEN = 30


class ImhotepMultiCfC(nn.Module):
    def __init__(self, input_size: int, units: int = 19, output_size: int = 3):
        super().__init__()
        wiring = AutoNCP(units, output_size)
        self.cfc = CfC(input_size, wiring, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.cfc(x)
        return out[:, -1, :]
