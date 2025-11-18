import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

# ----------------------------
# Small building blocks
# ----------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.block.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.block(x)


class DeconvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=4, s=2, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.block.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.block(x)
    

# ----------------------------
# Pooling / Unpooling
# ----------------------------

class ChannelPool(nn.Module):
    """
    Learnable channel pooling using a 1x1 convolution.
    Reduces the number of channels while preserving spatial structure.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after pooling.
        norm (bool): If True, adds BatchNorm2d.
        activation (bool): If True, adds ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 norm: bool = True, activation: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.block.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, C_in, H, W]
        Returns:
            Tensor of shape [B, C_out, H, W]
        """
        return self.block(x)


class ChannelUnPool(nn.Module):
    """
    Learnable channel unpooling using a 1x1 convolution.
    Expands the number of channels while preserving spatial structure.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after unpooling.
        norm (bool): If True, adds BatchNorm2d.
        activation (bool): If True, adds ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 norm: bool = True, activation: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.block.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, C_in, H, W]
        Returns:
            Tensor of shape [B, C_out, H, W]
        """
        return self.block(x)


# ----------------------------
# Encoder
# ----------------------------

class Encoder(nn.Module):
    """
    Baseline Conv Encoder:
      Input:  B x 3 x H x W  (any H, W)
      Output: z (B x z_dim), context dict for Decoder
    """
    def __init__(self, z_dim: int = 512, in_ch: int = 3, base_ch: int = 64, gap_ch: int = 1):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch*2, base_ch*4  # 64, 128, 256
        self.gap_ch = gap_ch
        self.z_dim = z_dim

        # Stage 0 (no downsample)
        self.conv0 = ConvBNReLU(in_ch, c1, k=3, s=1, p=1)

        # Down 1: /2
        self.down1 = ConvBNReLU(c1, c2, k=3, s=2, p=1)
        self.conv1 = ConvBNReLU(c2, c2, k=3, s=1, p=1)

        # Down 2: /2
        self.down2 = ConvBNReLU(c2, c3, k=3, s=2, p=1)
        self.conv2 = ConvBNReLU(c3, c3, k=3, s=1, p=1)

        # Down 3: /2  -> final 256 x (H/8) x (W/8)
        # self.down3 = ConvBNReLU(c3, c3, k=3, s=2, p=1)
        # self.conv3 = ConvBNReLU(c3, c3, k=3, s=1, p=1)

        # Bottleneck projection: GAP -> Linear to z
        self.gap = ChannelPool(c3, gap_ch, norm=False, activation=False)
        self.fc_mu = None

    def _build_fc_if_needed(self, Hb: int, Wb: int):
        in_features = Hb * Wb * self.gap_ch
        if self.fc_mu is None:
            device = next(self.parameters()).device
            self.fc_mu = nn.Linear(in_features, self.z_dim).to(device)
            # nn.init.kaiming_normal_(self.fc_mu.weight, nonlinearity='linear')
            # nn.init.zeros_(self.fc_mu.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, C, H, W = x.shape

        # 2) conv pyramid
        x = self.conv0(x)       # 64, H,   W
        x = self.down1(x)       # 128, H/2, W/2
        x = self.conv1(x)
        x = self.down2(x)       # 256, H/4, W/4
        x = self.conv2(x)
        # x = self.down3(x)       # 256, H/8, W/8
        # x = self.conv3(x)

        # 3) record spatial size at the bottleneck
        Hb, Wb = x.shape[-2:]

        # 4) global average pool -> linear to z
        pooled = self.gap(x).flatten(1)   # B x 256
        self._build_fc_if_needed(Hb, Wb)
        z = self.fc_mu(pooled)            # B x z_dim

        # context for Decoder
        context = {
            "orig_hw": (H, W),
            "bottleneck_hw": (Hb, Wb)
        }
        return z#, context


# ----------------------------
# Decoder
# ----------------------------

class Decoder(nn.Module):
    """
    Baseline Conv Decoder:
      Input:  z (B x z_dim), context dict from Encoder
      Output: x_hat (B x 3 x H x W) cropped to original size
    """
    def __init__(self, z_dim: int = 512, out_ch: int = 3, base_ch: int = 64, gap_ch: int = 1, out_act: str = "sigmoid"):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch*2, base_ch*4  # 64, 128, 256
        self.gap_ch = gap_ch
        self.z_dim = z_dim
        self.out_act = out_act

        # We don't know Hb, Wb at init; we'll create fc dynamically on first call.
        self.fc = None
        self._cached_shape = None  # (Hb, Wb)
        self.gap = ChannelUnPool(gap_ch, c3, norm=False, activation=False)

        # Up path (mirror of encoder)
        # self.up1 = DeconvBNReLU(c3, c3, k=4, s=2, p=1)  # H/8 -> H/4
        # self.conv1 = ConvBNReLU(c3, c3, k=3, s=1, p=1)

        self.up2 = DeconvBNReLU(c3, c2, k=4, s=2, p=1)  # H/4 -> H/2
        self.conv2 = ConvBNReLU(c2, c2, k=3, s=1, p=1)

        self.up3 = DeconvBNReLU(c2, c1, k=4, s=2, p=1)  # H/2 -> H
        self.conv3 = ConvBNReLU(c1, c1, k=3, s=1, p=1)

        self.out_conv = nn.Conv2d(c1, out_ch, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _build_fc_if_needed(self, Hb: int, Wb: int):
        if self._cached_shape == (Hb, Wb) and self.fc is not None:
            return
        device = next(self.parameters()).device
        self.fc = nn.Linear(self.z_dim, self.gap_ch * Hb * Wb).to(device)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)
        self._cached_shape = (Hb, Wb)

    def forward(self, z: torch.Tensor, context: Dict = {
                "orig_hw": (224, 176),
                "bottleneck_hw": (16, 16)
            }
        ) -> torch.Tensor:
        Hb, Wb = context["bottleneck_hw"]
        H, W = context["orig_hw"]

        # 1) FC to bottleneck map
        self._build_fc_if_needed(Hb, Wb)
        x = self.fc(z)
        x = x.view(-1, self.gap_ch, Hb, Wb)  # B x gap_ch x Hb x Wb
        x = self.gap(x)

        # 2) upsampling path
        # x = self.up1(x)
        # x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        x = self.conv3(x)

        # At this point, x should be B x (base_ch//2) x H x W
        x = self.out_conv(x)

        if self.out_act == "sigmoid":
            x = torch.sigmoid(x)
        elif self.out_act == "tanh":
            x = torch.tanh(x)
        else:
            # linear output
            pass
        return x
