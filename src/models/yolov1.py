import torch
from torch import nn

def conv_block(in_ch: int, out_ch: int, k: int, stride: int) -> nn.Sequential:
    # BatchNorm is not in the paper, but training from scratch barely converges without it.
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1),
    )


class YoloV1(nn.Module):
    """YOLOv1 with one box per cell.

    Output: (N, S, S, 5 + C), each cell holding [x, y, sqrt_w, sqrt_h, conf, class scores...],
    where x, y are offsets within the cell and w, h are relative to the image.
    """

    def __init__(self, S: int = 7, C: int = 80, head_hidden: int = 4096, dropout: float = 0.5):
        super().__init__()
        self.S, self.C = S, C

        # 24 conv layers, as in the YOLOv1 paper. 448x448 input -> 7x7x1024.
        # Shapes below are (C, H, W) per image for a (3, 448, 448) input, with S=7, C=80.
        self.backbone = nn.Sequential(
            conv_block(3, 64, 7, 2),        # (64, 224, 224)
            nn.MaxPool2d(2, 2),             # (64, 112, 112)

            conv_block(64, 192, 3, 1),      # (192, 112, 112)
            nn.MaxPool2d(2, 2),             # (192, 56, 56)

            conv_block(192, 128, 1, 1),     # (128, 56, 56)
            conv_block(128, 256, 3, 1),     # (256, 56, 56)
            conv_block(256, 256, 1, 1),     # (256, 56, 56)
            conv_block(256, 512, 3, 1),     # (512, 56, 56)
            nn.MaxPool2d(2, 2),             # (512, 28, 28)

            conv_block(512, 256, 1, 1),     # (256, 28, 28)
            conv_block(256, 512, 3, 1),     # (512, 28, 28)
            conv_block(512, 256, 1, 1),     # (256, 28, 28)
            conv_block(256, 512, 3, 1),     # (512, 28, 28)
            conv_block(512, 256, 1, 1),     # (256, 28, 28)
            conv_block(256, 512, 3, 1),     # (512, 28, 28)
            conv_block(512, 256, 1, 1),     # (256, 28, 28)
            conv_block(256, 512, 3, 1),     # (512, 28, 28)
            conv_block(512, 512, 1, 1),     # (512, 28, 28)
            conv_block(512, 1024, 3, 1),    # (1024, 28, 28)
            nn.MaxPool2d(2, 2),             # (1024, 14, 14)

            conv_block(1024, 512, 1, 1),    # (512, 14, 14)
            conv_block(512, 1024, 3, 1),    # (1024, 14, 14)
            conv_block(1024, 512, 1, 1),    # (512, 14, 14)
            conv_block(512, 1024, 3, 1),    # (1024, 14, 14)
            conv_block(1024, 1024, 3, 1),   # (1024, 14, 14)
            conv_block(1024, 1024, 3, 2),   # (1024, 7, 7)

            conv_block(1024, 1024, 3, 1),   # (1024, 7, 7)
            conv_block(1024, 1024, 3, 1),   # (1024, 7, 7)
        )

        self.head = nn.Sequential(
            nn.Flatten(),                               # (50176,)  = 1024 * 7 * 7
            nn.Linear(1024 * S * S, head_hidden),       # (4096,)
            nn.LeakyReLU(0.1),                          # (4096,)
            nn.Dropout(dropout),                        # (4096,)
            nn.Linear(head_hidden, S * S * (5 + C)),    # (4165,)   = 7 * 7 * 85
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).view(-1, self.S, self.S, 5 + self.C)  # (N, 7, 7, 85)
