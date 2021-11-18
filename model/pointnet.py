import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


def addBlock(
    name: str,
    type: str,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 1,
    bias: bool = False,
    bn: bool = True,
    relu: bool = True,
    dropout: bool = False,
):
    block = nn.Sequential()
    if type == "linear":
        block.add_module(f"{name}_linear", nn.Linear(in_channels, out_channels, bias=bias))
    elif type == "conv":
        block.add_module(f"{name}_conv", nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=bias))
    if bn:
        block.add_module(f"{name}_bn", nn.BatchNorm1d(out_channels))
    if relu:
        block.add_module(f"{name}_relu", nn.ReLU(inplace=True))
    if dropout:
        block.add_module(f"{name}_dropout", nn.Dropout())
    return block


class PointNet(nn.Module):
    def __init__(self, args: argparse.Namespace, output_channels: int = 40):
        super().__init__()
        self.args = args
        self.output_channels = output_channels
        # self.conv_block1 = addBlock("conv_block1", "conv", self.k, 64, kernel_size=1, bias=False)
        self.block1 = addBlock("block1", "conv", 3, 64, kernel_size=1, bias=False)
        self.block2 = addBlock("block2", "conv", 64, 64, kernel_size=1, bias=False)
        self.block3 = addBlock("block3", "conv", 64, 64, kernel_size=1, bias=False)
        self.block4 = addBlock("block4", "conv", 64, 128, kernel_size=1, bias=False)
        self.block5 = addBlock("block5", "conv", 128, 1024, kernel_size=1, bias=False)
        self.block6 = addBlock("block6", "linear", 1024, 512, bias=False, dropout=True)
        self.block7 = addBlock("block7", "linear", 512, self.output_channels, bias=False, bn=False, relu=False)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = self.block6(x)
        x = self.block7(x)
        return x
