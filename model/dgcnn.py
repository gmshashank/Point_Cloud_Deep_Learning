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


def knn(x: torch.Tensor, k: int) -> int:
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 20, idx: int = None):
    batch_size = x.shape(0)
    num_points = x.shape(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k=k)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class DGCNN_Classification(nn.Module):
    def __init__(self, args: argparse.Namespace, output_channels: int = 40):
        super().__init__()
        self.args = args
        self.k = args.k
        self.output_channels = output_channels

        self.block1 = addBlock("block1", "conv", 2 * 3, 64, kernel_size=1, bias=False)
        self.block2 = addBlock("block2", "conv", 2 * 64, 64, kernel_size=1, bias=False)
        self.block3 = addBlock("block3", "conv", 2 * 64, 128, kernel_size=1, bias=False)
        self.block4 = addBlock("block4", "conv", 2 * 128, 256, kernel_size=1, bias=False)
        self.block5 = addBlock("block5", "conv", 512, 1024, kernel_size=1, bias=False)
        self.block6 = addBlock("block6", "linear", 2 * 1024, 512, bias=False, dropout=True)
        self.block7 = addBlock("block7", "linear", 512, 256, dropout=True)
        self.block8 = addBlock("block8", "linear", 256, self.output_channels, bn=False, relu=False, dropout=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.block1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.block2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.block3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.block4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.block5(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # max pooling
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # average pooling
        x = torch.cat((x1, x2), dim=1)

        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        return x
