import torch
from torch import nn

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    def __init__(self,model, args):
        super().__init__()
        self.args = args
        self.backbone = model

        inp_dim = self.backbone.FC_layers[-1].in_features
        self.backbone.FC_layers[-1] = nn.Identity()
        layers= []
        for i in range(args.projector_layers-1):
            layers.append(nn.Linear(inp_dim,inp_dim, bias=False))
            layers.append(nn.BatchNorm1d(inp_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(inp_dim, inp_dim, bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(inp_dim, affine=False)

    def forward(self, y1, y2, batch_items, num_nodes):
        _, emb1 = self.backbone(y1,batch_items, num_nodes)
        _, emb2 = self.backbone(y2,batch_items, num_nodes)
        z1 = self.projector(emb1)
        z2 = self.projector(emb2)

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(z1.shape[0])
        # # sum the cross-correlation matrix between all gpus
        # c.div_(self.args.batch_size)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss, emb1
