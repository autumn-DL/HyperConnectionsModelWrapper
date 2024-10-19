import torch
import torch.nn as nn
from HyperConnectionsWrapper.HyperConnectionsWrapper import HyperConnectionsWrapper
class example(nn.Module):
    def __init__(self,dim=128):
        super().__init__()
        self.l=nn.Linear(dim,dim)
    def forward(self,x):
        return self.l(x)

if __name__ == '__main__':
    # 这只是个简单的例子 实际使用请在求和好加上norm
    hyper_connection_rate=4
    x=torch.randn(1,20,128)
    x=x.unsqueeze(-2)
    if hyper_connection_rate != 1:
        x = torch.cat([x] * hyper_connection_rate, dim=-2)
    m=HyperConnectionsWrapper(model=example(dim=128),dim=128,hyper_connection_rate=hyper_connection_rate,hyper_connection_layer_id=0,hyper_connection_dynamic=True)
    out=m(x)
    out=out.sum(-2)
    print(out.shape)
    x=torch.randn(1,20,20,128)
    x=x.unsqueeze(-2)
    if hyper_connection_rate != 1:
        x = torch.cat([x] * hyper_connection_rate, dim=-2)
    m=HyperConnectionsWrapper(model=example(dim=128),dim=128,hyper_connection_rate=hyper_connection_rate,hyper_connection_layer_id=0,hyper_connection_dynamic=True)
    out=m(x)
    out=out.sum(-2)
    print(out.shape)