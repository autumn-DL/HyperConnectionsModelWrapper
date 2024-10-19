# HyperConnectionsWrapper

[HyperConnections](https://arxiv.org/pdf/2409.19606)

# usage

```python
import torch 
from HyperConnectionsWrapper.HyperConnectionsWrapper import HyperConnectionsWrapper
import torch.nn as nn
class example(nn.Module):
    def __init__(self,dim=128):
        super().__init__()
        self.l=nn.Linear(dim,dim)
    def forward(self,x):
        return self.l(x)
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
```
## note
please add post norm