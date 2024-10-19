
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

'''https://arxiv.org/pdf/2409.19606'''

class HyperConnection(nn.Module):
    def __init__(self, dim, rate, layer_id, dynamic, device=None):
        super(HyperConnection, self).__init__()
        self.rate = rate
        self.layer_id = layer_id
        self.dynamic = dynamic
        self.static_beta = nn.Parameter(torch.ones((rate,), device=device))
        init_alpha0 = torch.zeros((rate, 1), device=device)
        init_alpha0[layer_id % rate, 0] = 1.
        self.static_alpha = nn.Parameter(torch.cat([init_alpha0, torch.eye((rate), device=
        device)], dim=1))
        if self.dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros((dim, rate+1), device=device))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros((dim, ), device=device))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
            self.layer_norm = nn.LayerNorm(dim)
    def width_connection(self, h):
        '''

        :param h: B L N C
        :return:
        '''
        # get alpha and beta
        if self.dynamic:
            norm_h = self.layer_norm(h)
        if self.dynamic:
            wc_weight = norm_h @ self.dynamic_alpha_fn
            wc_weight = F.tanh(wc_weight)
            dynamic_alpha = wc_weight * self.dynamic_alpha_scale
            alpha = dynamic_alpha + self.static_alpha[None, None, ...]
        else:
            alpha = self.static_alpha[None, None, ...]
        if self.dynamic:
            dc_weight = norm_h @ self.dynamic_beta_fn
            dc_weight = F.tanh(dc_weight)
            dynamic_beta = dc_weight * self.dynamic_beta_scale
            beta = dynamic_beta + self.static_beta[None, None, ...]
        else:
            beta = self.static_beta[None, None, ...]
            # width connection
        mix_h = alpha.transpose(-1, -2) @ h
        return mix_h, beta
    def depth_connection(self, latent_h, mix_h_o, beta):
        h = torch.einsum("blh,bln->blnh", latent_h, beta) + mix_h_o#[..., 1:, :]
        return h

    def run_width_connection(self,x):
        ndim=x.ndim
        if ndim==4:
            mix_h,beta=self.width_connection(x)
            latent_h=mix_h[..., 0,:]
            h_o=mix_h[..., 1:, :]
            return latent_h,h_o,beta
        elif ndim==5:
            B,H,W,N,C=x.shape
            x=rearrange(x,'b h w n c -> b (h w) n c')
            mix_h, beta = self.width_connection(x)
            latent_h = mix_h[..., 0,:]
            mix_h_o = mix_h[..., 1:, :]
            latent_h=rearrange(latent_h,'b (h w)  c -> b h w  c',h=W)

            return latent_h,mix_h_o,beta
        else:
            raise NotImplementedError
    def run_depth_connection(self,latent_h,mix_h_o,beta):
        ndim=latent_h.ndim
        if ndim==3:
            h=self.depth_connection(latent_h,mix_h_o,beta)
            return h
        elif ndim==4:
            B,H,W,C=latent_h.shape
            latent_h=rearrange(latent_h,'b h w c -> b (h w) c')
            h = self.depth_connection(latent_h, mix_h_o, beta)
            h = rearrange(h, 'b (h w) n c -> b h w n c', h=W)
            return h
    def forward(self,x):
        pass
        # m_h,b=self.width_connection(x)
        # h_x=m_h[..., 0]
        # # h_o=m_h[..., 1:, :]
        # return self.depth_connection(m_h,h_x,b)

if __name__ == '__main__':
    thc=HyperConnection(4,2,0,False)
    x=torch.randn(1,16,4,2,4)
    out=thc.run_width_connection(x)
    print(out[0].shape,out[1].shape,out[2].shape)
    out=thc.run_depth_connection(out[0],out[1],out[2])
    print(out.shape)

    x=torch.randn(1,16,2,4)
    out=thc.run_width_connection(x)
    print(out[0].shape,out[1].shape,out[2].shape)
    out=thc.run_depth_connection(out[0],out[1],out[2])
    print(out.shape)