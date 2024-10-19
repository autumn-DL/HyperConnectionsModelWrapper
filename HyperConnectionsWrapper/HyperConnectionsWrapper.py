import torch
import torch.nn as nn
from .HyperConnections import HyperConnection


class HyperConnectionsWrapper(nn.Module):
    def __init__(
            self,
            model,
            dim,
            hyper_connection_rate,
            hyper_connection_layer_id,
            hyper_connection_dynamic,
    ):
        super().__init__()
        self.model = model
        self.hyper_connection = HyperConnection(dim=dim,
                                                rate=hyper_connection_rate,
                                                layer_id=hyper_connection_layer_id,
                                                dynamic=hyper_connection_dynamic)


    def forward(self, x, *args, **kwargs):
        latent_h, mix_h_o, beta = self.hyper_connection.run_width_connection(x)
        x_ = self.model(latent_h, *args, **kwargs)
        model_out=self.hyper_connection.run_depth_connection(x_,mix_h_o,beta)
        return model_out

