import torch

from pyvene.models.interventions import (
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
    InterventionOutput,
)


class ConceptVectorIntervention(
    SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention
):
    """
    Phi(h) = h + v

    Adapted from: https://github.com/stanfordnlp/axbench/blob/c01a22adcc6a2e1d3ec663c7577afac85ae03771/axbench/models/interventions.py#L718
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(self.embed_dim, kwargs["low_rank_dimension"], bias=False)
        self.factor = torch.nn.Parameter(torch.empty(()))

        bound_factor = kwargs.get('factor_init_scale', 1.0) * (self.embed_dim ** 0.5)
        bound_vector = kwargs.get('vector_init_scale', 1.0) / (self.embed_dim ** 0.5)
        with torch.no_grad():
            torch.nn.init.uniform_(self.proj.weight.data, a=-bound_vector, b=bound_vector)
            self.factor.data = bound_factor * torch.ones_like(self.factor.data)

    def forward(self, base, source=None, subspaces=None):
        v = []
        for i in range(base.shape[0]):
            v += [self.proj.weight[0]]
        v = torch.stack(v, dim=0).unsqueeze(dim=-1)  # bs, h, 1
        steering_vec = v.permute(0, 2, 1)  # bs, 1, h

        output = base + self.factor * steering_vec

        return output.to(base.dtype)
