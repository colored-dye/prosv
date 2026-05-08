"""
This module is inspired by https://github.com/stanfordnlp/pyvene/blob/main/pyvene/models/interventions.py
and https://github.com/stanfordnlp/pyreft/blob/main/pyreft/interventions.py.

For representation interventions, the input argument to `forward()` method, `x`,
is always 2D: `[L, H]`, where `L` is number of tokens to intervene on,
and `H` is representation dimension.
"""

from typing import Literal
import torch
import torch.nn as nn


class BaseAdapter(nn.Module):
    """
    LoRA-like subspace projections.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        low_rank_dim,
        alpha=1.0,
        init_regime: Literal["init_A", "init_B", "init_AB"] = "init_A",
        **kwargs,
    ):
        super().__init__()
        self.A = nn.Parameter(torch.empty((low_rank_dim, in_dim)))
        """Shape: (r, d)"""
        self.B = nn.Parameter(torch.empty((out_dim, low_rank_dim)))
        """Shape: (d, r)"""
        self.scaling = alpha / low_rank_dim

        with torch.no_grad():
            if init_regime == "init_A":
                nn.init.kaiming_uniform_(self.A.data)
                nn.init.zeros_(self.B.data)
            elif init_regime == "init_B":
                nn.init.zeros_(self.A.data)
                nn.init.kaiming_uniform_(self.B.data)
            elif init_regime == "init_AB":
                nn.init.kaiming_uniform_(self.A.data)
                nn.init.kaiming_uniform_(self.B.data)
            else:
                raise ValueError(f"Unknown init method: `{init_regime}`")

    def forward(self, x):
        raise NotImplementedError()


class ParameterAdapter(nn.Module):
    def __init__(self, lin: nn.Module):
        super().__init__()
        self.lin = lin
        """Target linear subnetwork."""
        self.lin.requires_grad_(False)

    def forward(self, x):
        raise NotImplementedError()

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for k in tuple(state.keys()):
            if k.startswith("lin."):
                del state[k]
        return state

    def _load_from_state_dict(self, state_dict, *args, **kwargs):
        lin_state = self.lin.state_dict()
        for k, v in lin_state.items():
            state_dict["lin." + k] = v
        return super()._load_from_state_dict(state_dict, *args, **kwargs)


class ParameterDoRA(BaseAdapter, ParameterAdapter):
    """
    https://arxiv.org/abs/2402.09353 / \
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/dora.py

    h = Wh
    V' = W + BA; h' = V'h
    W' = m V' / ||V'||_c
    """

    def __init__(self, lin: nn.Linear, in_dim, out_dim, low_rank_dim, alpha=1.0):
        BaseAdapter.__init__(
            self,
            in_dim=in_dim,
            out_dim=out_dim,
            low_rank_dim=low_rank_dim,
            alpha=alpha,
        )
        ParameterAdapter.__init__(self, lin=lin)

        self.m = nn.Parameter(torch.empty(out_dim, 1))

        with torch.no_grad():
            self.m.data = torch.linalg.norm(self.lin.weight, dim=1)

    def get_weight_norm(self, weight, lora_weight, scaling):
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def forward(self, x):
        lora_weight = self.B @ self.A

        weight_norm = self.get_weight_norm(
            self.lin.weight,
            lora_weight.detach(),
            self.scaling,
        )
        weight_norm = weight_norm.detach()
        mag_norm_scale = (self.m.view(-1) / weight_norm).view(1, -1)

        lora_result = x.to(dtype=self.A.dtype) @ self.A.T @ self.B.T

        result_orig = self.lin(x).to(dtype=self.A.dtype)
        result_dora = mag_norm_scale * result_orig
        result_dora = result_dora + mag_norm_scale * lora_result * self.scaling
        return result_dora.to(dtype=x.dtype)


class ParameterLoRA(BaseAdapter):
    """
    h = Wh
    W' = W + BA
    """

    def __init__(self, lin: nn.Linear, in_dim, out_dim, low_rank_dim, alpha=1.0):
        BaseAdapter.__init__(
            self,
            in_dim=in_dim,
            out_dim=out_dim,
            low_rank_dim=low_rank_dim,
            alpha=alpha,
            # init_regime='init_AB',
        )

        self.lin = lin
        """Target linear subnetwork."""
        self.lin.requires_grad_(False)

    def forward(self, x):
        lora_result = x.to(dtype=self.A.dtype) @ self.A.T @ self.B.T

        result_orig = self.lin(x).to(dtype=self.A.dtype)
        result_lora = result_orig + lora_result * self.scaling
        return result_lora.to(dtype=x.dtype)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for k in tuple(state.keys()):
            if k.startswith("lin."):
                del state[k]
        return state

    def _load_from_state_dict(self, state_dict, *args, **kwargs):
        lin_state = self.lin.state_dict()
        for k, v in lin_state.items():
            state_dict["lin." + k] = v
        return super()._load_from_state_dict(state_dict, *args, **kwargs)


class ParameterLoreft(nn.Module):
    """
    h = Wh,
    W' = W + BA - B B^T W.
    """

    def __init__(self, lin: nn.Linear, in_dim, out_dim, low_rank_dim, alpha=1.0):
        super().__init__()
        self.rotate_layer = nn.utils.parametrizations.orthogonal(
            nn.Linear(out_dim, low_rank_dim, bias=False)
        )
        # self.rotate_layer = nn.Linear(out_dim, low_rank_dim, bias=False)
        self.learned_source = nn.Linear(in_dim, low_rank_dim, bias=False)
        self.scaling = alpha / low_rank_dim

        self.lin = lin
        self.lin.requires_grad_(False)

        with torch.no_grad():
            nn.init.orthogonal_(self.rotate_layer.weight.data)
            nn.init.kaiming_uniform_(self.learned_source.weight.data)

    def forward(self, x):
        result_orig = self.lin(x)
        rotated_base = self.rotate_layer(
            result_orig.to(dtype=self.rotate_layer.weight.dtype)
        )
        rotated_source = self.learned_source(x.to(self.learned_source.weight.dtype))
        diff = self.scaling * torch.matmul(
            (rotated_source - rotated_base).to(self.rotate_layer.weight.dtype),
            self.rotate_layer.weight,
        )
        return result_orig + diff.to(dtype=x.dtype)

    def to(self, device=None, dtype=None, non_blocking=False):
        """Orthogonal layer cannot cast to (b)float16."""
        if dtype is not None:
            self.learned_source.to(dtype=dtype, device=device)
            self.rotate_layer.to(device=device)
            return super().to(device=device, non_blocking=non_blocking)
        else:
            return super().to(device=device, dtype=dtype, non_blocking=non_blocking)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for k in tuple(state.keys()):
            if k.startswith("lin."):
                del state[k]
        return state

    def _load_from_state_dict(self, state_dict, *args, **kwargs):
        lin_state = self.lin.state_dict()
        for k, v in lin_state.items():
            state_dict["lin." + k] = v
        return super()._load_from_state_dict(state_dict, *args, **kwargs)


# class ParameterLoRAClamping(ParameterAdapter):
#     """
#     h = Wh
#     W' = W + BA - B B.T
#     """

#     def __init__(self, lin: nn.Linear, in_dim, out_dim, low_rank_dim, alpha=1.0):
#         BaseAdapter.__init__(
#             self,
#             in_dim=in_dim,
#             out_dim=out_dim,
#             low_rank_dim=low_rank_dim,
#             alpha=alpha,
#             init_regime="init_AB",
#         )
#         ParameterAdapter.__init__(self, lin=lin)

#         self.B = nn.utils.parametrizations.orthogonal(self.B)

#     def forward(self, x):
#         lora_result = x.to(dtype=self.A.dtype) @ self.A.T - x.to(dtype=self.A.dtype) @ self.B
#         lora_result = lora_result @ self.B.T

#         result_orig = self.lin(x).to(dtype=self.A.dtype)
#         result_lora = result_orig + lora_result * self.scaling
#         return result_lora.to(dtype=x.dtype)


class RepresentationDoRA(BaseAdapter):
    """
    W = I; h = Wh
    V' = I + BA; h' = V'h
    W' = m V' / ||V'||_c
    """

    def __init__(self, embed_dim, low_rank_dim, alpha=1.0, **kwargs):
        super().__init__(
            in_dim=embed_dim,
            out_dim=embed_dim,
            low_rank_dim=low_rank_dim,
            alpha=alpha,
        )
        self.m = nn.Parameter(torch.ones(embed_dim, 1))

    def get_weight_norm(self, lora_weight, scaling):
        weight = scaling * lora_weight
        diag = torch.diag(weight)
        weight[range(weight.size(0)), range(weight.size(0))] = diag + 1.0
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def forward(self, x):
        lora_weight = self.B @ self.A

        weight_norm = self.get_weight_norm(
            lora_weight.detach(),
            self.scaling,
        )
        weight_norm = weight_norm.detach()
        mag_norm_scale = (self.m.view(-1) / weight_norm).view(1, -1)

        lora_result = x.to(dtype=self.A.dtype) @ self.A.T @ self.B.T

        result_dora = mag_norm_scale * x.to(dtype=self.A.dtype)
        result_dora = result_dora + mag_norm_scale * lora_result * self.scaling
        return result_dora.to(dtype=x.dtype)


class RepresentationLoRA(BaseAdapter):
    """
    W = I; h = Wh
    W' = I + BA
    h' = W'h
    """

    def __init__(self, embed_dim, low_rank_dim, alpha=1.0, **kwargs):
        super().__init__(
            in_dim=embed_dim,
            out_dim=embed_dim,
            low_rank_dim=low_rank_dim,
            alpha=alpha,
            init_regime="init_AB",
        )

    def forward(self, x):
        lora_result = x.to(dtype=self.A.dtype) @ self.A.T @ self.B.T
        result_lora = lora_result * self.scaling
        return x + result_lora.to(dtype=x.dtype)


class DecoupledLoRA(ParameterAdapter):
    """
    h = x + f(x)
    h' = h + W_s f(x) + W_p x
    """

    def __init__(self, lin: nn.Module, embed_dim, low_rank_dim, alpha=1.0):
        ParameterAdapter.__init__(self, lin=lin)
        self.parallel_adapter = BaseAdapter(
            in_dim=embed_dim, out_dim=embed_dim, low_rank_dim=low_rank_dim, alpha=alpha
        )
        self.serial_adapter = BaseAdapter(
            in_dim=embed_dim, out_dim=embed_dim, low_rank_dim=low_rank_dim, alpha=alpha
        )

        # same initialization
        with torch.no_grad():
            self.serial_adapter.A.data = self.parallel_adapter.A.data.clone()
            self.serial_adapter.B.data = self.parallel_adapter.B.data.clone()

    def forward(self, x):
        orig_result = self.lin(x).to(dtype=self.serial_adapter.A.dtype)
        serial_result = orig_result @ self.serial_adapter.A.T @ self.serial_adapter.B.T
        parallel_result = (
            x.to(dtype=self.parallel_adapter.A.dtype)
            @ self.parallel_adapter.A.T
            @ self.parallel_adapter.B.T
        )
        result = (
            orig_result
            + self.serial_adapter.scaling * serial_result
            + self.parallel_adapter.scaling * parallel_result
        )
        return result.to(dtype=x.dtype)


class LoreftAdapter(nn.Module):
    """Low-rank ReFT (LoReFT)."""

    def __init__(self, embed_dim, low_rank_dim=1, **kwargs):
        super().__init__()
        self.rotate_layer = nn.utils.parametrizations.orthogonal(
            nn.Linear(embed_dim, low_rank_dim, bias=False)
        )
        self.learned_source = nn.Linear(embed_dim, low_rank_dim, bias=True)
        self.scaling = 1.0

        with torch.no_grad():
            # nn.init.kaiming_uniform_(self.learned_source.weight.data)
            # nn.init.normal_(self.learned_source.bias.data, mean=0.0, std=embed_dim**0.5)
            nn.init.zeros_(self.learned_source.bias.data)

    def to(self, device=None, dtype=None, non_blocking=False):
        """Orthogonal layer cannot cast to (b)float16."""
        if dtype is not None:
            self.learned_source.to(dtype=dtype)
            return super().to(device=device, non_blocking=non_blocking)
        else:
            return super().to(device=device, dtype=dtype, non_blocking=non_blocking)

    def forward(self, x):
        rotated_base = self.rotate_layer(x.to(dtype=self.rotate_layer.weight.dtype))
        rotated_source = self.learned_source(
            x.to(dtype=self.learned_source.weight.dtype)
        )
        diff = rotated_source - rotated_base
        output = x + torch.matmul(diff, self.rotate_layer.weight).to(dtype=x.dtype)
        return output

    @property
    def A(self):
        """Shape: (r, d)"""
        return self.rotate_layer.weight

    @property
    def B(self):
        """Shape: (d, r)"""
        return self.learned_source.weight.T


class DireftUnitAdapter(nn.Module):
    """Direct ReFT with orthonormal write-outs (DiReFTUnit)."""

    def __init__(self, embed_dim, low_rank_dim=1, **kwargs):
        super().__init__()
        self.rotate_layer = nn.utils.parametrizations.orthogonal(
            nn.Linear(embed_dim, low_rank_dim, bias=False)
        )
        self.learned_source = nn.Linear(embed_dim, low_rank_dim, bias=True)
        self.scaling = 1.0

        with torch.no_grad():
            # nn.init.kaiming_uniform_(self.learned_source.weight.data)
            # nn.init.normal_(self.learned_source.bias.data, mean=0.0, std=embed_dim**0.5)
            nn.init.zeros_(self.learned_source.bias.data)

    def to(self, device=None, dtype=None, non_blocking=False):
        """Orthogonal layer cannot cast to (b)float16."""
        if dtype is not None:
            self.learned_source.to(dtype=dtype)
            return super().to(device=device, non_blocking=non_blocking)
        else:
            return super().to(device=device, dtype=dtype, non_blocking=non_blocking)

    def forward(self, x):
        rotated_source = self.learned_source(
            x.to(dtype=self.learned_source.weight.dtype)
        )
        output = x + torch.matmul(
            rotated_source.to(dtype=self.rotate_layer.weight.dtype),
            self.rotate_layer.weight,
        ).to(dtype=x.dtype)
        return output

    @property
    def A(self):
        """Shape: (r, d)"""
        return self.rotate_layer.weight

    @property
    def B(self):
        """Shape: (d, r)"""
        return self.learned_source.weight.T


class DireftAdapter(nn.Module):
    """Direct ReFT (DiReFT)."""

    def __init__(self, embed_dim, low_rank_dim=1, **kwargs):
        super().__init__()
        self.rotate_layer = nn.Linear(embed_dim, low_rank_dim, bias=False)
        self.learned_source = nn.Linear(embed_dim, low_rank_dim, bias=True)
        self.scaling = 1.0

        with torch.no_grad():
            # nn.init.kaiming_uniform_(self.learned_source.weight.data)
            # nn.init.normal_(self.learned_source.bias.data, mean=0.0, std=embed_dim**0.5)
            nn.init.zeros_(self.learned_source.bias.data)

    def to(self, device=None, dtype=None, non_blocking=False):
        """Orthogonal layer cannot cast to (b)float16."""
        if dtype is not None:
            self.learned_source.to(dtype=dtype)
            return super().to(device=device, non_blocking=non_blocking)
        else:
            return super().to(device=device, dtype=dtype, non_blocking=non_blocking)

    def forward(self, x):
        rotated_source = self.learned_source(
            x.to(dtype=self.learned_source.weight.dtype)
        )
        output = x + torch.matmul(
            rotated_source.to(dtype=self.rotate_layer.weight.dtype),
            self.rotate_layer.weight,
        ).to(dtype=x.dtype)
        return output

    @property
    def A(self):
        """Shape: (r, d)"""
        return self.rotate_layer.weight

    @property
    def B(self):
        """Shape: (d, r)"""
        return self.learned_source.weight.T


class BiLinearAdapter(nn.Module):
    """Bi-linear."""

    def __init__(self, embed_dim, low_rank_dim=1, **kwargs):
        super().__init__()
        self.w = nn.Linear(embed_dim, low_rank_dim, bias=False)
        self.v = nn.Linear(embed_dim, low_rank_dim, bias=False)
        self.p = nn.Linear(embed_dim, low_rank_dim, bias=False)

    def forward(self, x):
        proj_w = self.w(x.to(dtype=self.w.weight.dtype))
        proj_v = self.v(x.to(dtype=self.v.weight.dtype))
        proj = proj_w * proj_v
        reproj = torch.matmul(proj.to(dtype=self.p.weight.dtype), self.p.weight)
        output = x + reproj.to(dtype=x.dtype)
        return output


class AdditionFreeIntervention(nn.Module):
    """AddFree steering vector."""

    def __init__(
        self, embed_dim, factor_init_scale=1.0, vector_init_scale=1.0, **kwargs
    ):
        """
        a = factor_init_scale / sqrt(fan_in),
        v ~ \mathcal{N}(0, vector_variance/sqrt(fan_in)).
        """
        super().__init__()

        self.proj = nn.Linear(embed_dim, 1, bias=False)
        self.factor = nn.Parameter(torch.empty(()))

        bound_factor = factor_init_scale * (embed_dim ** 0.5)
        bound_vector = vector_init_scale / (embed_dim ** 0.5)
        with torch.no_grad():
            nn.init.uniform_(self.proj.weight.data, a=-bound_vector, b=bound_vector)
            self.factor.data = bound_factor * torch.ones_like(self.factor.data)

    def forward(self, x):
        v = self.factor * self.proj.weight
        return x + v.to(dtype=x.dtype)


class ClampFreeIntervention(nn.Module):
    """
    ClampFree steering vector, which is essentially AddFree with clamping,
    or ClampUnit with free vector norm.
    """

    def __init__(self, embed_dim, factor_init_scale=1.0, vector_init_scale=1.0, **kwargs):
        """
        a = factor_init_scale / sqrt(fan_in),
        v ~ \mathcal{N}(0, vector_variance/sqrt(fan_in)).
        """
        super().__init__()

        self.proj = nn.Linear(embed_dim, 1, bias=False)
        self.factor = nn.Parameter(torch.empty(()))

        bound_vector = vector_init_scale / (embed_dim**0.5)
        bound_factor = factor_init_scale * (embed_dim**0.5)
        with torch.no_grad():
            nn.init.uniform_(self.proj.weight.data, a=-bound_vector, b=bound_vector)
            self.factor.data = bound_factor * torch.ones_like(self.factor.data)

    def forward(self, x):
        v = self.proj.weight
        v_norm = v.data.norm().detach() # Saves memory, similar to DoRA
        u = v / v_norm
        base = x.to(dtype=v.dtype) @ u.T @ u
        source = self.factor * v
        return x + (source - base).to(dtype=x.dtype)


class PreferenceIntervention(nn.Module):
    def __init__(self, embed_dim, factor_init_scale=1.0, vector_init_scale=1.0, **kwargs):
        """
        a = factor_init_scale / sqrt(fan_in),
        v ~ \mathcal{N}(0, vector_variance/sqrt(fan_in)).
        """
        super().__init__()

        self.mode: Literal["add", "clamp"] = "add"

        self.proj = nn.Linear(embed_dim, 1, bias=False)
        self.factor = nn.Parameter(torch.empty(()))

        bound_vector = vector_init_scale / (embed_dim**0.5)
        bound_factor = factor_init_scale * (embed_dim**0.5)
        with torch.no_grad():
            nn.init.uniform_(self.proj.weight.data, a=-bound_vector, b=bound_vector)
            self.factor.data = bound_factor * torch.ones_like(self.factor.data)

    def forward(self, x):
        if self.mode == "add":
            v = self.factor * self.proj.weight
            return x + v.to(dtype=x.dtype)
        elif self.mode == "clamp":
            v = self.proj.weight
            v_norm = v.data.norm()#.detach() # Saves memory, similar to DoRA
            u = v / v_norm
            latent = x.to(dtype=v.dtype) @ u.T
            latent = nn.functional.relu(latent)
            base = latent @ u
            return x - base.to(dtype=x.dtype)
        else:
            raise ValueError(f"Invalid mode: `{self.mode}`")


class AdditionUnitIntervention(nn.Module):
    """AddUnit steering vector."""

    def __init__(self, embed_dim, factor_init_scale=1.0, **kwargs):
        """
        a = factor_init_scale / sqrt(fan_in),
        v ~ \mathcal{N}(0, vector_variance/sqrt(fan_in)).
        """
        super().__init__()

        self.proj = nn.Linear(embed_dim, 1, bias=False)
        self.factor = nn.Parameter(torch.empty(()))

        bound_factor = factor_init_scale * (embed_dim**0.5)
        with torch.no_grad():
            nn.init.orthogonal_(self.proj.weight.data)
            self.factor.data = bound_factor * torch.ones_like(self.factor.data)

    def forward(self, x):
        v = self.factor * self.proj.weight
        return x + v.to(dtype=x.dtype)


class ClampUnitIntervention(nn.Module):
    """ClampUnit steering vector."""

    def __init__(self, embed_dim, factor_init_scale=1.0, **kwargs):
        """
        a = factor_init_scale / sqrt(fan_in),
        v ~ \mathcal{N}(0, vector_variance/sqrt(fan_in)).
        """
        super().__init__()

        self.proj = nn.Linear(embed_dim, 1, bias=False)
        self.factor = nn.Parameter(torch.empty(()))

        bound_factor = factor_init_scale * (embed_dim**0.5)
        with torch.no_grad():
            nn.init.orthogonal_(self.proj.weight.data)
            self.factor.data = bound_factor * torch.ones_like(self.factor.data)

    def forward(self, x):
        base = self.proj(x.to(dtype=self.proj.weight.dtype))
        diff = (self.factor - base) * self.proj.weight
        return x + diff.to(dtype=x.dtype)

