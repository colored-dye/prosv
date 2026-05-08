"""
This module is inspired by https://github.com/stanfordnlp/pyvene/blob/main/pyvene/models/intervenable_base.py.
"""

from collections import OrderedDict
import copy
from dataclasses import dataclass, field
from enum import Enum
import gc
import json
from loguru import logger
from pathlib import Path
from typing import Callable, Dict, List, Union

import torch
from torch import Tensor
from torch import nn

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .constants import LOCATION_PAD
import reft.interventions as reft_interventions
from .nethook import TraceDict


class InterventionMode(Enum):
    """
    Modes of interventions, depending on intervention locations and stage.
    """

    ALL_TRAINING = 0
    """Full-sequence intervention, at training time. Usually **not** used."""
    ALL_GENERATION = 1
    """Full-sequence intervention, at inference time."""
    PROMPT_ONLY = 2
    """Prompt-only intervention, either training or inference time."""


@dataclass
class RepresentationConfig:
    """
    Defines an instance of intervention.
    """

    layer: int = field(default=None)
    """Necessary."""
    embed_dim: int = field(default=None)
    """Necessary."""
    low_rank_dim: int = field(default=None)
    """Necessary."""
    target_module: str = field(default=None)
    """Necessary."""
    intervention: nn.Module = field(default=None)
    """Necessary."""
    intervention_type: str = field(default=None)
    """Necessary."""
    factor_init_scale: float = field(default=0.0)
    vector_init_scale: float = field(default=0.0)
    alpha: float = field(default=0.0)


class IntervenableConfig:
    """Collects `RepresentationConfig` and is used to initialize `Intervenable`."""

    representations: List[RepresentationConfig]

    def __init__(
        self,
        representations: Union[List[RepresentationConfig], List[Dict]],
        **kwargs,
    ):
        casted_representations = []
        for reprs in representations:
            if isinstance(reprs, RepresentationConfig):
                pass
            elif isinstance(reprs, dict):
                dummy_repr = RepresentationConfig()
                reprs = RepresentationConfig(
                    **{k: v for k, v in reprs.items() if k in dummy_repr.__dict__}
                )
            else:
                raise ValueError(f"`{reprs}` format is not supported.")
            reprs = self._fix_representation(reprs)
            casted_representations.append(reprs)
        self.representations = casted_representations

    def _fix_representation(self, rep: RepresentationConfig):
        """
        Fix missing fields.

        Initialize intervention instances if there is None.
        """
        if rep.intervention_type is None and rep.intervention is not None:
            # Get class name if None
            rep.intervention_type = rep.intervention.__class__.__name__
        elif rep.intervention_type is not None and rep.intervention is None:
            # Initialize class if None
            args = copy.copy(rep.__dict__)
            del args["intervention"]

            intervention_class = getattr(reft_interventions, rep.intervention_type)
            rep.intervention = intervention_class(**args)
        elif rep.intervention_type is None and rep.intervention is None:
            raise ValueError(
                "`intervention_type` and `intervention` should not be None at the same time!"
            )
        return rep

    def save(self, save_dir):
        to_save = []
        for reprs in self.representations:
            reprs_dict = copy.copy(reprs.__dict__)
            del reprs_dict["intervention"]
            to_save.append(reprs_dict)
        save_dir = Path(save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "config.json"
        with open(save_path, "w") as fp:
            json.dump(to_save, fp, indent=2)
        logger.warning(f"Saved config to {save_path}")

    @staticmethod
    def load(load_dir):
        load_dir = Path(load_dir).resolve()
        load_path = load_dir / "config.json"
        logger.warning(f"Loading config from {load_path}")
        config = json.load(open(load_path))
        if isinstance(config, dict):
            config = [config]
        return IntervenableConfig(config)


class IntervenableModel(nn.Module):
    """
    Model with parameterized representation interventions.

    We use `TraceDict` of `nethook.py`, which allows for elegant
    integration of vanilla model functions without implementing new classes
    for intervened model components.
    """

    def __init__(self, model: PreTrainedModel, config: IntervenableConfig):
        super().__init__()

        self.model = model
        self.device = self.model.device
        self.dtype = self.model.dtype

        self.config = config
        self.target_modules = [cfg.target_module for cfg in self.config.representations]
        """Modules to intervene on, e.g., `["model.layers.10"]`."""
        self.interventions = {}
        """Mapping from module name to interventions."""

        for rep in self.config.representations:
            intv = rep.intervention
            if isinstance(intv, nn.Module):
                module_device = tuple(
                    model.get_submodule(rep.target_module).parameters()
                )[0].device
                intv.to(device=module_device, dtype=self.dtype)
            self.interventions[rep.target_module] = intv

        self.disable_model_gradients()

    @staticmethod
    def load(load_directory, model: PreTrainedModel):
        """Load both config and interventions."""
        config = IntervenableConfig.load(load_directory)

        load_path = Path(load_directory, "state_dict.pt")
        logger.warning(f"Loading interventions from {load_path}")

        state_dict = torch.load(
            load_path,
            map_location="cpu",
            weights_only=True,
        )

        # Load intervention weights
        for repr in config.representations:
            repr.intervention.load_state_dict(state_dict[repr.target_module])
        intervenable = IntervenableModel(model=model, config=config)

        intervenable.clear_cache()
        return intervenable

    def save(self, save_directory):
        """Save both config and interventions."""
        self.config.save(save_directory)

        state_dict = OrderedDict()
        for module_name in self.target_modules:
            _state_dict = OrderedDict(
                [
                    (k, v.data.cpu())
                    for k, v in self.interventions[module_name].state_dict().items()
                ]
            )
            state_dict[module_name] = _state_dict
        save_path = Path(save_directory, "state_dict.pt").resolve()
        torch.save(state_dict, save_path)
        logger.warning(f"Saved interventions to {save_path}")

    def forward(
        self,
        locations: Union[List[List[int]], Tensor],
        intervention_mode: InterventionMode = InterventionMode.PROMPT_ONLY,
        *args,
        **kwargs,
    ) -> CausalLMOutput:
        """
        Mirrors `forward()` method of vanilla models.
        One should **always** check `locations` and `intervention_mode` arguments.

        Args:
            locations: 2D intervention locations.
        """
        if isinstance(locations, Tensor):
            locations = [loc[loc != LOCATION_PAD].tolist() for loc in locations]

        edit_fn = _spawn_edit_fn(
            interventions=self.interventions,
            locations=locations,
            intervention_mode=intervention_mode,
        )

        with TraceDict(
            module=self.model,
            layers=self.target_modules,
            retain_output=False,
            retain_input=False,
            retain_grad=False,
            edit_output=edit_fn,
        ):
            outputs = self.model(*args, **kwargs)
        return outputs

    @torch.no_grad()
    def generate(
        self,
        locations: Union[List[List[int]], Tensor],
        intervention_mode: InterventionMode = InterventionMode.PROMPT_ONLY,
        *args,
        **kwargs,
    ):
        """
        Mirrors `generate()` method of vanilla models.
        One should **always** check `locations` and `intervention_mode` arguments.

        Args:
            locations: 2D intervention locations.
            intervention_mode: Special consideration for generation
                with full-sequence interventions.
        """
        if isinstance(locations, Tensor):
            locations = [loc[loc != LOCATION_PAD].tolist() for loc in locations]

        edit_fn = _spawn_edit_fn(
            interventions=self.interventions,
            locations=locations,
            intervention_mode=intervention_mode,
        )

        if kwargs.get("use_cache", None) is not None and not kwargs['use_cache']:
            logger.error("KV cache must be used for generation! Please proceed with discretion.")

        with TraceDict(
            module=self.model,
            layers=self.target_modules,
            retain_output=False,
            retain_input=False,
            retain_grad=False,
            edit_output=edit_fn,
        ):
            outputs = self.model.generate(*args, **kwargs)
        return outputs

    def get_trainable_params(self, separate=False):
        """
        :param separate: Return factor/vector parameters separately if True;
            return them together otherwise.
        """
        if not separate:
            trainable_params = []
            for _, intv in self.interventions.items():
                trainable_params.extend([p for p in intv.parameters() if p.requires_grad])
            return trainable_params
        else:
            factor_params = []
            vector_params = []
            for _, intv in self.interventions.items():
                if hasattr(intv, "factor") and isinstance(intv.factor, nn.Parameter):
                    factor_params.append(intv.factor)
                if (
                    hasattr(intv, "proj")
                    and hasattr(intv.proj, "weight")
                    and isinstance(intv.proj.weight, nn.Parameter)
                ):
                    vector_params.append(intv.proj.weight)
            return factor_params, vector_params

    def get_num_trainable_params(self):
        """
        :return num_trainable_params, ratio_trainable_params:
        """
        trainable_params = self.get_trainable_params()
        num_trainable_params = sum(
            p.numel() for p in trainable_params if p.requires_grad
        )
        num_model_params = sum(p.numel() for p in self.model.parameters())
        ratio_trainable_params = num_trainable_params / num_model_params
        return num_trainable_params, ratio_trainable_params

    def disable_model_gradients(self):
        """
        Disable gradient in the model
        """
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @classmethod
    def clear_cache(self):
        torch.cuda.empty_cache()
        gc.collect()


def _gather(x, locations, dim=1):
    """
    Parallel gather along `dim` where `locations` is a sequence (len == batch)
    of index lists (variable lengths). Returns (gathered, mask).

    gathered has size max_len at `dim` (padded where necessary).
    mask is a boolean tensor (batch, max_len) indicating valid entries.

    (thanks to Haiku 4.5)
    """
    device = x.device
    batch = x.size(0)
    max_len = max((len(loc) for loc in locations), default=0)
    if max_len == 0:
        shape = list(x.shape)
        shape[dim] = 0
        return x.new_empty(shape), torch.zeros(
            (batch, 0), dtype=torch.bool, device=device
        )

    idx = torch.full((batch, max_len), x.size(1) - 1, dtype=torch.long, device=device)
    mask = torch.zeros((batch, max_len), dtype=torch.bool, device=device)
    for i, loc in enumerate(locations):
        if len(loc) == 0:
            continue
        idx[i, : len(loc)] = torch.tensor(loc, dtype=torch.long, device=device)
        mask[i, : len(loc)] = True

    # Build index tensor with same rank as x for torch.gather
    trailing = [1] * (x.ndim - dim - 1)
    index_shape = list(idx.shape) + trailing
    index_expanded = idx.view(*index_shape).expand(
        *list(idx.shape), *list(x.shape[dim + 1 :])
    )

    gathered = torch.gather(x, dim=dim, index=index_expanded)
    return gathered, mask


def _scatter(values, locations, out, dim=1, mask=None):
    """
    Overwrite entries in `out` using `values` placed at `locations`.

    If `mask` is provided, `values` should contain only the masked (True) positions
    from the gathered tensor (i.e., values = gathered[mask]).

    Does not accept a mask or pad values — it simply writes new values
    at the indices specified by `locations`. Unwritten positions in a newly created
    `out` are left uninitialized (use an explicit `out` if you need a known fill).

    (thanks to Haiku 4.5)
    """
    device = values.device
    seq_len = out.size(1)
    batch = out.size(0)
    max_len = max((len(loc) for loc in locations), default=0)

    # build idx from locations
    idx = torch.full((batch, max_len), 0, dtype=torch.long, device=device)
    lengths = []
    for i, loc in enumerate(locations):
        lengths.append(len(loc))
        if len(loc) == 0:
            continue
        loc_t = torch.tensor(loc, dtype=torch.long, device=device)
        if loc_t.max().item() >= seq_len:
            raise IndexError(
                f"location index >= seq_len ({loc_t.max().item()} >= {seq_len})"
            )
        idx[i, : len(loc)] = loc_t
    lengths = torch.tensor(lengths, dtype=torch.long, device=device)

    if out.device != device:
        out = out.to(device)
    if out.dtype != values.dtype:
        out = out.to(values.dtype)

    # flatten trailing dims
    trailing_size = (
        int(torch.prod(torch.tensor(values.shape[1:], device=device)))
        if values.dim() > 1
        else 1
    )
    values_flat = values.reshape(-1, trailing_size)  # (num_valid, trailing_size)
    out_flat = out.reshape(batch, out.shape[1], trailing_size)

    # build boolean of valid positions from lengths
    pos = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
    valid = pos < lengths.unsqueeze(1)  # (batch, max_len)

    # if external mask is provided, combine with valid positions
    if mask is not None:
        valid = valid & mask

    if valid.any():
        b_idx = torch.arange(batch, device=device).unsqueeze(1).expand(-1, max_len)
        b_sel = b_idx[valid]  # (num_valid,)
        idx_sel = idx[valid]  # (num_valid,)
        out_flat[b_sel, idx_sel] = values_flat

    out = out_flat.reshape(out.shape)
    return out


def _spawn_edit_fn(
    interventions: Dict[str, Callable],
    locations: List[List[int]] = None,
    intervention_mode: InterventionMode = InterventionMode.PROMPT_ONLY,
    prefix_length=0,
):
    """
    Return a function as `edit_output` argument of `TraceDict`/`Trace`.

    Args:
        locations: Locations to intervene on, like
            `[[locations_of_first_seq], [locations_of_second_seq], ...]`.
            Example: For `[[0,1], [4]]`, we intervene on 0th/1st tokens
            of the batch index 0, as well as 4th token of batch index 1.
        intervention_mode: Special treatment for batched inference
            of full-sequence interventions.
    """

    def _edit_fn_all_training(output, layer, inputs):
        """
        Intervene on `prompt + response`, but skip prompt prefix.
        This function is not usually used.

        Requires:
            * prefix_length

        Args:
            layer: module name, e.g., `model.layers.10`.
        """
        if layer not in interventions:
            return output

        rest = None
        if isinstance(output, tuple):
            output, *rest = output

        if output.size(1) > 1: # skip prompt prefix
            y = output.clone()
            suffix = interventions[layer](output[:, prefix_length:])
            y[:, prefix_length:] = suffix
        else:
            y = interventions[layer](output)

        if rest is not None:
            return y, *rest
        else:
            return y

    def _edit_fn_all_generation(output, layer, inputs):
        """
        Intervene on `prompt + response`.
        `locations` is needed since left padding is used during batched inference.
        """
        if layer not in interventions:
            return output

        rest = None
        if isinstance(output, tuple):
            output, *rest = output

        if output.size(1) <= 1:
            # Response part
            y = interventions[layer](output)
        else:
            # Prompt part
            # Crucial, avoid gradient issues with in-place operations
            y = output.clone()

            gathered, mask = _gather(x=output, locations=locations)
            # Only intervene on selected tokens
            intervened = interventions[layer](gathered[mask])

            y = _scatter(out=y, locations=locations, values=intervened, mask=mask)

        if rest is not None:
            return y, *rest
        else:
            return y

    def _edit_fn_prompt_only(output, layer, inputs):
        """
        Intervene only on prompt.

        Requires:
            * locations

        Args:
            layer: module name, e.g., `model.layers.10`.
        """
        if layer not in interventions:
            return output

        rest = None
        if isinstance(output, tuple):
            output, *rest = output

        # Response part
        if output.size(1) <= 1:
            # logger.debug("No intervention")
            if rest is not None:
                return output, *rest
            else:
                return output

        # Prompt part
        # logger.debug("Intervention!")

        y = output.clone()  # Crucial, avoid gradient issues with in-place operations

        gathered, mask = _gather(x=output, locations=locations)
        intervened = interventions[layer](
            gathered[mask]
        )  # Only intervene on selected tokens

        y = _scatter(out=y, locations=locations, values=intervened, mask=mask)

        if rest is not None:
            return y, *rest
        else:
            return y

    if intervention_mode == InterventionMode.PROMPT_ONLY: # default
        _edit_fn = _edit_fn_prompt_only
    elif intervention_mode == InterventionMode.ALL_GENERATION: # sometimes used
        _edit_fn = _edit_fn_all_generation
    elif intervention_mode == InterventionMode.ALL_TRAINING: # almost never used
        _edit_fn = _edit_fn_all_training
    else:
        raise ValueError(f"Unknown intervention mode: `{intervention_mode}`")

    return _edit_fn
