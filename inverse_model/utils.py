# Modified in 2026 by Yu Fujikata for XANES conditioning and public release.
# Derived from DiffSpectra and distributed under Apache-2.0; see LICENSE.
import torch
import os
import logging
import numpy as np
from rdkit import Chem


SPECTRUM_METADATA = {
    'uv': {'length': 701, 'patch_index': 0},
    'ir': {'length': 3501, 'patch_index': 1},
    'raman': {'length': 3501, 'patch_index': 2},
    'xanes_CK': {'length': 300, 'patch_index': 3},
    'xanes_NK': {'length': 300, 'patch_index': 4},
    'xanes_OK': {'length': 300, 'patch_index': 5},
}

ABSORBING_EDGE_TO_ATOM_TYPE = {
    'xanes_CK': 1,
    'xanes_NK': 2,
    'xanes_OK': 3,
}

LEGACY_SPECTRA_VERSION_MAP = {
    'none': [],
    'uv': ['uv'],
    'uv_mini': ['uv'],
    'ir': ['ir'],
    'ir_mini': ['ir'],
    'raman': ['raman'],
    'raman_mini': ['raman'],
    'xanes': ['xanes_CK', 'xanes_NK', 'xanes_OK'],
    'allspectra': ['uv', 'ir', 'raman'],
    'allspectra_mini': ['uv', 'ir', 'raman'],
    'allspectra_xanes': ['uv', 'ir', 'raman', 'xanes_CK', 'xanes_NK', 'xanes_OK'],
    'ir_plus_xanes': ['ir', 'xanes_CK', 'xanes_NK', 'xanes_OK'],
    'uv_plus_xanes': ['uv', 'xanes_CK', 'xanes_NK', 'xanes_OK'],
    'raman_plus_xanes': ['raman', 'xanes_CK', 'xanes_NK', 'xanes_OK'],
    'ir_plus_raman_plus_xanes': ['ir', 'raman', 'xanes_CK', 'xanes_NK', 'xanes_OK'],
}


def resolve_used_spectra(spectra_version='none', used_spectra=None):
    if used_spectra is not None and len(used_spectra) > 0:
        resolved = list(used_spectra)
    else:
        if spectra_version not in LEGACY_SPECTRA_VERSION_MAP:
            raise ValueError(f"Unknown spectra_version: {spectra_version}")
        resolved = list(LEGACY_SPECTRA_VERSION_MAP[spectra_version])

    invalid = [name for name in resolved if name not in SPECTRUM_METADATA]
    if invalid:
        raise ValueError(f"Unknown spectra names in used_spectra: {invalid}")
    return resolved


def get_used_spectra(data_config):
    used_spectra = getattr(data_config, 'used_spectra', None)
    spectra_version = getattr(data_config, 'spectra_version', 'none')
    return resolve_used_spectra(spectra_version=spectra_version, used_spectra=used_spectra)


def get_absorbing_edge(data_config):
    absorbing_edge = getattr(data_config, 'absorbing_edge', '')
    if absorbing_edge:
        if absorbing_edge not in ABSORBING_EDGE_TO_ATOM_TYPE:
            raise ValueError(f"Unsupported absorbing_edge: {absorbing_edge}")
        return absorbing_edge

    used_spectra = get_used_spectra(data_config)
    xanes_spectra = [name for name in used_spectra if name in ABSORBING_EDGE_TO_ATOM_TYPE]
    if len(xanes_spectra) == 1:
        return xanes_spectra[0]
    if len(xanes_spectra) == 0:
        return ''
    raise ValueError(
        "When using site-specific conditioning, please set exactly one absorbing edge."
    )


def stack_spectra_context(items, used_spectra):
    if len(used_spectra) == 0:
        return None
    return [torch.stack([getattr(item, spectrum_name) for item in items], dim=0)
            for spectrum_name in used_spectra]


def build_condition_context(items, used_spectra, use_site_spectra=False,
                            site_spectra_field='site_spectra', exs_field='exs'):
    context = {}

    spectra = stack_spectra_context(items, used_spectra)
    if spectra is not None:
        context['spectra'] = spectra

    if use_site_spectra:
        context['site_spectra'] = torch.stack(
            [getattr(item, site_spectra_field) for item in items], dim=0
        )
        context['exs'] = torch.stack(
            [getattr(item, exs_field) for item in items], dim=0
        )

    if not context:
        return None
    if len(context) == 1 and 'spectra' in context:
        return context['spectra']
    return context


def move_context_to_device(context, device):
    if context is None:
        return None
    if isinstance(context, list):
        return [move_context_to_device(value, device) for value in context]
    if isinstance(context, dict):
        return {key: move_context_to_device(value, device) for key, value in context.items()}
    return context.to(device)


def serialize_rdmol(rdmol):
    if rdmol is None or isinstance(rdmol, str):
        return rdmol
    return Chem.MolToMolBlock(rdmol)


def ensure_rdmol(rdmol):
    if rdmol is None or hasattr(rdmol, 'GetNumAtoms'):
        return rdmol
    if isinstance(rdmol, str):
        mol = Chem.MolFromMolBlock(rdmol, removeHs=False, sanitize=True)
        if mol is None:
            raise ValueError("Failed to reconstruct RDKit Mol from MolBlock string.")
        return mol
    raise TypeError(f"Unsupported rdmol type: {type(rdmol)!r}")


def _validate_model_state_dict(current_state_dict, loaded_state_dict):
    missing_keys = [key for key in current_state_dict.keys() if key not in loaded_state_dict]
    unexpected_keys = [key for key in loaded_state_dict.keys() if key not in current_state_dict]
    shape_mismatches = [
        key for key, value in current_state_dict.items()
        if key in loaded_state_dict and value.shape != loaded_state_dict[key].shape
    ]

    if missing_keys or unexpected_keys or shape_mismatches:
        problems = []
        if missing_keys:
            problems.append(f"missing keys: {missing_keys[:5]}")
        if unexpected_keys:
            problems.append(f"unexpected keys: {unexpected_keys[:5]}")
        if shape_mismatches:
            preview = [
                f"{key} current={tuple(current_state_dict[key].shape)} "
                f"ckpt={tuple(loaded_state_dict[key].shape)}"
                for key in shape_mismatches[:5]
            ]
            problems.append(f"shape mismatches: {preview}")
        raise RuntimeError("; ".join(problems))


def _validate_optimizer_state_dict(current_optimizer, loaded_optimizer_state):
    current_groups = current_optimizer.state_dict()['param_groups']
    loaded_groups = loaded_optimizer_state['param_groups']

    if len(current_groups) != len(loaded_groups):
        raise ValueError(
            f"optimizer group count mismatch: current={len(current_groups)} "
            f"ckpt={len(loaded_groups)}"
        )

    for idx, (current_group, loaded_group) in enumerate(zip(current_groups, loaded_groups)):
        current_size = len(current_group['params'])
        loaded_size = len(loaded_group['params'])
        if current_size != loaded_size:
            raise ValueError(
                f"optimizer group {idx} parameter count mismatch: "
                f"current={current_size} ckpt={loaded_size}"
            )


def restore_checkpoint(ckpt_dir, state, device, reset_on_error=False):
    if not os.path.exists(ckpt_dir):
        if not os.path.exists(os.path.dirname(ckpt_dir)):
            os.makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state

    loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)

    try:
        _validate_model_state_dict(state['model'].state_dict(), loaded_state['model'])
        _validate_optimizer_state_dict(state['optimizer'], loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=True)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state
    except (KeyError, RuntimeError, ValueError) as exc:
        if not reset_on_error:
            raise
        logging.warning(
            "Checkpoint at %s is incompatible with the current run (%s). "
            "Starting from scratch with the current config.",
            ckpt_dir,
            exc,
        )
        return state


def initialize_training_from_checkpoint(ckpt_path, state, device,
                                        load_optimizer=False,
                                        load_ema=True,
                                        reset_step=True):
    if not ckpt_path:
        return state

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Initial checkpoint path error: {ckpt_path}")

    loaded_state = torch.load(ckpt_path, map_location=device, weights_only=False)

    loaded_model_state = loaded_state['model'] if 'model' in loaded_state else loaded_state
    _validate_model_state_dict(state['model'].state_dict(), loaded_model_state)
    state['model'].load_state_dict(loaded_model_state, strict=True)

    if load_optimizer and 'optimizer' in loaded_state:
        _validate_optimizer_state_dict(state['optimizer'], loaded_state['optimizer'])
        state['optimizer'].load_state_dict(loaded_state['optimizer'])

    if load_ema and 'ema' in loaded_state:
        state['ema'].load_state_dict(loaded_state['ema'])
    else:
        state['ema'].shadow_params = [
            p.clone().detach() for p in state['model'].parameters() if p.requires_grad
        ]
        if state['ema'].num_updates is not None:
            state['ema'].num_updates = 0

    if reset_step:
        state['step'] = 0
    elif 'step' in loaded_state:
        state['step'] = loaded_state['step']

    return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


def get_data_scaler(config):
    """Data normalizer"""
    # not consider bias here
    if isinstance(config.model.normalize_factors, str):
        normalize_factors = config.model.normalize_factors.split(',')
        normalize_factors = [int(normalize_factor) for normalize_factor in normalize_factors]
    else:
        normalize_factors = config.model.normalize_factors

    if len(normalize_factors) == 3:
        pos_norm, atom_type_norm, fc_charge_norm = normalize_factors
        edge_norm = 1
    else:
        pos_norm, atom_type_norm, fc_charge_norm, edge_norm = normalize_factors

    centered = config.data.centered

    def scale_fn(pos, atom_type, fc_charge, node_mask, edge_type=None, edge_mask=None):
        if centered:
            atom_type = atom_type * 2. - 1.

        if pos is not None:
            pos = pos / pos_norm * node_mask
        atom_type = atom_type / atom_type_norm * node_mask
        fc_charge = fc_charge / fc_charge_norm * node_mask

        if edge_type is not None:
            if centered:
                edge_type = edge_type * 2. - 1.
            edge_type = edge_type / edge_norm
            edge_type = edge_type * edge_mask.reshape(node_mask.size(0), node_mask.size(1), node_mask.size(1), 1)
            return pos, atom_type, fc_charge, edge_type

        return pos, atom_type, fc_charge

    return scale_fn


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    # not consider bias here
    if isinstance(config.model.normalize_factors, str):
        normalize_factors = config.model.normalize_factors.split(',')
        normalize_factors = [int(normalize_factor) for normalize_factor in normalize_factors]
    else:
        normalize_factors = config.model.normalize_factors

    if len(normalize_factors) == 3:
        pos_norm, atom_type_norm, fc_charge_norm = normalize_factors
        edge_norm = 1
    else:
        pos_norm, atom_type_norm, fc_charge_norm, edge_norm = normalize_factors

    centered = config.data.centered

    def inverse_scale_fn(pos, atom_type, fc_charge, node_mask, edge_type=None, edge_mask=None):
        if pos is not None:
            pos = pos * pos_norm * node_mask
        atom_type = atom_type * atom_type_norm
        fc_charge = fc_charge * fc_charge_norm * node_mask
        if centered:
            atom_type = (atom_type + 1.) / 2. * node_mask

        if edge_type is not None:
            edge_type = edge_type * edge_norm
            if centered:
                edge_type = (edge_type + 1.) / 2.
            edge_type = edge_type * edge_mask.reshape(node_mask.size(0), node_mask.size(1), node_mask.size(1), 1)
            return pos, atom_type, fc_charge, edge_type

        return pos, atom_type, fc_charge

    return inverse_scale_fn


def get_self_cond_fn(config):
    # To simplify: directly return

    process_type = config.model.self_cond_type  # 'ori', 'clamp'
    compress_edge = config.data.compress_edge
    atom_types = config.data.atom_types
    include_fc = config.model.include_fc_charge
    atom_type_scale = np.array([0., 1.])
    fc_scale = np.array(config.data.fc_scale)
    edge_type_scale = np.array([0., 1.])
    if isinstance(config.model.normalize_factors, str):
        normalize_factors = config.model.normalize_factors.split(',')
        normalize_factors = [int(normalize_factor) for normalize_factor in normalize_factors]
    else:
        normalize_factors = config.model.normalize_factors
    _, atom_type_norm, fc_norm, edge_norm = normalize_factors

    # get the value scale
    centered = config.data.centered
    if centered:
        atom_type_scale = atom_type_scale * 2. - 1.
        edge_type_scale = edge_type_scale * 2. - 1.
    atom_type_scale = atom_type_scale / atom_type_norm
    fc_scale = fc_scale / fc_norm
    edge_type_scale = edge_type_scale / edge_norm

    def process_self_cond(cond_x, cond_edge_x):
        if process_type == 'ori':
            return cond_x, cond_edge_x
        elif process_type == 'clamp':
            atom_x = cond_x[:, :, 3:3+atom_types]
            atom_x = atom_x.clamp(atom_type_scale[0], atom_type_scale[1])
            cond_x[:, :, 3:3+atom_types] = atom_x
            if include_fc:
                atom_fc = cond_x[:, :, -1:]
                atom_fc = atom_fc.clamp(fc_scale[0], fc_scale[1])
                cond_x[:, :, -1:] = atom_fc
            cond_edge_x = cond_edge_x.clamp(edge_type_scale[0], edge_type_scale[1])
            return cond_x, cond_edge_x
        else:
            raise ValueError("Self-condition data process error.")

    return process_self_cond


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]
