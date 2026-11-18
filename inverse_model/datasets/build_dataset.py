# Modified in 2026 by Yu Fujikata for XANES conditioning and public release.
# Derived from DiffSpectra and distributed under Apache-2.0; see ../LICENSE.
import logging
import os

from torch_geometric.transforms import Compose, ToDevice
from rdkit import Chem
from .qm9s_dataset import QM9SDataset
from .datasets_config import get_dataset_info
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
import torch
from scipy.spatial.transform import Rotation
from utils import get_used_spectra, build_condition_context, get_absorbing_edge, ABSORBING_EDGE_TO_ATOM_TYPE, ensure_rdmol


prop2idx = {'mu': 0, 'alpha': 1, 'homo': 2, 'lumo': 3, 'gap': 4, 'Cv': 11}


class SiteConditionedDataset(Dataset):
    def __init__(self, base_dataset, absorbing_edge, exs_field='exs', site_spectra_field='site_spectra'):
        self.base_dataset = base_dataset
        self.absorbing_edge = absorbing_edge
        self.exs_field = exs_field
        self.site_spectra_field = site_spectra_field
        self.target_atom_type = ABSORBING_EDGE_TO_ATOM_TYPE[absorbing_edge]
        self.sample_index, self.debug_stats = self._build_sample_index()
        self._log_summary()

    def _build_sample_index(self):
        sample_index = []
        total_molecules = len(self.base_dataset)
        molecules_with_exs = 0
        molecules_with_target_atom = 0
        molecules_with_active_target_site = 0
        molecules_with_nonzero_target_site = 0
        total_active_sites = 0
        total_nonzero_active_sites = 0
        skipped_zero_spectrum_sites = 0

        for data_idx in range(len(self.base_dataset)):
            data = self.base_dataset[data_idx]
            if not hasattr(data, self.exs_field):
                continue
            molecules_with_exs += 1
            num_atom = int(data.num_atom)
            exs = getattr(data, self.exs_field).reshape(-1)[:num_atom]
            atom_types = data.atom_type.reshape(-1)[:num_atom]
            if (atom_types == self.target_atom_type).any():
                molecules_with_target_atom += 1
            active_sites = ((exs > 0) & (atom_types == self.target_atom_type)).nonzero(as_tuple=False).view(-1)
            if active_sites.numel() > 0:
                molecules_with_active_target_site += 1
                total_active_sites += active_sites.numel()
            filtered_active_sites = active_sites
            if hasattr(data, self.site_spectra_field):
                site_spectra = getattr(data, self.site_spectra_field)[:num_atom]
                site_nonzero = site_spectra.abs().sum(dim=1) > 0
                filtered_active_sites = active_sites[site_nonzero[active_sites]]
                skipped_zero_spectrum_sites += int(active_sites.numel() - filtered_active_sites.numel())
            if filtered_active_sites.numel() > 0:
                molecules_with_nonzero_target_site += 1
                total_nonzero_active_sites += filtered_active_sites.numel()
            for site_idx in filtered_active_sites.tolist():
                sample_index.append((data_idx, site_idx))
        debug_stats = {
            'total_molecules': total_molecules,
            'molecules_with_exs': molecules_with_exs,
            'molecules_with_target_atom': molecules_with_target_atom,
            'molecules_with_active_target_site': molecules_with_active_target_site,
            'molecules_with_nonzero_target_site': molecules_with_nonzero_target_site,
            'total_active_sites': total_active_sites,
            'total_nonzero_active_sites': total_nonzero_active_sites,
            'skipped_zero_spectrum_sites': skipped_zero_spectrum_sites,
        }
        return sample_index, debug_stats

    def _log_summary(self):
        logging.info(
            "SiteConditionedDataset[%s]: base=%d, with_%s=%d, with_target_atom=%d, "
            "with_active_target_site=%d, with_nonzero_target_site=%d, expanded_samples=%d",
            self.absorbing_edge,
            self.debug_stats['total_molecules'],
            self.exs_field,
            self.debug_stats['molecules_with_exs'],
            self.debug_stats['molecules_with_target_atom'],
            self.debug_stats['molecules_with_active_target_site'],
            self.debug_stats['molecules_with_nonzero_target_site'],
            len(self.sample_index),
        )

    def __len__(self):
        return len(self.sample_index)

    @staticmethod
    def _build_front_permutation(num_atom, site_idx, device):
        keep = [i for i in range(num_atom) if i != site_idx]
        return torch.tensor([site_idx] + keep, dtype=torch.long, device=device)

    @staticmethod
    def _permute_atom_vector(tensor, perm, num_atom):
        flat = tensor.reshape(-1)
        if flat.numel() < num_atom:
            return tensor
        reordered = flat.clone()
        reordered[:num_atom] = flat[:num_atom][perm.cpu()]
        return reordered.reshape_as(tensor)

    @staticmethod
    def _permute_atom_matrix(tensor, perm, num_atom):
        tensor = tensor.clone()
        if tensor.dim() == 2 and tensor.size(0) >= num_atom:
            tensor[:num_atom] = tensor[:num_atom][perm.cpu()]
            return tensor
        if tensor.dim() == 3 and tensor.size(0) == 1 and tensor.size(1) >= num_atom:
            tensor[:, :num_atom] = tensor[:, :num_atom][:, perm.cpu()]
            return tensor
        if tensor.dim() == 3 and tensor.size(0) >= num_atom:
            tensor[:num_atom] = tensor[:num_atom][perm.cpu()]
            return tensor
        return tensor

    @staticmethod
    def _permute_dense_edge_tensor(tensor, perm, num_atom):
        tensor = tensor.clone()
        perm_cpu = perm.cpu()
        if tensor.dim() == 3 and tensor.size(0) >= num_atom and tensor.size(1) >= num_atom:
            tensor[:num_atom, :num_atom] = tensor[:num_atom, :num_atom][perm_cpu][:, perm_cpu]
            return tensor
        if tensor.dim() == 4 and tensor.size(0) == 1 and tensor.size(1) >= num_atom and tensor.size(2) >= num_atom:
            tensor[:, :num_atom, :num_atom] = tensor[:, :num_atom, :num_atom][:, perm_cpu][:, :, perm_cpu]
            return tensor
        return tensor

    @staticmethod
    def _permute_edge_index(edge_index, perm, num_atom):
        perm_cpu = perm.cpu()
        inv_perm = torch.empty(num_atom, dtype=torch.long)
        inv_perm[perm_cpu] = torch.arange(num_atom, dtype=torch.long)
        edge_index = edge_index.clone()
        edge_index = edge_index[:, edge_index[0] < num_atom]
        edge_index = edge_index[:, edge_index[1] < num_atom]
        edge_index = inv_perm[edge_index]
        return edge_index

    def _reorder_site_to_front(self, data, site_idx):
        num_atom = int(data.num_atom)
        perm = self._build_front_permutation(num_atom, site_idx, data.atom_type.device)

        if hasattr(data, 'atom_type'):
            data.atom_type = self._permute_atom_vector(data.atom_type, perm, num_atom)
        if hasattr(data, 'pos'):
            data.pos = self._permute_atom_matrix(data.pos, perm, num_atom)
        if hasattr(data, 'fc'):
            data.fc = self._permute_atom_vector(data.fc, perm, num_atom)
        if hasattr(data, 'charge'):
            data.charge = self._permute_atom_vector(data.charge, perm, num_atom)
        if hasattr(data, 'atom_one_hot'):
            data.atom_one_hot = self._permute_atom_matrix(data.atom_one_hot, perm, num_atom)

        if hasattr(data, 'edge_one_hot'):
            data.edge_one_hot = self._permute_dense_edge_tensor(data.edge_one_hot, perm, num_atom)
        if hasattr(data, 'edge_index'):
            original_edge_index = data.edge_index.clone()
            data.edge_index = self._permute_edge_index(data.edge_index, perm, num_atom)
            if hasattr(data, 'edge_type'):
                valid_edges = torch.logical_and(original_edge_index[0] < num_atom, original_edge_index[1] < num_atom)
                data.edge_type = data.edge_type[valid_edges]

        if hasattr(data, self.exs_field):
            exs = getattr(data, self.exs_field)
            one_hot_exs = torch.zeros(exs.numel(), dtype=exs.dtype, device=exs.device)
            one_hot_exs[0] = 1
            setattr(data, self.exs_field, one_hot_exs.reshape_as(exs))

        site_field = getattr(data, 'site_spectra_field', None)
        if site_field is not None and hasattr(data, site_field):
            setattr(data, site_field, self._permute_atom_matrix(getattr(data, site_field), perm, num_atom))

        for candidate_field in ('site_spectra', 'xanes_CK', 'xanes_NK', 'xanes_OK'):
            if hasattr(data, candidate_field):
                setattr(data, candidate_field, self._permute_atom_matrix(getattr(data, candidate_field), perm, num_atom))

        if hasattr(data, 'rdmol') and data.rdmol is not None:
            rdmol = ensure_rdmol(data.rdmol)
            order = perm.cpu().tolist() + list(range(num_atom, rdmol.GetNumAtoms()))
            try:
                data.rdmol = Chem.RenumberAtoms(rdmol, order)
            except Exception:
                pass

        data.absorbing_site_idx = torch.tensor(0, dtype=torch.long)
        return data

    def __getitem__(self, idx):
        data_idx, site_idx = self.sample_index[idx]
        data = self.base_dataset[data_idx].clone()
        return self._reorder_site_to_front(data, site_idx)


class MoleculeAtomFilteredDataset(Dataset):
    def __init__(self, base_dataset, target_atom_types):
        self.base_dataset = base_dataset
        self.target_atom_types = tuple(sorted(set(int(v) for v in target_atom_types)))
        self.selected_indices, self.debug_stats = self._build_index()
        self._log_summary()

    def _build_index(self):
        selected_indices = []
        total_molecules = len(self.base_dataset)
        kept_molecules = 0
        for data_idx in range(total_molecules):
            data = self.base_dataset[data_idx]
            num_atom = int(data.num_atom)
            atom_types = data.atom_type.reshape(-1)[:num_atom]
            has_target = False
            for atom_type in self.target_atom_types:
                if (atom_types == atom_type).any():
                    has_target = True
                    break
            if has_target:
                selected_indices.append(data_idx)
                kept_molecules += 1
        debug_stats = {
            'total_molecules': total_molecules,
            'kept_molecules': kept_molecules,
            'filtered_out_molecules': total_molecules - kept_molecules,
        }
        return selected_indices, debug_stats

    def _log_summary(self):
        logging.info(
            "MoleculeAtomFilteredDataset[target_atom_types=%s]: base=%d, kept=%d, filtered_out=%d",
            self.target_atom_types,
            self.debug_stats['total_molecules'],
            self.debug_stats['kept_molecules'],
            self.debug_stats['filtered_out_molecules'],
        )

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.selected_indices[idx]]


class MinimalConditionDataset(Dataset):
    """Adapter for generation-only eval with minimal condition inputs."""

    def __init__(self, base_dataset, used_spectra):
        self.base_dataset = base_dataset
        self.used_spectra = tuple(used_spectra)
        self._validate_base_dataset()

    def _validate_base_dataset(self):
        if len(self.base_dataset) == 0:
            raise ValueError("MinimalConditionDataset received an empty base dataset.")
        sample = self.base_dataset[0]
        missing = []
        if not hasattr(sample, 'num_atom'):
            missing.append('num_atom')
        for spectrum_name in self.used_spectra:
            if not hasattr(sample, spectrum_name):
                missing.append(spectrum_name)
        if missing:
            raise ValueError(
                "Minimal condition eval requires samples to contain num_atom and "
                f"the requested spectra fields. Missing: {missing}"
            )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx].clone()
        num_atom = int(data.num_atom)

        if not hasattr(data, 'pos'):
            data.pos = torch.zeros((num_atom, 3), dtype=torch.float32)
        if not hasattr(data, 'rdmol'):
            data.rdmol = None
        if not hasattr(data, 'idx'):
            data.idx = torch.tensor(idx, dtype=torch.long)
        return data


def maybe_filter_dataset_by_condition_atoms(dataset, config, minimal_condition_only=False):
    if minimal_condition_only:
        return dataset

    if not getattr(config.data, 'filter_molecules_by_condition_atoms', False):
        return dataset

    used_spectra = get_used_spectra(config.data)
    target_atom_types = [
        ABSORBING_EDGE_TO_ATOM_TYPE[name]
        for name in used_spectra
        if name in ABSORBING_EDGE_TO_ATOM_TYPE
    ]
    if not target_atom_types:
        logging.info(
            "Atom-based dataset filtering is enabled, but no xanes spectra were found in used_spectra=%s. Skipping filter.",
            used_spectra,
        )
        return dataset

    filtered_dataset = MoleculeAtomFilteredDataset(dataset, target_atom_types)
    if len(filtered_dataset) == 0:
        raise ValueError(
            "Atom-based dataset filtering produced 0 samples. "
            f"used_spectra={used_spectra}, target_atom_types={target_atom_types}"
        )
    return filtered_dataset


def maybe_wrap_minimal_condition_dataset(dataset, config, minimal_condition_only=False):
    if not minimal_condition_only:
        return dataset
    return MinimalConditionDataset(dataset, get_used_spectra(config.data))


def maybe_wrap_site_conditioned_dataset(dataset, config):
    if not getattr(config.data, 'use_site_spectra', False):
        return dataset
    if not getattr(config.data, 'expand_site_samples', False):
        return dataset

    absorbing_edge = get_absorbing_edge(config.data)
    if not absorbing_edge:
        raise ValueError(
            "Site-conditioned expansion requires `config.data.absorbing_edge` or a single xanes spectrum in used_spectra."
        )
    wrapped_dataset = SiteConditionedDataset(
        dataset,
        absorbing_edge=absorbing_edge,
        exs_field=getattr(config.data, 'exs_field', 'exs'),
        site_spectra_field=getattr(config.data, 'site_spectra_field', 'site_spectra'),
    )
    if len(wrapped_dataset) == 0:
        stats = wrapped_dataset.debug_stats
        raise ValueError(
            "Site-conditioned dataset expansion produced 0 samples. "
            f"absorbing_edge={absorbing_edge}, exs_field={getattr(config.data, 'exs_field', 'exs')}, "
            f"site_spectra_field={getattr(config.data, 'site_spectra_field', 'site_spectra')}, "
            f"total_molecules={stats['total_molecules']}, "
            f"molecules_with_exs={stats['molecules_with_exs']}, "
            f"molecules_with_target_atom={stats['molecules_with_target_atom']}, "
            f"molecules_with_active_target_site={stats['molecules_with_active_target_site']}, "
            f"molecules_with_nonzero_target_site={stats['molecules_with_nonzero_target_site']}, "
            f"skipped_zero_spectrum_sites={stats['skipped_zero_spectrum_sites']}. "
            "Please check exs, atom_type encoding, site spectra contents, split contents, and absorbing_edge."
        )
    return wrapped_dataset


def validate_site_conditioned_fields(dataset, config):
    if not getattr(config.data, 'use_site_spectra', False):
        return

    required_fields = [getattr(config.data, 'exs_field', 'exs')]
    site_spectra_field = getattr(config.data, 'site_spectra_field', '')
    if site_spectra_field:
        required_fields.append(site_spectra_field)

    data_keys = set(dataset.data.to_dict().keys())
    missing_fields = [field for field in required_fields if field not in data_keys]
    if not missing_fields:
        return

    processed_file = getattr(config.data, 'processed_file', '')
    if processed_file and not os.path.isabs(processed_file):
        processed_file = os.path.join(config.data.root, 'processed', processed_file)

    raise ValueError(
        "Processed dataset is missing required site-conditioning fields. "
        f"processed_file={processed_file or '<default>'}, "
        f"missing_fields={missing_fields}, "
        f"available_fields={sorted(data_keys)}. "
        "Please regenerate the processed dataset or point config.data.processed_file "
        "to a site-conditioned .pt file that contains exs and the requested site spectra field."
    )


def summarize_dataset_obj(dataset, split_name):
    summary = {
        'split': split_name,
        'type': type(dataset).__name__,
        'num_samples': len(dataset),
    }
    if isinstance(dataset, SiteConditionedDataset):
        summary['base_dataset_type'] = type(dataset.base_dataset).__name__
        summary['base_num_samples'] = len(dataset.base_dataset)
        summary['absorbing_edge'] = dataset.absorbing_edge
        summary['exs_field'] = dataset.exs_field
        summary['site_expansion'] = dict(dataset.debug_stats)
    if isinstance(dataset, MoleculeAtomFilteredDataset):
        summary['base_dataset_type'] = type(dataset.base_dataset).__name__
        summary['base_num_samples'] = len(dataset.base_dataset)
        summary['target_atom_types'] = list(dataset.target_atom_types)
        summary['atom_filter'] = dict(dataset.debug_stats)
    if isinstance(dataset, MinimalConditionDataset):
        summary['base_dataset_type'] = type(dataset.base_dataset).__name__
        summary['base_num_samples'] = len(dataset.base_dataset)
        summary['used_spectra'] = list(dataset.used_spectra)
    return summary


def build_dataset_summary(config, train_ds=None, val_ds=None, test_ds=None, first_train_ds=None):
    processed_file = getattr(config.data, 'processed_file', '')
    if processed_file:
        if os.path.isabs(processed_file):
            resolved_processed_file = processed_file
        else:
            resolved_processed_file = os.path.join(config.data.root, 'processed', processed_file)
    else:
        resolved_processed_file = ''

    summary = {
        'data_root': config.data.root,
        'processed_file': processed_file,
        'resolved_processed_file': resolved_processed_file,
        'spectra_version': config.data.spectra_version,
        'used_spectra': list(get_used_spectra(config.data)),
        'use_site_spectra': getattr(config.data, 'use_site_spectra', False),
        'expand_site_samples': getattr(config.data, 'expand_site_samples', False),
        'absorbing_edge': getattr(config.data, 'absorbing_edge', ''),
        'site_spectra_field': getattr(config.data, 'site_spectra_field', ''),
        'exs_field': getattr(config.data, 'exs_field', ''),
        'splits': [],
    }

    if first_train_ds is not None:
        summary['splits'].append(summarize_dataset_obj(first_train_ds, 'first_train'))
    if train_ds is not None:
        summary['splits'].append(summarize_dataset_obj(train_ds, 'train'))
    if val_ds is not None:
        summary['splits'].append(summarize_dataset_obj(val_ds, 'valid'))
    if test_ds is not None:
        summary['splits'].append(summarize_dataset_obj(test_ds, 'test'))

    return summary


def get_dataset(config, transform=True, minimal_condition_only=False):
    """Create dataset for training and evaluation."""

    # Obtain dataset info
    dataset_info = get_dataset_info(config.data.info_name)  # 'qm9_second_half' -> qm9_second_half object

    # get transform
    if transform and not minimal_condition_only:         # True
        name_transform = getattr(config.data, 'transform', 'EdgeComSpectra')   # 'EdgeComSpectra'
        if name_transform == 'EdgeComSpectra':
            transform = EdgeComSpectraTransform(
                dataset_info['atom_encoder'].values(),
                config.data.include_aromatic,
                use_normalize=config.data.use_normalize,
                site_spectra_field=getattr(config.data, 'site_spectra_field', 'site_spectra'),
            )
        else:
            raise ValueError('Invalid data transform name')
    else:
        transform = None

    # Build up dataset
    if config.data.name == 'QM9S':  # QM9
        dataset = QM9SDataset(
            config.data.root,
            transform=transform,
            spectra_version=config.data.spectra_version,
            used_spectra=get_used_spectra(config.data),
            processed_file=getattr(config.data, 'processed_file', ''),
        )
        validate_site_conditioned_fields(dataset, config)
        # import pickle
        # with open ("data_set.pk","wb") as fp:
        #     pickle.dump(dataset,fp)
        # import sys
        # print("aaa")
        # sys.exit()
    else:
        raise ValueError('Undefined dataset name.')

    use_full_dataset_for_eval = getattr(config.eval, 'use_full_dataset', False)

    # Split dataset
    if 'diffspectra' == config.exp_type:
        split_idx = dataset.get_cond_idx_split()
        first_train_dataset = dataset.index_select(split_idx['first_train'])
        second_train_dataset = dataset.index_select(split_idx['second_train'])
        val_dataset = dataset.index_select(split_idx['valid'])
        test_dataset = dataset.index_select(split_idx['test'])
        first_train_dataset = maybe_wrap_site_conditioned_dataset(first_train_dataset, config)
        second_train_dataset = maybe_wrap_site_conditioned_dataset(second_train_dataset, config)
        val_dataset = maybe_wrap_site_conditioned_dataset(val_dataset, config)
        test_dataset = maybe_wrap_site_conditioned_dataset(test_dataset, config)
        # import pickle
        # with open ("test_dataset.pk","wb") as fp:
        #     pickle.dump(test_dataset,fp)
        # import sys
        # print("aaa")
        # sys.exit()
        return first_train_dataset, second_train_dataset, val_dataset, test_dataset, dataset_info
    
    if 'diffspectra_xanes' == config.exp_type:
        if use_full_dataset_for_eval:
            first_train_dataset = None
            second_train_dataset = maybe_filter_dataset_by_condition_atoms(dataset, config, minimal_condition_only=minimal_condition_only)
            val_dataset = maybe_filter_dataset_by_condition_atoms(dataset, config, minimal_condition_only=minimal_condition_only)
            test_dataset = maybe_filter_dataset_by_condition_atoms(dataset, config, minimal_condition_only=minimal_condition_only)
            second_train_dataset = maybe_wrap_minimal_condition_dataset(second_train_dataset, config, minimal_condition_only=minimal_condition_only)
            val_dataset = maybe_wrap_minimal_condition_dataset(val_dataset, config, minimal_condition_only=minimal_condition_only)
            test_dataset = maybe_wrap_minimal_condition_dataset(test_dataset, config, minimal_condition_only=minimal_condition_only)
            second_train_dataset = maybe_wrap_site_conditioned_dataset(second_train_dataset, config)
            val_dataset = maybe_wrap_site_conditioned_dataset(val_dataset, config)
            test_dataset = maybe_wrap_site_conditioned_dataset(test_dataset, config)
            return first_train_dataset, second_train_dataset, val_dataset, test_dataset, dataset_info

        split_idx = dataset.get_cond_idx_split(my_data=True)
        print(split_idx)
        first_train_dataset = None
        second_train_dataset = dataset.index_select(split_idx['train'])
        val_dataset = dataset.index_select(split_idx['valid'])
        test_dataset = dataset.index_select(split_idx['test'])
        second_train_dataset = maybe_filter_dataset_by_condition_atoms(second_train_dataset, config, minimal_condition_only=minimal_condition_only)
        val_dataset = maybe_filter_dataset_by_condition_atoms(val_dataset, config, minimal_condition_only=minimal_condition_only)
        test_dataset = maybe_filter_dataset_by_condition_atoms(test_dataset, config, minimal_condition_only=minimal_condition_only)
        second_train_dataset = maybe_wrap_minimal_condition_dataset(second_train_dataset, config, minimal_condition_only=minimal_condition_only)
        val_dataset = maybe_wrap_minimal_condition_dataset(val_dataset, config, minimal_condition_only=minimal_condition_only)
        test_dataset = maybe_wrap_minimal_condition_dataset(test_dataset, config, minimal_condition_only=minimal_condition_only)
        second_train_dataset = maybe_wrap_site_conditioned_dataset(second_train_dataset, config)
        val_dataset = maybe_wrap_site_conditioned_dataset(val_dataset, config)
        test_dataset = maybe_wrap_site_conditioned_dataset(test_dataset, config)
        # import pickle
        # with open ("test_dataset.pk","wb") as fp:
        #     pickle.dump(test_dataset,fp)
        # import sys
        # print("aaa")
        # sys.exit()
        return first_train_dataset, second_train_dataset, val_dataset, test_dataset, dataset_info

    split_idx = dataset.get_idx_split()
    train_dataset = dataset.index_select(split_idx['train'])
    val_dataset = dataset.index_select(split_idx['valid'])
    test_dataset = dataset.index_select(split_idx['test'])

    return train_dataset, val_dataset, test_dataset, dataset_info


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


# setup dataloader
def get_dataloader(train_ds, val_ds, test_ds, config):
    used_spectra = get_used_spectra(config.data)
    if config.model.name == 'DiffSpectra_DMT' or config.model.name == "DMT":
        aug_rotation = True
        aug_translation = True
    elif config.model.name == 'DiffSpectra_DGT' or config.model.name == 'CDGS':
        aug_rotation = False
        aug_translation = False
    else:
        raise ValueError(f"Model {config.model.name} not recognized.")

    if config.only_2D:
        collate_fn = CollateSpectr2D(
            spectra_version=config.data.spectra_version,
            used_spectra=used_spectra,
            use_site_spectra=getattr(config.data, 'use_site_spectra', False),
            site_spectra_field=getattr(config.data, 'site_spectra_field', 'site_spectra'),
            exs_field=getattr(config.data, 'exs_field', 'exs'),
        )
    else:
        collate_fn = CollateSpectra(
            spectra_version=config.data.spectra_version,
            used_spectra=used_spectra,
            use_site_spectra=getattr(config.data, 'use_site_spectra', False),
            site_spectra_field=getattr(config.data, 'site_spectra_field', 'site_spectra'),
            exs_field=getattr(config.data, 'exs_field', 'exs'),
            aug_rotation=aug_rotation,
            aug_translation=aug_translation,
            aug_translation_scale=config.data.aug_translation_scale
        )

    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True,
                                num_workers=config.data.num_workers, collate_fn=collate_fn, drop_last=config.training.dataloader_drop_last)
    val_loader = DataLoader(val_ds, batch_size=config.training.eval_batch_size, shuffle=False,
                                num_workers=config.data.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config.training.eval_batch_size, shuffle=False,
                                num_workers=config.data.num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


# transform data
class EdgeComSpectraTransform(object):
    """
    Transform data with node and edge features. Compress single/double/triple bond types to one channel.
    Edge:
        0-th ch: exist edge or not
        1-th ch: 0, 1, 2, 3; other bonds, single, double, triple bonds
        2-th ch: aromatic bond or not
    """

    def __init__(self, atom_type_list, include_aromatic, use_normalize=True, site_spectra_field='site_spectra'):
        super().__init__()
        self.atom_type_list = torch.tensor(list(atom_type_list))
        self.include_aromatic = include_aromatic
        self.use_normalize = use_normalize
        self.site_spectra_field = site_spectra_field

    def __call__(self, data: Data):
        # add atom type one_hot
        atom_type = data.atom_type
        edge_type = data.edge_type

        atom_one_hot = atom_type.unsqueeze(-1) == self.atom_type_list.unsqueeze(0)
        data.atom_one_hot = atom_one_hot.float()

        edge_bond = edge_type.clone()
        edge_bond[edge_bond == 4] = 0
        edge_bond = edge_bond / 3.
        edge_feat = [edge_bond]
        if self.include_aromatic:
            edge_aromatic = (edge_type == 4).float()
            edge_feat.append(edge_aromatic)
        edge_feat = torch.stack(edge_feat, dim=-1)

        edge_index = data.edge_index
        dense_shape = (data.num_nodes, data.num_nodes, edge_feat.size(-1))
        dense_edge_one_hot = torch.zeros((data.num_nodes**2, edge_feat.size(-1)), device=atom_type.device)

        idx1, idx2 = edge_index[0], edge_index[1]
        idx = idx1 * data.num_nodes + idx2
        idx = idx.unsqueeze(-1).expand(edge_feat.size())
        dense_edge_one_hot.scatter_add_(0, idx, edge_feat)
        dense_edge_one_hot = dense_edge_one_hot.reshape(dense_shape)

        # edge feature channel [edge_exist; bond_order; aromatic_exist]
        edge_exist = (dense_edge_one_hot.sum(dim=-1, keepdim=True) != 0).float()
        dense_edge_one_hot = torch.cat([edge_exist, dense_edge_one_hot], dim=-1)
        data.edge_one_hot = dense_edge_one_hot

        # log(x+1) normalization for molecular spectral data
        if self.use_normalize:
            normalized_fields = set()

            def _normalize_field(field_name):
                if hasattr(data, field_name) and field_name not in normalized_fields:
                    setattr(data, field_name, torch.log10(getattr(data, field_name) + 1))
                    normalized_fields.add(field_name)

            if hasattr(data, 'ir'):
                _normalize_field('ir')
            if hasattr(data, 'uv'):
                _normalize_field('uv')
            if hasattr(data, 'raman'):
                _normalize_field('raman')
            if hasattr(data, 'xanes_CK'):
                _normalize_field('xanes_CK')
            if hasattr(data, 'xanes_NK'):
                _normalize_field('xanes_NK')
            if hasattr(data, 'xanes_OK'):
                _normalize_field('xanes_OK')
            if hasattr(data, self.site_spectra_field):
                _normalize_field(self.site_spectra_field)
        return data


def pad_node_feature(x, pad_len):
    x_len, x_dim = x.size()
    if x_len < pad_len:
        new_x = x.new_zeros([pad_len, x_dim], dtype=x.dtype)
        new_x[:x_len, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_edge_feature(x, pad_len):
    # x: [N_node, N_node, ch]
    x_len, _, x_dim = x.size()
    if x_len < pad_len:
        new_x = x.new_zeros([pad_len, pad_len, x_dim])
        new_x[:x_len, :x_len, :] = x
        x = new_x
    return x.unsqueeze(0)


def get_node_mask(node_num, pad_len, dtype):
    node_mask = torch.zeros(pad_len, dtype=dtype)
    node_mask[:node_num] = 1.
    return node_mask.unsqueeze(0)


# collate function: padding with the max node

def collate_node(items):
    items = [(item.one_hot, item.pos, item.fc, item.num_atom) for item in items]
    one_hot, positions, formal_charges, num_atoms = zip(*items)
    max_node_num = max(num_atoms)

    # padding features
    one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in one_hot])
    positions = torch.cat([pad_node_feature(i, max_node_num) for i in positions])
    formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    # repack
    return dict(
        one_hot=one_hot,
        atom_mask=node_mask,
        edge_mask=edge_mask,
        positions=positions,
        formal_charges=formal_charges
    )


def collate_edge(items):
    items = [(item.atom_one_hot, item.edge_one_hot, item.fc, item.pos, item.num_atom) for item in items]
    atom_one_hot, edge_one_hot, formal_charges, positions, num_atoms = zip(*items)

    max_node_num = max(num_atoms)

    # padding features
    atom_one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in atom_one_hot])
    formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])
    positions = torch.cat([pad_node_feature(i, max_node_num) for i in positions])
    edge_one_hot = torch.cat([pad_edge_feature(i, max_node_num) for i in edge_one_hot])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, atom_one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    # repack
    return dict(
        atom_one_hot=atom_one_hot,
        edge_one_hot=edge_one_hot,
        positions=positions,
        formal_charges=formal_charges,
        atom_mask=node_mask,
        edge_mask=edge_mask
    )


def collate_edge_2D(items):
    items = [(item.atom_one_hot, item.edge_one_hot, item.fc, item.num_atom) for item in items]
    atom_one_hot, edge_one_hot, formal_charges, num_atoms = zip(*items)

    max_node_num = max(num_atoms)

    # padding features
    atom_one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in atom_one_hot])
    formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])
    edge_one_hot = torch.cat([pad_edge_feature(i, max_node_num) for i in edge_one_hot])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, atom_one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    # repack
    return dict(
        atom_one_hot=atom_one_hot,
        edge_one_hot=edge_one_hot,
        formal_charges=formal_charges,
        atom_mask=node_mask,
        edge_mask=edge_mask
    )


def collate_cond(items):
    # collate_fn for the condition generation
    items = [(item.atom_one_hot, item.edge_one_hot, item.fc, item.pos, item.num_atom, item.property) for item in items]
    atom_one_hot, edge_one_hot, formal_charges, positions, num_atoms, property = zip(*items)

    max_node_num = max(num_atoms)

    # padding features
    atom_one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in atom_one_hot])
    formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])
    positions = torch.cat([pad_node_feature(i, max_node_num) for i in positions])
    edge_one_hot = torch.cat([pad_edge_feature(i, max_node_num) for i in edge_one_hot])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, atom_one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    # context property
    property = torch.stack(property, dim=0)

    # repack
    return dict(
        atom_one_hot=atom_one_hot,
        edge_one_hot=edge_one_hot,
        positions=positions,
        formal_charges=formal_charges,
        atom_mask=node_mask,
        edge_mask=edge_mask,
        context=property
    )

class CollateSpectra(object):
    def __init__(self, spectra_version='allspectra', used_spectra=None, use_site_spectra=False,
                 site_spectra_field='site_spectra', exs_field='exs',
                 aug_rotation=False, aug_translation=False, aug_translation_scale=0.01):
        self.spectra_version = spectra_version
        self.used_spectra = list(used_spectra or [])
        self.use_site_spectra = use_site_spectra
        self.site_spectra_field = site_spectra_field
        self.exs_field = exs_field
        self.aug_rotation = aug_rotation
        self.aug_translation = aug_translation
        self.aug_translation_scale = aug_translation_scale

    def augment_positions(self, positions, atom_mask):
        batch_size = positions.shape[0]
        num_max_nodes = positions.shape[1]
        dtype = positions.dtype
        
        # Create mask to distinguish between actual atoms and padding positions
        mask = atom_mask.unsqueeze(-1)  # [batch_size, num_nodes, 1]
        
        if self.aug_rotation:
            rot_aug = Rotation.random(batch_size)
            # Only rotate actual atom positions
            positions_np = positions.numpy()
            # Rotate each molecule
            for i in range(batch_size):
                positions_np[i] = rot_aug[i].apply(positions_np[i])
            positions = torch.from_numpy(positions_np).to(dtype)
            # Keep padding positions as 0
            positions = positions * mask
            
        if self.aug_translation:
            # Only translate actual atom positions
            trans_aug = self.aug_translation_scale * torch.randn(batch_size, 1, 3, dtype=dtype).repeat(1, num_max_nodes, 1)
            positions = positions + trans_aug
            # Keep padding positions as 0
            positions = positions * mask
            
        return positions

    def pad_and_mask(self, atom_one_hot, edge_one_hot, formal_charges, positions, num_atoms):
        max_node_num = max(num_atoms)

        # padding features
        atom_one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in atom_one_hot])
        formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])
        positions = torch.cat([pad_node_feature(i, max_node_num) for i in positions])
        edge_one_hot = torch.cat([pad_edge_feature(i, max_node_num) for i in edge_one_hot])

        # atom mask
        node_mask = torch.cat([get_node_mask(i, max_node_num, atom_one_hot.dtype) for i in num_atoms])

        # edge mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.reshape(-1, 1)

        return atom_one_hot, edge_one_hot, formal_charges, positions, node_mask, edge_mask

    def __call__(self, items):
        struct_items = [(item.atom_one_hot, item.edge_one_hot, item.fc, item.pos, item.num_atom) for item in items]
        atom_one_hot, edge_one_hot, formal_charges, positions, num_atoms = zip(*struct_items)

        # Use base class pad_and_mask method
        atom_one_hot, edge_one_hot, formal_charges, positions, node_mask, edge_mask = self.pad_and_mask(
            atom_one_hot, edge_one_hot, formal_charges, positions, num_atoms
        )

        spectra = build_condition_context(
            items,
            self.used_spectra,
            use_site_spectra=self.use_site_spectra,
            site_spectra_field=self.site_spectra_field,
            exs_field=self.exs_field,
        )

        # Data augmentation
        positions = self.augment_positions(positions, node_mask)

        return dict(
            atom_one_hot=atom_one_hot,
            edge_one_hot=edge_one_hot,
            positions=positions,
            formal_charges=formal_charges,
            atom_mask=node_mask,
            edge_mask=edge_mask,
            context=spectra
        )


class CollateSpectr2D(object):
    def __init__(self, spectra_version='allspectra', used_spectra=None, use_site_spectra=False,
                 site_spectra_field='site_spectra', exs_field='exs'):
        self.spectra_version = spectra_version
        self.used_spectra = list(used_spectra or [])
        self.use_site_spectra = use_site_spectra
        self.site_spectra_field = site_spectra_field
        self.exs_field = exs_field

    def pad_and_mask(self, atom_one_hot, edge_one_hot, formal_charges, num_atoms):
        max_node_num = max(num_atoms)

        # padding features
        atom_one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in atom_one_hot])
        formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])
        edge_one_hot = torch.cat([pad_edge_feature(i, max_node_num) for i in edge_one_hot])

        # atom mask
        node_mask = torch.cat([get_node_mask(i, max_node_num, atom_one_hot.dtype) for i in num_atoms])

        # edge mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.reshape(-1, 1)

        return atom_one_hot, edge_one_hot, formal_charges, node_mask, edge_mask

    def __call__(self, items):
        struct_items = [(item.atom_one_hot, item.edge_one_hot, item.fc, item.num_atom) for item in items]
        atom_one_hot, edge_one_hot, formal_charges, num_atoms = zip(*struct_items)

        # Use base class pad_and_mask method
        atom_one_hot, edge_one_hot, formal_charges, node_mask, edge_mask = self.pad_and_mask(
            atom_one_hot, edge_one_hot, formal_charges, num_atoms
        )

        spectra = build_condition_context(
            items,
            self.used_spectra,
            use_site_spectra=self.use_site_spectra,
            site_spectra_field=self.site_spectra_field,
            exs_field=self.exs_field,
        )


        return dict(
            atom_one_hot=atom_one_hot,
            edge_one_hot=edge_one_hot,
            formal_charges=formal_charges,
            atom_mask=node_mask,
            edge_mask=edge_mask,
            context=spectra
        )
