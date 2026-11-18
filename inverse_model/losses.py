# Modified in 2026 by Yu Fujikata for XANES conditioning and public release.
# Derived from DiffSpectra and distributed under Apache-2.0; see LICENSE.
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from random import random
from utils import *
import profiling

from models.utils import remove_mean_with_mask, check_mask_correct, \
    assert_mean_zero_with_mask, assert_correctly_masked, sample_combined_position_feature_noise, \
    sample_symmetric_edge_feature_noise, sample_gaussian_with_mask


def get_optimizer(config, params):
    """Return a flax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=config.optim.lr, amsgrad=True, weight_decay=1e-12)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!'
        )
    return optimizer


def gradient_clipping(params, gradnorm_queue, max_grad, disable_log):
    if max_grad <= 1.0:
        torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad)
        return

    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()
    max_grad_norm = min(max_grad_norm, max_grad)

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        params, max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if not disable_log:
        if float(grad_norm) > 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std():
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')
    return grad_norm


class Queue():
    # from EDM
    # Gradient clipping
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def optimization_manager(config):
    """Return an optimize_fn based on `config`."""

    gradnorm_queue = Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.
    disable_log = config.optim.disable_grad_log

    def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimize with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            # torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            gradient_clipping(params, gradnorm_queue, grad_clip, disable_log)
        optimizer.step()

    return optimize_fn


def get_step_fn(noise_scheduler, train, optimize_fn, scaler, config, prop_dist=None):
    if config.pred_edge:
        if config.only_2D:
            loss_fn = get_sde_2D_loss_fn(noise_scheduler, train, scaler, config)
        else:
            loss_fn = get_sde_graph_loss_fn(noise_scheduler, train, scaler, config, prop_dist)    # will execute this loss function
    else:
         loss_fn = get_sde_node_loss_fn(noise_scheduler, train, scaler, config)

    def step_fn(state, batch):
        model = state['model']
        log_freq = int(getattr(getattr(config, "profiling", None), "log_freq", 100))
        reset_after_log = bool(getattr(getattr(config, "profiling", None), "reset_after_log", True))
        top_k = getattr(getattr(config, "profiling", None), "top_k", None)
        if top_k is not None:
            top_k = int(top_k)
        if train:
            optimizer = state['optimizer']
            with profiling.record("step_total"):
                with profiling.record("optimizer_zero_grad"):
                    optimizer.zero_grad()
                with profiling.record("loss_forward_total"):
                    loss = loss_fn(model, batch)    # forward diffusion and compute training loss
                with profiling.record("loss_backward"):
                    loss.backward()
                with profiling.record("optimizer_step"):
                    optimize_fn(optimizer, model.parameters(), step=state['step'])
                state['step'] += 1
                with profiling.record("ema_update"):
                    state['ema'].update(model.parameters())
            if profiling.enabled() and log_freq > 0 and state['step'] % log_freq == 0:
                profiling.report(step=state['step'], reset_after=reset_after_log, top_k=top_k)
        else:
            with torch.no_grad():
                ema = state['ema']
                with profiling.record("eval_step_total"):
                    with profiling.record("ema_store_copy"):
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                    with profiling.record("loss_forward_total"):
                        loss = loss_fn(model, batch)
                    with profiling.record("ema_restore"):
                        ema.restore(model.parameters())
        return loss

    return step_fn


def get_sde_node_loss_fn(noise_scheduler, train, scaler, config):
    """The loss function for node features and positions."""
    device = config.device
    include_charges = config.model.include_fc_charge
    pred_data = config.model.pred_data
    reduce_mean = config.training.reduce_mean

    loss_weights = config.model.loss_weights.split(',')
    loss_weights = [float(loss_weight) for loss_weight in loss_weights]
    noise_align = config.model.noise_align
    self_cond = config.model.self_cond

    def loss_fn(model, batch):
        if train:
            model.train()
        else:
            model.eval()

        # process and normalize the batch data
        xh, node_mask, edge_mask = process_batch(batch, device, include_charges, scaler)

        n_nodes = torch.sum(node_mask.squeeze(-1), dim=-1)
        # sample the noisy samples
        t_eps = 1e-5
        t = torch.rand(xh.shape[0], device=xh.device) * (1. - t_eps) + t_eps
        alpha_t, sigma_t = noise_scheduler.marginal_prob(t)
        noise = sample_combined_position_feature_noise(xh.shape[0], xh.shape[1], xh.shape[2] - 3, node_mask)

        z_t = expand_dims(alpha_t, xh.dim()) * xh + expand_dims(sigma_t, noise.dim()) * noise
        # assert_mean_zero_with_mask(z_t[:, :, :3], node_mask)
        if noise_align:
            if pred_data:
                align_pos = get_align_position(z_t, xh)
            else:
                noise = get_align_noise(z_t, xh, alpha_t, sigma_t, noise, node_mask)

        # compute loss
        noise_level = torch.log(alpha_t**2 / sigma_t**2)
        if self_cond:
            assert pred_data
            cond_x = None
            if random() < 0.5:
                with torch.no_grad():
                    cond_x = model(t, z_t, node_mask, edge_mask, noise_level=noise_level, cond_x=cond_x)
                    cond_x = cond_x.detach_()
            pred = model(t, z_t, node_mask, edge_mask, noise_level=noise_level, cond_x=cond_x)
        else:
            pred = model(t, z_t, node_mask, edge_mask, noise_level=noise_level)

        if pred_data:
            # data prediction loss
            losses_pos = torch.square(pred[:, :, :3] - align_pos)
            losses_pos = torch.mean(losses_pos, dim=-1)
            losses_pos = torch.sum(losses_pos, dim=-1)
            atom_type_pred = pred[:, :, 3:]
            atom_type_tar = xh[:, :, 3:]
            losses_atom_types = torch.square(atom_type_pred - atom_type_tar).mean(dim=-1)
            losses_atom_types = torch.sum(losses_atom_types, dim=-1)
        else:
            # noise prediction loss
            losses_atom = torch.square(noise - pred)
            losses_pos = losses_atom[:, :, :3]
            losses_atom_types = losses_atom[:, :, 3:]
            losses_pos = torch.mean(losses_pos, dim=-1)
            losses_atom_types = torch.mean(losses_atom_types, dim=-1)
            losses_pos = torch.sum(losses_pos, dim=-1)
            losses_atom_types = torch.sum(losses_atom_types, dim=-1)

        if reduce_mean:
            losses_pos = losses_pos / n_nodes
            losses_atom_types = losses_atom_types / n_nodes

        losses = loss_weights[0] * losses_pos + loss_weights[1] * losses_atom_types
        if pred_data:
            norm = torch.sqrt(alpha_t / sigma_t)
            losses = expand_dims(norm, losses.dim()) * losses

        return losses.mean()

    return loss_fn


def get_sde_2D_loss_fn(noise_scheduler, train, scaler, config):
    """The loss function for node features and edges. (2D graphs)"""
    device = config.device
    include_charges = config.model.include_fc_charge
    pred_data = config.model.pred_data
    reduce_mean = config.training.reduce_mean

    loss_weights = config.model.loss_weights.split(',')
    loss_weights = [float(loss_weight) for loss_weight in loss_weights]
    self_cond = config.model.self_cond

    def loss_fn(model, batch):
        if train:
            model.train()
        else:
            model.eval()

        # process and normalize the batch data
        xh, edge_x, node_mask, edge_mask, context = process_batch_2D(batch, device, include_charges, scaler)

        n_nodes = torch.sum(node_mask.squeeze(-1), dim=-1)
        # sample the noisy samples
        t_eps = 1e-5
        t = torch.rand(xh.shape[0], device=xh.device) * (1. - t_eps) + t_eps
        alpha_t, sigma_t = noise_scheduler.marginal_prob(t)
        noise = sample_gaussian_with_mask(xh.size(), xh.device, node_mask)
        edge_noise = sample_symmetric_edge_feature_noise(edge_x.shape[0], edge_x.shape[1], edge_x.shape[-1], edge_mask)

        z_t = expand_dims(alpha_t, xh.dim()) * xh + expand_dims(sigma_t, noise.dim()) * noise
        edge_z_t = expand_dims(alpha_t, edge_x.dim()) * edge_x + expand_dims(sigma_t, edge_noise.dim()) * edge_noise

        # compute loss
        noise_level = torch.log(alpha_t**2 / sigma_t**2)
        if self_cond:
            assert pred_data
            cond_x, cond_edge_x = None, None
            if random() < 0.5:
                with torch.no_grad():
                    cond_x, cond_edge_x = model(t, z_t, node_mask, edge_mask, edge_x=edge_z_t, noise_level=noise_level,
                                                cond_x=cond_x, cond_edge_x=cond_edge_x, context=context)
                    cond_x, cond_edge_x = cond_x.detach_(), cond_edge_x.detach_()
            pred, edge_pred = model(t, z_t, node_mask, edge_mask, edge_x=edge_z_t, noise_level=noise_level,
                                    cond_x=cond_x, cond_edge_x=cond_edge_x, context=context)
        else:
            pred, edge_pred = model(t, z_t, node_mask, edge_mask, edge_x=edge_z_t, noise_level=noise_level, context=context)

        if pred_data:
            # data prediction loss
            losses_atom_types = torch.square(pred - xh).mean(dim=-1)
            losses_atom_types = torch.sum(losses_atom_types, dim=-1)
            losses_edge = torch.square(edge_x - edge_pred).mean(dim=-1)
            losses_edge = torch.sum(losses_edge.reshape(xh.size(0), -1), dim=-1)
        else:
            # noise prediction loss
            losses_atom_types = torch.square(noise - pred)
            losses_atom_types = torch.mean(losses_atom_types, dim=-1)
            losses_atom_types = torch.sum(losses_atom_types, dim=-1)
            losses_edge = torch.square(edge_noise - edge_pred)
            losses_edge = torch.mean(losses_edge, dim=-1)
            losses_edge = torch.sum(losses_edge.reshape(losses_edge.shape[0], -1), dim=-1)

        if reduce_mean:
            losses_atom_types = losses_atom_types / n_nodes
            losses_edge = losses_edge / (torch.sum(edge_mask.reshape(losses_edge.shape[0], -1), dim=-1) + 1e-8)

        losses = loss_weights[1] * losses_atom_types + loss_weights[2] * losses_edge

        if pred_data:
            norm = torch.sqrt(alpha_t / sigma_t)
            losses = expand_dims(norm, losses.dim()) * losses

        return losses.mean()

    return loss_fn


def get_sde_graph_loss_fn(noise_scheduler, train, scaler, config, prop_norm=None):
    """The loss function for node features, positions and edge features"""

    device = config.device
    include_charges = config.model.include_fc_charge
    reduce_mean = config.training.reduce_mean
    noise_align = config.model.noise_align
    pred_data = config.model.pred_data
    loss_weights = config.model.loss_weights.split(',')
    loss_weights = [float(loss_weight) for loss_weight in loss_weights]
    self_cond = config.model.self_cond
    local_loss_beta = getattr(config.model, 'local_loss_beta', 0.0)
    local_loss_hops = getattr(config.model, 'local_loss_hops', 0)
    use_local_loss = getattr(config.data, 'use_site_spectra', False) and local_loss_beta > 0 and local_loss_hops > 0
    if self_cond:
        cond_process_fn = get_self_cond_fn(config)
    model_name = config.model.name

    def get_local_masks(edge_x, node_mask):
        batch_size, max_n_nodes = node_mask.shape[0], node_mask.shape[1]
        node_valid = node_mask.squeeze(-1) > 0
        if config.data.centered:
            adj = edge_x[..., 0] > 0
        else:
            adj = edge_x[..., 0] > 0.5
        eye = torch.eye(max_n_nodes, device=edge_x.device, dtype=torch.bool).unsqueeze(0)
        adj = adj & ~eye
        adj = adj & node_valid.unsqueeze(1) & node_valid.unsqueeze(2)

        local_node_mask = torch.zeros_like(node_valid)
        local_node_mask[:, 0] = node_valid[:, 0]
        frontier = local_node_mask.clone()
        for _ in range(local_loss_hops):
            frontier = (adj.float() @ frontier.float().unsqueeze(-1)).squeeze(-1) > 0
            local_node_mask = local_node_mask | frontier
        local_node_mask = local_node_mask & node_valid

        local_edge_mask = (
            local_node_mask.unsqueeze(1)
            & local_node_mask.unsqueeze(2)
            & node_valid.unsqueeze(1)
            & node_valid.unsqueeze(2)
            & ~eye
        ).float()
        return local_node_mask.float(), local_edge_mask

    def loss_fn(model, batch):
        if train:
            model.train()
        else:
            model.eval()

        # process and normalize the batch data with edge data
        with profiling.record("loss/process_edge_batch"):
            xh, edge_x, node_mask, edge_mask, context = process_edge_batch(batch, device, include_charges,
                                                                            scaler, prop_norm, model_name)
            n_nodes = torch.sum(node_mask.squeeze(-1), dim=-1)

        # sample the noisy samples
        with profiling.record("loss/noise_sampling"):
            t_eps = 1e-5
            t = torch.rand(xh.shape[0], device=xh.device) * (1. - t_eps) + t_eps
            alpha_t, sigma_t = noise_scheduler.marginal_prob(t)
            noise = sample_combined_position_feature_noise(xh.shape[0], xh.shape[1], xh.shape[2] - 3, node_mask)
            edge_noise = sample_symmetric_edge_feature_noise(edge_x.shape[0], edge_x.shape[1], edge_x.shape[-1], edge_mask)

            z_t = expand_dims(alpha_t, xh.dim()) * xh + expand_dims(sigma_t, noise.dim()) * noise
            edge_z_t = expand_dims(alpha_t, edge_x.dim()) * edge_x + expand_dims(sigma_t, edge_noise.dim()) * edge_noise

        # align position noise
        with profiling.record("loss/noise_align_total"):
            if noise_align:
                if pred_data:
                    # If it's a DMT model, the real structure has translation data augmentation and is not zero-centered, while the predicted structure is zero-centered, so we need to convert the real structure to zero-centered first
                    if config.model.name == 'DMT':
                        align_pos = get_align_position(z_t, xh)
                    elif config.model.name == 'DMT_WO_EQ':
                        pos_0_com = remove_mean_with_mask(xh[:, :, :3], node_mask)
                        pos_t_com = remove_mean_with_mask(z_t[:, :, :3], node_mask)
                        align_pos = get_align_position_v2(pos_t_com, pos_0_com)
                    else:
                        raise NotImplementedError(f'{config.model.name} not supported yet!')
                else:
                    noise = get_align_noise(z_t, xh, alpha_t, sigma_t, noise, node_mask)
            else:
                align_pos = xh[:, :, :3]

        # compute loss
        with profiling.record("loss/noise_level"):
            noise_level = torch.log(alpha_t**2 / sigma_t**2)
        if self_cond:
            assert pred_data
            cond_x, cond_edge_x = None, None
            if random() < 0.5:
                with torch.no_grad():
                    with profiling.record("loss/model_self_cond_forward"):
                        cond_x, cond_edge_x = model(t, z_t, node_mask, edge_mask, edge_x=edge_z_t, noise_level=noise_level,
                                                    cond_x=cond_x, cond_edge_x=cond_edge_x, context=context,
                                                    alpha_t=alpha_t, sigma_t=sigma_t)
                    cond_x, cond_edge_x = cond_x.detach_(), cond_edge_x.detach_()
                    # post_process self_cond values
                    with profiling.record("loss/self_cond_post_process"):
                        cond_x, cond_edge_x = cond_process_fn(cond_x, cond_edge_x)
            with profiling.record("loss/model_forward"):
                pred, edge_pred = model(t, z_t, node_mask, edge_mask, edge_x=edge_z_t, noise_level=noise_level,
                                        cond_x=cond_x, cond_edge_x=cond_edge_x, context=context,
                                        alpha_t=alpha_t, sigma_t=sigma_t)
        else:
            with profiling.record("loss/model_forward"):
                pred, edge_pred = model(t, z_t, node_mask, edge_mask, edge_x=edge_z_t, noise_level=noise_level,
                                        context=context, alpha_t=alpha_t, sigma_t=sigma_t)

        with profiling.record("loss/loss_terms"):
            if pred_data:
                # data prediction loss
                losses_pos_per_node = torch.square(pred[:, :, :3] - align_pos)
                losses_pos_per_node = torch.mean(losses_pos_per_node, dim=-1)
                atom_type_pred = pred[:, :, 3:]
                atom_type_tar = xh[:, :, 3:]
                losses_atom_per_node = torch.square(atom_type_pred - atom_type_tar).mean(dim=-1)
                losses_edge_per_edge = torch.square(edge_x - edge_pred).mean(dim=-1)
            else:
                # noise prediction loss
                losses_atom = torch.square(noise - pred)
                losses_edge = torch.square(edge_noise - edge_pred)
                losses_pos_per_node = torch.mean(losses_atom[:, :, :3], dim=-1)
                losses_atom_per_node = torch.mean(losses_atom[:, :, 3:], dim=-1)
                losses_edge_per_edge = torch.mean(losses_edge, dim=-1)

            losses_pos = torch.sum(losses_pos_per_node, dim=-1)
            losses_atom_types = torch.sum(losses_atom_per_node, dim=-1)
            losses_edge = torch.sum(losses_edge_per_edge.reshape(xh.size(0), -1), dim=-1)

            if reduce_mean:
                losses_pos = losses_pos / n_nodes
                losses_atom_types = losses_atom_types / n_nodes
                losses_edge = losses_edge / (torch.sum(edge_mask.reshape(losses_edge.shape[0], -1), dim=-1) + 1e-8)
            global_losses = loss_weights[0] * losses_pos + loss_weights[1] * losses_atom_types + loss_weights[2] * losses_edge

            if use_local_loss:
                local_node_mask, local_edge_mask = get_local_masks(edge_x, node_mask)
                local_pos = torch.sum(losses_pos_per_node * local_node_mask, dim=-1)
                local_atom = torch.sum(losses_atom_per_node * local_node_mask, dim=-1)
                local_edge = torch.sum(losses_edge_per_edge * local_edge_mask, dim=(1, 2))

                if reduce_mean:
                    local_node_denom = local_node_mask.sum(dim=-1).clamp_min(1.0)
                    local_edge_denom = local_edge_mask.sum(dim=(1, 2)).clamp_min(1.0)
                    local_pos = local_pos / local_node_denom
                    local_atom = local_atom / local_node_denom
                    local_edge = local_edge / local_edge_denom

                local_losses = (
                    loss_weights[0] * local_pos
                    + loss_weights[1] * local_atom
                    + loss_weights[2] * local_edge
                )
                losses = global_losses + local_loss_beta * local_losses
            else:
                losses = global_losses

            # scale if predict data
            if pred_data:
                norm = torch.sqrt(alpha_t / sigma_t)
                losses = expand_dims(norm, losses.dim()) * losses

            return losses.mean()

    return loss_fn


#################### Utils Functions ####################

@torch.no_grad()
def get_align_noise(z_t, xh, alpha_t, sigma_t, noise, node_mask):
    with profiling.record("align_noise/slice_positions"):
        pos_t = z_t[:, :, :3]
        pos_0 = xh[:, :, :3]

    rotations = kabsch_batch(pos_t, pos_0)  # [batch_size, 3, 3]
    with profiling.record("align_noise/einsum_rotate"):
        align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0)

    with profiling.record("align_noise/update_noise"):
        noise_pos = (pos_t - expand_dims(alpha_t, align_pos_0.dim()) * align_pos_0) / expand_dims(sigma_t, pos_t.dim())
        noise[:, :, :3] = noise_pos
    return noise


@ torch.no_grad()
def get_align_position(z_t, xh):
    with profiling.record("align_position/slice_positions"):
        pos_t = z_t[:, :, :3]
        pos_0 = xh[:, :, :3]

    rotations = kabsch_batch(pos_t, pos_0)  # [batch_size, 3, 3]
    with profiling.record("align_position/einsum_rotate"):
        align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0)

    return align_pos_0

@ torch.no_grad()
def get_align_position_v2(pos_t_com, pos_0_com):
    rotations = kabsch_batch(pos_t_com, pos_0_com)  # [batch_size, 3, 3]
    with profiling.record("align_position_v2/einsum_rotate"):
        align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0_com)

    return align_pos_0


@torch.no_grad()
def kabsch(coord_pred, coord_tar):
    with profiling.record("kabsch/covariance"):
        A = coord_pred.transpose(0, 1) @ coord_tar  # [3, 3]
    with profiling.record("kabsch/svd"):
        U, S, Vt = torch.linalg.svd(A)
    with profiling.record("kabsch/rotation"):
        corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=coord_pred.device))  # [3, 3]
        rotation = (U @ corr_mat) @ Vt
    return rotation


@torch.no_grad()
def kabsch_batch(coords_pred, coords_tar):
    """Batch version of Kabsch algorithm."""
    with profiling.record("kabsch_batch/total"):
        with profiling.record("kabsch_batch/covariance"):
            A = torch.einsum("...ki, ...kj -> ...ij", coords_pred, coords_tar)
        with profiling.record("kabsch_batch/svd"):
            U, S, Vt = torch.linalg.svd(A)
        with profiling.record("kabsch_batch/rotation"):
            sign_detA = torch.sign(torch.det(A))  # [batch_size]
            corr_mat_diag = torch.ones((A.size(0), U.size(-1)), device=A.device)  # [batch_size, 3]
            corr_mat_diag[:, -1] = sign_detA  # [batch_size, 3]
            corr_mat = torch.diag_embed(corr_mat_diag)  # [batch_size, 3, 3]
            rotation = torch.einsum("...ij, ...jk, ...kl -> ...il", U, corr_mat, Vt)  # [batch_size, 3, 3]

    return rotation


def process_batch(batch, device, include_charges, scaler):
    pos = batch['positions'].to(device)
    node_mask = batch['atom_mask'].to(device).unsqueeze(2)
    edge_mask = batch['edge_mask'].to(device)
    atom_type = batch['one_hot'].to(device)
    fc_charge = (batch['formal_charges'] if include_charges else torch.zeros(0)).to(device)

    # scaler
    pos = remove_mean_with_mask(pos, node_mask)
    pos, atom_type, fc_charge = scaler(pos, atom_type, fc_charge, node_mask)

    # pack data
    xh = torch.cat([pos, atom_type, fc_charge], dim=2)

    return xh, node_mask, edge_mask


def process_batch_2D(batch, device, include_charges, scaler):
    node_mask = batch['atom_mask'].to(device).unsqueeze(2)
    edge_mask = batch['edge_mask'].to(device)
    atom_type = batch['atom_one_hot'].to(device)
    edge_type = batch['edge_one_hot'].to(device)
    fc_charge = (batch['formal_charges'] if include_charges else torch.zeros(0)).to(device)
    pos = None

    # support context
    if 'context' in batch:
        context = move_context_to_device(batch['context'], device)
    else:
        context = None

    # scaler
    # pos = remove_mean_with_mask(pos, node_mask)
    _, atom_type, fc_charge, edge_type = scaler(pos, atom_type, fc_charge, node_mask, edge_type, edge_mask)

    # pack data
    xh = torch.cat([atom_type, fc_charge], dim=2)
    return xh, edge_type, node_mask, edge_mask, context


@torch.no_grad()
def process_edge_batch(batch, device, include_charges, scaler, prop_norm, model_name):
    pos = batch['positions'].to(device)                      # [128, 21, 3]
    node_mask = batch['atom_mask'].to(device).unsqueeze(2)   # [128, 21, 1]
    edge_mask = batch['edge_mask'].to(device)                # [56448, 1], actually is [128, 21, 21]
    atom_type = batch['atom_one_hot'].to(device)             # [128, 21, 5]
    edge_type = batch['edge_one_hot'].to(device)             # [128, 21, 21, 2]
    fc_charge = (batch['formal_charges'] if include_charges else torch.zeros(0)).to(device)  # [128, 21, 1]

    # support context
    if 'context' in batch:
        context = move_context_to_device(batch['context'], device)
    else:
        context = None

    # scaler
    # If it's a DMT model, need to normalize positions first; if it's a DMT_WO_EQ model, because translation data augmentation is used, no need to normalize positions
    if model_name == 'DMT':
        pos = remove_mean_with_mask(pos, node_mask)
    elif model_name == 'DMT_WO_EQ':
        pos = pos
    else:
        raise NotImplementedError(f'{model_name} not supported yet!')
    pos, atom_type, fc_charge, edge_type = scaler(pos, atom_type, fc_charge, node_mask, edge_type, edge_mask)

    # pack data
    xh = torch.cat([pos, atom_type, fc_charge], dim=2)  # [128, 21, 9]

    return xh, edge_type, node_mask, edge_mask, context
