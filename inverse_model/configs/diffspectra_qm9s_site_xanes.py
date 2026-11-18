"""Training/evaluation config for absorber-site-conditioned generation on QM9."""

import ml_collections
import torch
import os


def get_config():
    config = ml_collections.ConfigDict()

    config.exp_type = 'diffspectra_xanes'
    config.pred_edge = True
    config.only_2D = False

    # data
    config.data = data = ml_collections.ConfigDict()
    data.root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    data.name = 'QM9S'
    # Use the site-conditioned broad XANES processed set. The previous
    # large_mol path was overwritten with a base QM9-style file that no longer
    # contains exs / per-site XANES fields.
    data.processed_file = 'data_qm9_xanes_site.pt'
    data.info_name = 'qm9_second_half'
    data.num_workers = 16

    data.compress_edge = True
    data.centered = True
    data.include_aromatic = False
    data.atom_types = 5
    data.bond_types = 4
    data.fc_scale = [-1., 1.]
    data.max_node = 29

    # spectra / site conditioning
    # Override these from CLI to switch edge.
    # ml_collections CLI cannot override list fields, so keep this as a tuple and
    # pass tuple syntax on the command line, e.g. "('xanes_CK',)".
    #   --config.data.used_spectra "('xanes_CK',)" --config.data.absorbing_edge xanes_CK --config.data.site_spectra_field xanes_CK
    #   --config.data.used_spectra "('xanes_NK',)" --config.data.absorbing_edge xanes_NK --config.data.site_spectra_field xanes_NK
    #   --config.data.used_spectra "('xanes_OK',)" --config.data.absorbing_edge xanes_OK --config.data.site_spectra_field xanes_OK
    data.spectra_version = 'allspectra'
    data.used_spectra = ('xanes_OK',)
    data.use_site_spectra = True
    data.expand_site_samples = True
    data.absorbing_edge = 'xanes_OK'
    data.site_spectra_field = 'xanes_OK'
    data.exs_field = 'exs'
    data.aug_translation_scale = 0.1
    data.transform = 'EdgeComSpectra'
    data.use_normalize = True

    # SDE
    config.sde = sde = ml_collections.ConfigDict()
    sde.schedule = 'cosine'
    sde.continuous_beta_0 = 0.1
    sde.continuous_beta_1 = 20.

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'DMT'
    model.pred_data = True
    model.include_fc_charge = True
    model.normalize_factors = '1, 4, 4, 1'
    model.ema_decay = 0.999
    model.edge_ch = 2
    model.nf = 256
    model.n_layers = 6
    model.n_heads = 16
    model.dropout = 0.1
    model.cond_time = True
    model.dist_gbf = True
    model.gbf_name = 'CondGaussianLayer'
    model.self_cond = True
    model.self_cond_type = 'ori'
    model.edge_quan_th = 0.
    model.n_extra_heads = 2
    model.CoM = True
    model.mlp_ratio = 2
    model.spatial_cut_off = 2.
    model.softmax_inf = True
    model.trans_name = 'TransMixLayer'
    model.cond_ch = 1
    model.pretrained_specformer_path = ''
    model.patch_len = [20, 50, 50, 10, 10, 10]
    model.stride = [10, 25, 25, 5, 5, 5]
    model.loss_weights = '1., 0.25, 0.1'
    model.noise_align = True
    model.local_loss_beta = 1.0
    model.local_loss_hops = 2

    # training
    config.training = training = ml_collections.ConfigDict()
    training.dataloader_drop_last = False
    training.distributed = True
    training.num_gpus = torch.cuda.device_count()
    effective_gpus = max(training.num_gpus, 1)
    if training.num_gpus > 1:
        training.dataloader_drop_last = True
    print('number of gpus:', training.num_gpus)
    training.world_size = int(os.environ.get('WORLD_SIZE', effective_gpus))
    training.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    base_batch_size = 256
    training.batch_size = base_batch_size * effective_gpus
    training.eval_batch_size = base_batch_size * effective_gpus
    training.eval_samples = base_batch_size * effective_gpus

    if training.distributed and training.num_gpus > 0:
        config.device = torch.device(f'cuda:{training.local_rank}')
    else:
        config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    training.reduce_mean = False
    training.log_freq = 500 // effective_gpus
    training.n_iters = 2000000 // effective_gpus
    training.snapshot_freq = 50000 // effective_gpus
    training.snapshot_freq_for_preemption = 10000 // effective_gpus
    training.snapshot_sampling = False

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'AdamW'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 100000
    optim.grad_clip = 10.
    optim.disable_grad_log = True

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'ancestral'
    sampling.steps = 1000
    sampling.vis_row = 4
    sampling.vis_col = 4

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.enable_sampling = True
    evaluate.batch_size = base_batch_size * effective_gpus
    evaluate.num_samples = 10000
    evaluate.begin_ckpt = 40
    evaluate.end_ckpt = 40
    evaluate.ckpts = ''
    evaluate.sub_geometry = True
    evaluate.save_mols = 'false'
    evaluate.sampling_temperature = 1.0

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config.sampling_nums = 1

    config.profiling = profiling = ml_collections.ConfigDict()
    profiling.enabled = False
    profiling.log_freq = 100
    profiling.sync_cuda = False
    profiling.reset_after_log = False
    profiling.top_k = 50

    return config
