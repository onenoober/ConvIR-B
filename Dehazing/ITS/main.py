import os
import torch
import argparse
import random
import shutil
from torch.backends import cudnn
from models.APDRConvIR import build_apdr_net
from models.ConvIR import build_net as build_convir_net
from models.DPGAConvIR import build_dpga_net
from train import _train
from eval import _eval


def build_model(args):
    if args.arch == 'convir':
        return build_convir_net(args.version, args.data, args.fam_mode)
    if args.arch == 'dpga':
        if args.fam_mode != 'original':
            raise ValueError('--fam_mode must stay original when --arch dpga is used.')
        return build_dpga_net(
            args.version,
            args.data,
            prior_embed_channels=args.dpga_prior_embed_channels,
            adapter_reduction=args.dpga_adapter_reduction,
            adapter_residual_scale=args.dpga_adapter_residual_scale,
            adapter_scale_init=args.dpga_adapter_scale_init,
            adapter_bootstrap_scale=args.dpga_adapter_bootstrap_scale,
            dark_patch=args.dpga_dark_patch,
            local_patch=args.dpga_local_patch,
            active_adapters=args.dpga_active_adapters,
            scale_multiplier=args.dpga_scale_multiplier,
        )
    if args.fam_mode != 'original':
        raise ValueError('--fam_mode must stay original when --arch apdr is used.')
    return build_apdr_net(
        args.version,
        args.data,
        apdr_prior_mode=args.apdr_prior_mode,
        apdr_residual_max=args.apdr_residual_max,
        apdr_gate_max=args.apdr_gate_max,
        apdr_gate_init=args.apdr_gate_init,
        apdr_force_zero_gate=bool(args.apdr_force_zero_gate),
        apdr_active_scales=args.apdr_active_scales,
        apdr_selector_mode=args.apdr_selector_mode,
        apdr_residual_capacity=args.apdr_residual_capacity,
    )


def _load_checkpoint_model(path, map_location):
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and 'model' in state:
        return state['model']
    return state


def load_init_model(model, args):
    if not args.init_model:
        return
    if args.resume:
        raise ValueError('--init_model initializes weights; --resume restores optimizer state. Use only one.')
    state = _load_checkpoint_model(args.init_model, 'cpu')
    if args.arch == 'convir':
        model.load_state_dict(state)
        print(f'INIT_MODEL_LOAD path={args.init_model} missing=[] unexpected=[]')
        return
    result = model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if args.arch == 'apdr':
        allowed_prefixes = ('APDR_',)
    elif args.arch == 'dpga':
        allowed_prefixes = ('DPGA_',)
    else:
        allowed_prefixes = ()
    bad_missing = [key for key in missing if not key.startswith(allowed_prefixes)]
    if unexpected or bad_missing:
        raise RuntimeError(
            'Unexpected --init_model load result: '
            f'missing={missing}, unexpected={unexpected}'
        )
    print(f'INIT_MODEL_LOAD path={args.init_model} missing={missing} unexpected={unexpected}')


def apply_apdr_budget_args(model, args):
    if args.arch != 'apdr' or args.apdr_global_budget_tau is None:
        return
    if args.apdr_global_budget_temperature is None:
        raise ValueError('--apdr_global_budget_temperature is required when --apdr_global_budget_tau is set.')
    if not hasattr(model, 'active_apdr_prefixes'):
        return
    for prefix in model.active_apdr_prefixes():
        module = getattr(model, prefix)
        if hasattr(module, 'set_global_budget_calibration'):
            module.set_global_budget_calibration(
                args.apdr_global_budget_tau,
                args.apdr_global_budget_temperature,
                args.apdr_global_budget_power,
            )
    print(
        'APDR_BUDGET_CALIBRATION '
        f'tau={args.apdr_global_budget_tau} '
        f'temperature={args.apdr_global_budget_temperature} '
        f'power={args.apdr_global_budget_power}'
    )


def main(args):
    if (
        args.arch == 'apdr'
        and args.apdr_loss_scales == 'full_only'
        and args.apdr_active_scales != 'full'
    ):
        raise ValueError('--apdr_loss_scales full_only requires --apdr_active_scales full')

    # CUDNN
    if args.seed >= 0:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    model = build_model(args)
    # print(model)

    if torch.cuda.is_available():
        model.cuda()
    load_init_model(model, args)
    apply_apdr_budget_args(model, args)
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='ConvIR', type=str)
    parser.add_argument('--data', type=str, default='ITS', choices=['ITS', 'Haze4K', 'NHR', 'GTA5', 'real_haze'])
    parser.add_argument('--version', default='small', choices=['small', 'base', 'large'], type=str)
    parser.add_argument('--fam_mode', default='original', choices=['original', 'modres', 'fam2_modres'], type=str)
    parser.add_argument('--arch', default='convir', choices=['convir', 'apdr', 'dpga'], type=str)
    parser.add_argument('--apdr_prior_mode', default='rgb_haze', choices=['rgb_haze'], type=str)
    parser.add_argument('--apdr_residual_max', default=0.04, type=float)
    parser.add_argument('--apdr_gate_max', default=0.5, type=float)
    parser.add_argument('--apdr_gate_init', default=0.02, type=float)
    parser.add_argument('--apdr_force_zero_gate', default=0, choices=[0, 1], type=int)
    parser.add_argument('--apdr_selector_mode', default='v0', choices=['v0', 'v0_2', 'v0_2r'], type=str)
    parser.add_argument('--apdr_residual_capacity', default='linear', choices=['linear', 'shallow_mlp'], type=str)
    parser.add_argument('--apdr_global_budget_tau', default=None, type=float)
    parser.add_argument('--apdr_global_budget_temperature', default=None, type=float)
    parser.add_argument('--apdr_global_budget_power', default=1.0, type=float)
    parser.add_argument(
        '--apdr_active_scales',
        default='all',
        choices=['all', 'full'],
        type=str,
    )
    parser.add_argument(
        '--apdr_train_scope',
        default='all',
        choices=['all', 'apdr_only', 'apdr_residual_only'],
        type=str,
    )
    parser.add_argument('--apdr_anchor_lambda', default=0.0, type=float)
    parser.add_argument('--apdr_gate_lambda', default=0.0, type=float)
    parser.add_argument('--apdr_residual_lambda', default=0.0, type=float)
    parser.add_argument('--apdr_delta_lambda', default=0.0, type=float)
    parser.add_argument('--apdr_gate_supervision_lambda', default=0.0, type=float)
    parser.add_argument('--apdr_risk_temperature', default=5.0, type=float)
    parser.add_argument(
        '--apdr_loss_scales',
        default='all',
        choices=['all', 'full_only'],
        type=str,
    )
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--dpga_depth_cache_dir', default='', type=str)
    parser.add_argument('--dpga_train_depth_split', default='train', type=str)
    parser.add_argument('--dpga_eval_depth_split', default='test', type=str)
    parser.add_argument('--dpga_train_split_json', default='', type=str)
    parser.add_argument('--dpga_train_split_name', default='', type=str)
    parser.add_argument('--dpga_valid_split_json', default='', type=str)
    parser.add_argument('--dpga_valid_split_name', default='', type=str)
    parser.add_argument(
        '--dpga_train_scope',
        default='adapter_only',
        choices=['all', 'adapter_only'],
        type=str,
    )
    parser.add_argument('--dpga_prior_embed_channels', default=16, type=int)
    parser.add_argument('--dpga_adapter_reduction', default=2, type=int)
    parser.add_argument('--dpga_adapter_residual_scale', default=0.1, type=float)
    parser.add_argument('--dpga_adapter_scale_init', default=0.0, type=float)
    parser.add_argument('--dpga_adapter_bootstrap_scale', default=0.01, type=float)
    parser.add_argument('--dpga_dark_patch', default=15, type=int)
    parser.add_argument('--dpga_local_patch', default=31, type=int)
    parser.add_argument('--dpga_active_adapters', default='all', type=str)
    parser.add_argument('--dpga_scale_multiplier', default=1.0, type=float)
    parser.add_argument(
        '--dpga_tc_rec_loss',
        default='l1',
        choices=['l1', 'charbonnier'],
        type=str,
    )
    parser.add_argument('--dpga_tc_fft_lambda', default=0.1, type=float)
    parser.add_argument('--dpga_tc_anchor_lambda', default=0.0, type=float)
    parser.add_argument('--dpga_tc_chroma_lambda', default=0.0, type=float)
    parser.add_argument('--dpga_tc_delta_lambda', default=0.0, type=float)
    parser.add_argument('--dpga_tc_delta_tv_lambda', default=0.0, type=float)
    parser.add_argument('--dpga_tc_anchor_error_threshold', default=0.035, type=float)

    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)
    parser.add_argument('--data_dir', type=str, default='')

    # Train for its
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--grad_clip_norm', type=float, default=0.001)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--stop_epoch', type=int, default=-1)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--mod_stats_freq', type=int, default=0)
    parser.add_argument('--mod_stats_batches', type=int, default=64)
    parser.add_argument('--init_model', type=str, default='')
    parser.add_argument('--resume', type=str, default='')


    # uncomment for different datasets

    # Train for real-haze
    # parser.add_argument('--batch_size', type=int, default=2)
    # parser.add_argument('--learning_rate', type=float, default=2e-4)
    # parser.add_argument('--weight_decay', type=float, default=0)
    # parser.add_argument('--num_epoch', type=int, default=5000)
    # parser.add_argument('--print_freq', type=int, default=20)
    # parser.add_argument('--num_worker', type=int, default=4)
    # parser.add_argument('--save_freq', type=int, default=10)
    # parser.add_argument('--valid_freq', type=int, default=10)

    # Train for Haze4k
    # parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument('--learning_rate', type=float, default=4e-4)
    # parser.add_argument('--weight_decay', type=float, default=0)
    # parser.add_argument('--num_epoch', type=int, default=1000)
    # parser.add_argument('--print_freq', type=int, default=100)
    # parser.add_argument('--num_worker', type=int, default=8)
    # parser.add_argument('--save_freq', type=int, default=20)
    # parser.add_argument('--valid_freq', type=int, default=20)

    # Test
    parser.add_argument('--test_model', type=str, default='')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.model_name, 'Training-Results/')
    args.result_dir = os.path.join('results/', args.model_name, 'images', args.data)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    for source in (
        'models/layers.py',
        'models/ConvIR.py',
        'models/APDRConvIR.py',
        'models/apdr_modules.py',
        'models/DPGAConvIR.py',
        'data/data_load.py',
        'data/data_augment.py',
        'train.py',
        'valid.py',
        'eval.py',
        'main.py',
    ):
        if os.path.exists(source):
            shutil.copy2(source, args.model_save_dir)
    print(args)
    main(args)
