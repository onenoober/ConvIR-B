import os
import torch
import argparse
import random
import shutil
import hashlib
from torch.backends import cudnn
from models.ConvIR import build_net
from train import _train
from eval import _eval


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
    if args.init_model_partial:
        allowed_prefixes = tuple(prefix for prefix in args.partial_new_prefixes.split(',') if prefix)
        result = model.load_state_dict(state, strict=False)
        missing = [key for key in result.missing_keys if not key.startswith(allowed_prefixes)]
        unexpected = list(result.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                f'partial init failed: missing={missing} unexpected={unexpected} '
                f'allowed_new_prefixes={allowed_prefixes}'
            )
        loaded = [key for key in state if key in model.state_dict()]
        print(
            f'INIT_MODEL_PARTIAL_LOAD path={args.init_model} loaded={len(loaded)} '
            f'missing_allowed={len(result.missing_keys)} unexpected=[] '
            f'allowed_new_prefixes={allowed_prefixes}'
        )
        return
    model.load_state_dict(state)
    print(f'INIT_MODEL_LOAD path={args.init_model} missing=[] unexpected=[]')


def main(args):
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
    model = build_net(
        args.version,
        args.data,
        args.fam_mode,
        arch=args.arch,
        dta_variant=args.dta_variant,
        dta_prior_channels=args.dta_prior_channels,
        dta_gate_bias=args.dta_gate_bias,
        dta_gate_limit=args.dta_gate_limit,
        dta_gamma_limit=args.dta_gamma_limit,
        dta_beta_limit=args.dta_beta_limit,
        dta_alpha_init=args.dta_alpha_init,
        dta_depth_mode=args.dta_depth_mode,
        dta_confidence_floor=args.dta_confidence_floor,
        dta_confidence_local_scale=args.dta_confidence_local_scale,
        dta_output_residual_scale=args.dta_output_residual_scale,
    )
    # print(model)

    if torch.cuda.is_available():
        model.cuda()
    load_init_model(model, args)
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
    parser.add_argument('--fam_mode', default='original', choices=['original'], type=str)
    parser.add_argument('--arch', default='official_convir', choices=['official_convir', 'convir', 'dta', 'dta_v2'], type=str)
    parser.add_argument('--seed', default=-1, type=int)

    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)
    parser.add_argument('--data_dir', type=str, default='')

    # Train for its
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', '--leaning_rate', dest='learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--stop_epoch', type=int, default=-1)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--mod_stats_freq', type=int, default=0)
    parser.add_argument('--mod_stats_batches', type=int, default=64)
    parser.add_argument('--grad_clip_norm', type=float, default=0.001)
    parser.add_argument('--dta_grad_clip_norm', type=float, default=-1.0)
    parser.add_argument('--dta_neighbor_grad_clip_norm', type=float, default=-1.0)
    parser.add_argument('--init_model', type=str, default='')
    parser.add_argument('--init_model_partial', action='store_true')
    parser.add_argument('--partial_new_prefixes', type=str, default='DTA.')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--train_scope', default='all', choices=['all', 'adapter_only', 'adapter_neighbors'], type=str)
    parser.add_argument('--dta_depth_cache_dir', type=str, default='')
    parser.add_argument('--dta_train_depth_split', type=str, default='train')
    parser.add_argument('--dta_eval_depth_split', type=str, default='test')
    parser.add_argument('--valid_root_split', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--eval_root_split', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--dta_require_depth', action='store_true')
    parser.add_argument('--dta_variant', type=str, default='v1', choices=['v1', 'v2'])
    parser.add_argument('--dta_depth_mode', type=str, default='normal', choices=['normal', 'invert', 'zero', 'shuffle'])
    parser.add_argument('--dta_prior_channels', type=int, default=16)
    parser.add_argument('--dta_gate_bias', type=float, default=-6.0)
    parser.add_argument('--dta_gate_limit', type=float, default=0.05)
    parser.add_argument('--dta_gamma_limit', type=float, default=0.10)
    parser.add_argument('--dta_beta_limit', type=float, default=0.05)
    parser.add_argument('--dta_alpha_init', type=float, default=1.0)
    parser.add_argument('--dta_confidence_floor', type=float, default=0.25)
    parser.add_argument('--dta_confidence_local_scale', type=float, default=6.0)
    parser.add_argument('--dta_output_residual_scale', type=float, default=0.03)
    parser.add_argument('--dta_use_trans_gt', action='store_true')
    parser.add_argument('--dta_trans_weight', type=float, default=0.0)
    parser.add_argument('--dta_phys_weight', type=float, default=0.0)
    parser.add_argument('--dta_preserve_weight', type=float, default=0.0)
    parser.add_argument('--dta_preserve_trans_thresh', type=float, default=0.80)
    parser.add_argument('--dta_gate_ramp_start', type=float, default=-1.0)
    parser.add_argument('--dta_gate_ramp_mid', type=float, default=-1.0)
    parser.add_argument('--dta_gate_ramp_end', type=float, default=-1.0)
    parser.add_argument('--dta_gate_ramp_warmup_epochs', type=int, default=2)
    parser.add_argument('--dta_gate_ramp_mid_epochs', type=int, default=8)
    parser.add_argument('--dta_rank_weight', type=float, default=0.005)
    parser.add_argument('--dta_tv_weight', type=float, default=0.0005)
    parser.add_argument('--dta_proxy_weight', type=float, default=0.0)
    parser.add_argument('--dta_rank_pairs', type=int, default=512)
    parser.add_argument('--dta_rank_min_depth_gap', type=float, default=0.03)
    parser.add_argument('--split_json', type=str, default='')
    parser.add_argument('--split_name', type=str, default='')


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
    if args.arch in ('dta', 'dta_v2') and args.init_model and not args.init_model_partial:
        raise ValueError('DTA fine-tuning from official Haze4K weights requires --init_model_partial.')
    # Backward-compatible alias for route scripts that used the misspelled name.
    args.leaning_rate = args.learning_rate
    args.model_save_dir = os.path.join('results/', args.model_name, 'Training-Results/')
    args.result_dir = os.path.join('results/', args.model_name, 'images', args.data)
    if args.init_model and os.path.isfile(args.init_model):
        with open(args.init_model, 'rb') as handle:
            args.init_model_sha256 = hashlib.sha256(handle.read()).hexdigest()
    else:
        args.init_model_sha256 = ''
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    for source in ('models/layers.py', 'models/ConvIR.py', 'data/data_load.py', 'data/data_augment.py', 'train.py', 'valid.py', 'eval.py', 'main.py'):
        if os.path.exists(source):
            shutil.copy2(source, args.model_save_dir)
    print(args)
    main(args)
