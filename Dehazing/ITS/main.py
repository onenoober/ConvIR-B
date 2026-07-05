import os
import torch
import argparse
import random
import shutil
from torch.backends import cudnn
from models.ConvIR import build_bilfcf_net, build_net
from train import _train
from eval import _eval


BILFCF_ALLOWED_NEW_PREFIXES = ("BILFCF_",)


def _load_checkpoint_model(path, map_location):
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and 'model' in state:
        return state['model']
    return state


def load_haze4k_partial(model, checkpoint_path, allowed_new_prefixes):
    state = _load_checkpoint_model(checkpoint_path, 'cpu')
    model_state = model.state_dict()
    loaded = {}
    shape_mismatch = []
    unexpected = []

    for key, value in state.items():
        if key not in model_state:
            unexpected.append(key)
        elif model_state[key].shape != value.shape:
            shape_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))
        else:
            loaded[key] = value

    missing = [key for key in model_state if key not in loaded]
    bad_missing = [
        key for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_new_prefixes)
    ]
    if unexpected or shape_mismatch or bad_missing:
        raise RuntimeError(
            "partial-load failed: "
            f"unexpected={unexpected[:20]} shape_mismatch={shape_mismatch[:20]} "
            f"bad_missing={bad_missing[:20]}"
        )
    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)
    print(
        "INIT_MODEL_PARTIAL_LOAD "
        f"path={checkpoint_path} loaded={len(loaded)} "
        f"missing_new={len(missing)} unexpected=0 shape_mismatch=0 "
        f"allowed_new_prefixes={allowed_new_prefixes}"
    )


def load_init_model(model, args):
    if not args.init_model:
        return
    if args.resume:
        raise ValueError('--init_model initializes weights; --resume restores optimizer state. Use only one.')
    if args.arch == 'bilfcf_convir':
        load_haze4k_partial(model, args.init_model, BILFCF_ALLOWED_NEW_PREFIXES)
        return
    state = _load_checkpoint_model(args.init_model, 'cpu')
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
    if args.arch == 'bilfcf_convir':
        model = build_bilfcf_net(
            args.version,
            args.data,
            args.fam_mode,
            insertion=args.bilfcf_insertion,
            alpha_max=args.bilfcf_alpha_max,
            gate_bias=args.bilfcf_gate_bias,
            hidden_channels=args.bilfcf_hidden_channels,
            lowpass_kernel=args.bilfcf_lowpass_kernel,
        )
    else:
        model = build_net(args.version, args.data, args.fam_mode)
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
    parser.add_argument('--arch', default='official_convir', choices=['official_convir', 'convir', 'bilfcf_convir'], type=str)
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
    parser.add_argument('--init_model', type=str, default='')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--bilfcf_insertion', default='s5', choices=['s5'], type=str)
    parser.add_argument('--bilfcf_alpha_max', type=float, default=0.02)
    parser.add_argument('--bilfcf_gate_bias', type=float, default=-4.0)
    parser.add_argument('--bilfcf_hidden_channels', type=int, default=32)
    parser.add_argument('--bilfcf_lowpass_kernel', type=int, default=5)
    parser.add_argument('--bilfcf_train_scope', default='adapter_only', choices=['adapter_only', 'all'], type=str)
    parser.add_argument('--bilfcf_amplitude_loss_weight', type=float, default=0.01)


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
    # Backward-compatible alias for route scripts that used the misspelled name.
    args.leaning_rate = args.learning_rate
    args.model_save_dir = os.path.join('results/', args.model_name, 'Training-Results/')
    args.result_dir = os.path.join('results/', args.model_name, 'images', args.data)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    for source in ('models/layers.py', 'models/ConvIR.py', 'data/data_load.py', 'data/data_augment.py', 'train.py', 'valid.py', 'eval.py', 'main.py'):
        if os.path.exists(source):
            shutil.copy2(source, args.model_save_dir)
    print(args)
    main(args)
