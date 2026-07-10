import os
import torch
import argparse
import random
import shutil
from torch.backends import cudnn
from models.ConvIR import build_net
from train import _train
from eval import _eval
from d7c_gate import build_d7c_gate_producer, load_checkpoint_state, partial_load_model_state


def _load_checkpoint_model(path, map_location):
    return load_checkpoint_state(path, map_location)


def load_init_model(model, args):
    if not args.init_model:
        return
    if args.resume:
        raise ValueError('--init_model initializes weights; --resume restores optimizer state. Use only one.')
    state = _load_checkpoint_model(args.init_model, 'cpu')
    if args.allow_fam2_partial_init:
        if args.fam_mode not in ('fam2_modres', 'fam2_d7c_noop'):
            raise ValueError('--allow_fam2_partial_init requires a FAM2 modulation mode.')
        result = partial_load_model_state(
            model,
            state,
            {'FAM2.modulator.weight', 'FAM2.modulator.bias'},
        )
        print(
            'INIT_MODEL_PARTIAL_LOAD path=%s missing=%s unexpected=%s shape_mismatch=%s'
            % (
                args.init_model,
                result['missing_candidate_keys'],
                result['unexpected_keys'],
                result['shape_mismatch'],
            )
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
    model = build_net(args.version, args.data, args.fam_mode)
    # print(model)

    if torch.cuda.is_available():
        model.cuda()
    load_init_model(model, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.d7c_gate_producer = build_d7c_gate_producer(args, device)
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
    parser.add_argument('--fam_mode', default='original', choices=['original', 'fam2_modres', 'fam2_d7c_noop'], type=str)
    parser.add_argument('--arch', default='official_convir', choices=['official_convir', 'convir'], type=str)
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--d7c_gate_mode', default='none', choices=['none', 'd7c_fixed'], type=str)
    parser.add_argument('--d7c_base_checkpoint', type=str, default='')
    parser.add_argument('--d7c_density_artifact', type=str, default='')
    parser.add_argument('--d7c_need_artifact', type=str, default='')
    parser.add_argument('--d7c_threshold', type=float, default=0.5773006677627563)
    parser.add_argument('--allow_fam2_partial_init', action='store_true')

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
    if args.arch not in ('official_convir', 'convir'):
        raise ValueError('Official anchor only supports the official ConvIR-B architecture.')
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
