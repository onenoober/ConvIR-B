import os
import torch
import argparse
import random
import shutil
from torch.backends import cudnn
from models.ConvIR import build_net as build_convir_net
from models.PFDConvIR import build_pfd_net
from train import _train
from eval import _eval


def build_model(args):
    if args.arch == 'convir':
        return build_convir_net(args.version, args.data, args.fam_mode)
    if args.fam_mode != 'original':
        raise ValueError('--fam_mode must stay original when --arch pfd is used.')
    return build_pfd_net(
        args.version,
        args.data,
        pfd_rhfd=args.pfd_rhfd,
        pfd_hscm=args.pfd_hscm,
        pfd_pffb=args.pfd_pffb,
        pfd_pffb_high=args.pfd_pffb_high,
        pfd_teacher=args.pfd_teacher,
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
    bad_missing = [key for key in missing if not key.startswith('PFD_')]
    if unexpected or bad_missing:
        raise RuntimeError(
            'Unexpected --init_model load result: '
            f'missing={missing}, unexpected={unexpected}'
        )
    print(f'INIT_MODEL_LOAD path={args.init_model} missing={missing} unexpected={unexpected}')

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
    model = build_model(args)
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
    parser.add_argument('--fam_mode', default='original', choices=['original', 'modres', 'fam2_modres'], type=str)
    parser.add_argument('--arch', default='convir', choices=['convir', 'pfd'], type=str)
    parser.add_argument('--pfd_rhfd', default=0, choices=[0, 1], type=int)
    parser.add_argument('--pfd_hscm', default=0, choices=[0, 1], type=int)
    parser.add_argument('--pfd_pffb', default=0, choices=[0, 1], type=int)
    parser.add_argument('--pfd_pffb_high', default=0, choices=[0, 1], type=int)
    parser.add_argument('--pfd_teacher', default=0, choices=[0, 1], type=int)
    parser.add_argument('--seed', default=-1, type=int)

    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)
    parser.add_argument('--data_dir', type=str, default='')

    # Train for its
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
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
        'models/PFDConvIR.py',
        'models/pfd_modules.py',
        'models/haze_priors.py',
        'train.py',
        'main.py',
    ):
        if os.path.exists(source):
            shutil.copy2(source, args.model_save_dir)
    print(args)
    main(args)
