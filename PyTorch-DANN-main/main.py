import argparse
import datetime
import os
import random
import torch
import numpy as np
from torch.backends import cudnn
from solver import Solver


def main(args):
    if args.debug == True:
        torch.autograd.set_detect_anomaly(True)
        torch.cuda.set_device(0)
        torch.autograd.profiler.emit_nvtx()
    
    os.makedirs(args.model_path, exist_ok=True)
    solver = Solver(args)
    if args.mode=='train':
        print('hai sbagliato qualcosa, piciù')
        if args.method == 'src':
            solver.src()
        elif args.method == 'dann':
            solver.dann()

        solver.test()
    elif args.mode=='test':
        print('testing model')
        solver.test()
    


def update_args(args):
    
    args.channels = 3
    args.num_classes = 10
    args.cm = True
    

    if args.dset == 's2m':
        args.source = 'svhn'
        args.target = 'mnist'
        args.adapt_epochs = 1000
        
    elif args.dset == 'm2s':
        args.source = 'mnist'
        args.target = 'svhn'
        args.adapt_epochs = 1000

    elif args.dset == 'u2m':
        args.source = 'usps'
        args.target = 'mnist'
        args.channels = 1
        args.adapt_epochs = 1000  # Due to small size of USPS

    elif args.dset == 'm2u':
        args.source = 'mnist'
        args.target = 'usps'
        args.channels = 1
        args.adapt_epochs = 1000  # Due to small size of USPS

    elif args.dset == 'm2mm':
        args.source = 'mnist'
        args.target = 'mnistm'

    elif args.dset == 'sd2sv':
        args.source = 'sydigits'
        args.target = 'svhn'

    elif args.dset == 'signs':
        args.source = 'sysigns'
        args.target = 'gtsrb'
        args.num_classes = 43
        args.cm = False

    else:
        assert "Incorrect combination"
    if args.mode=='train':
        args.model_path = os.path.join(args.model_path, args.dset)
    print(args.model_path)
    args.adapt_test_epoch = args.adapt_epochs // 10

    return args


def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DANN')
    parser.add_argument('--p_thresh', type=float, default=0.9)
    parser.add_argument('--method', type=str, default='src', choices=['src', 'dann'])

    parser.add_argument('--src_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--log_step', type=int, default=50)

    parser.add_argument('--dset', type=str, default='s2m', choices=['s2m', 'm2s', 'u2m', 'm2u', 'm2mm', 'sd2sv', 'signs'])
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--seed', type=int, default=100)    
    parser.add_argument('--num_branches', type=int, default=10)
    parser.add_argument('--mode', type=str, default='test', choices=['train','test'])
    parser.add_argument('--ortho_scaling', type=float, default=0.5)
    parser.add_argument('--adapt_epochs', type=int, default=350)
    parser.add_argument('--debug', type=bool, default=False)

    args = parser.parse_args()
    args = update_args(args)

    manual_seed = args.seed
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    os.environ['PYTHONHASHSEED'] = str(manual_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

    main(args)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))
