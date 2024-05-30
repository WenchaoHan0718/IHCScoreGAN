from IHCScoreGAN import IHCScoreGAN
import argparse, os

from utils import str2bool

def parse_args() -> argparse.Namespace:
    '''
    Parses command-line arguments.

    Returns:
        argparse.Namespace: object consisting of parsed command-line args and values.
    '''

    desc = "Pytorch implementation of IHCScoreGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--exp_name', type=str, default='bcdataset', help='The experiment name')
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')

    parser.add_argument('--iteration', type=int, default=40000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=4, help='The batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of workers for dataloader multiprocessing')
    parser.add_argument('--print_freq', type=int, default=1000, help='The image print freq')
    parser.add_argument('--save_freq', type=int, default=5000, help='The model save freq')

    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0002, help='The weight decay')
    parser.add_argument('--decay_flag', type=str2bool, default=False, help='The decay flag')

    parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN loss')
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight for Cycle loss')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight for Identity loss')

    parser.add_argument('--ch', type=int, default=64, help='The number of channels per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblocks')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layers')

    parser.add_argument('--img_size', type=int, default=256, help='The image size')
    parser.add_argument('--img_ch', type=int, default=3, help='The number of image channels')

    parser.add_argument('--result_dir', type=str, default=r'./carl/CPAI/results', help='Directory name to save the results')
    parser.add_argument('--input_dir', type=str, default=r'./carl/CPAI/dataset', help='Directory name for input images (must contain "trainA", "trainB", "testA", and "testB" subdirectories)')
    parser.add_argument('--load_path', type=str, default=r'./carl/CPAI/results/bcdataset/model/IHCScoreGAN_Weights_BCData_Latest.pt', help='The path to model weights to load from')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode [cpu, cuda]')
    parser.add_argument('--resume', type=str2bool, default=False)

    parser.add_argument('--neighborhood_size', type=int, default=25, help='The local neighborhood size')
    parser.add_argument('--kp_threshold', type=float, default=0.5, help='The confidence threshold for the center point prediction')
    parser.add_argument('--save_images', type=str2bool, default=True, help='Whether to save test images to disk')

    return parser.parse_args()

def main():
    '''
    Model entry point for train and test loop.
    '''
    
    args = parse_args()
    assert args.batch_size >= 1, '--batch_size must be at least one'

    # Ensure results paths exist
    os.makedirs(os.path.join(args.result_dir, args.exp_name, 'model'), exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, args.exp_name, 'img'), exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, args.exp_name, 'logs'), exist_ok=True)

    gan = IHCScoreGAN(args)
    gan.build_model()

    if args.phase == 'train':
        assert not (args.resume & (args.load_path is None)), 'Must set --load_path to a model weights file, since --phase is set to "train" and --resume flag is True'
        
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test':
        assert not args.load_path is None, 'Must set --load_path to a model weights file, since --phase is set to "test"'

        gan.test()
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()
