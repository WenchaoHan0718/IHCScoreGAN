from UGATIT import UGATIT
from CycleGAN import CycleGAN
from KeypointGAN import KeypointGAN
from KeypointGAN_v2 import KeypointGAN_v2
import argparse
from utils import *

"""parsing and configuration"""

def parse_args():
    desc = "Pytorch implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='test', help='[train / test]')
    parser.add_argument('--light', type=str2bool, default=True, help='[U-GAT-IT full version / U-GAT-IT light version]')
    parser.add_argument('--dataset', type=str, default='ihc2maskvar_before2018', help='dataset_name')
    parser.add_argument('--load_iteration', type=str, default='40000', help='The label of weights file to load')

    parser.add_argument('--iteration', type=int, default=40000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=5000, help='The number of model save freq')
    parser.add_argument('--decay_flag', type=str2bool, default=False, help='The decay_flag')

    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0002, help='The weight decay')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN')
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight for Cycle')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight for Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight for CAM')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)

    parser.add_argument('--neighborhood_size', type=int, default=25, help='')
    parser.add_argument('--kp_threshold', type=float, default=0.5, help='')
    parser.add_argument('--n_samples', type=int, default=None, help='')
    parser.add_argument('--save_images', type=str2bool, default=True, help='')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --result_dir
    # check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    # check_folder(os.path.join(args.result_dir, args.dataset, 'img'))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()
      
    # for year in ['2017', '2014', '2016', '2018', '2019']:
    #     args.year = year
    #     args.dataset = 'ihc2maskvar_' + year
      
    # for phase in ['train', 'test']:
    #     args.phase = phase
      
    # for n_samples in [10000, 15000, 20000, 25000, 30000, 35000, 40000]:
    #     args.dataset = 'ihc2maskvar_before2018_N' + str(n_samples)
    #     args.n_samples = n_samples

    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    gan = KeypointGAN(args)

    # build graph
    gan.build_model()

    if args.phase == 'train' :
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test' :
        gan.test()
        print(" [*] Test finished!")

    if args.phase == 'select' :
        gan.select()
        print(" [*] Selection finished!")

if __name__ == '__main__':
    main()
