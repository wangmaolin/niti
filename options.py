import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--seed', type=int, default=-1, metavar='S',
                    help='random seed (default: None)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-all', action='store_true', default=False,
                    help='Save all checkpoints along training')
parser.add_argument('--data-dir', type=str, default='./cifar10',
                    help='dataset dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--init',
                    help='init ckpt')
parser.add_argument('--depth', default=7, type=int,
                    help='vgg depth')
args = parser.parse_args()