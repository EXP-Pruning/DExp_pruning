
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse

from data import imagenet
from models import *
from mask import *
import utils


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Evaluate')
parser.add_argument(
    '--data_dir',
    type=str,
    default='./data',
    help='dataset path')
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=('cifar10','cifar100','imagenet'),
    help='dataset')
parser.add_argument(
    '--arch',
    type=str,
    default='googlenet',
    choices=('resnet_50','vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet'),
    help='The architecture to prune')
parser.add_argument(
    '--test_model_dir',
    type=str,
    default='./result/tmp/pruned_checkpoint/googlenet_cov10.pt',
    help='The directory where the summaries will be stored.')
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=512,
    help='Batch size for validation.')
parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
cudnn.benchmark = True
cudnn.enabled = True


if args.dataset=='cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
elif args.dataset=='cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
elif args.dataset=='imagenet':
    data_tmp = imagenet.Data(args)
    trainloader = data_tmp.loader_train
    trainloader_1 = data_tmp.loader_train_1
    testloader = data_tmp.loader_test
else:
    assert 1==0

print('==> Building model..')
net = eval(args.arch)(args.dataset,compress_rate=[0.]*200)
net = net.cuda()
print(net)

if len(args.gpu)>1 and torch.cuda.is_available():
    device_id = []
    for i in range((len(args.gpu) + 1) // 2):
        device_id.append(i)
    net = torch.nn.DataParallel(net, device_ids=device_id)

def test():
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    net.eval()
    num_iterations = len(testloader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)

            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            # if batch_idx%print_freq==0:
            #     print(
            #         '({0}/{1}): '
            #         'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
            #             batch_idx, num_iterations, top1=top1, top5=top5))

        print("Final Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}".format(top1=top1, top5=top5))


if len(args.gpu)>1:
    convcfg = net.module.covcfg
else:
    convcfg = net.covcfg

cov_id=len(convcfg)
new_state_dict = OrderedDict()
pruned_checkpoint = torch.load(args.test_model_dir ,map_location='cuda:0')
tmp_ckpt = pruned_checkpoint['state_dict']
if len(args.gpu) == 1:
    for k, v in tmp_ckpt.items():
        new_state_dict[k.replace('module.', '')] = v
else:
    for k, v in tmp_ckpt.items():
        new_state_dict['module.' + k.replace('module.', '')] = v
net.load_state_dict(new_state_dict)
# cpra = []
# params = net.parameters()
# params = list(params)
# for index, item in enumerate(params):
#     if (index)%3==0:
#         f, c, w, h = item.size()
#         cnt_array = np.sum(item.cpu().detach().numpy() == 0)
#         cpra.append(format(cnt_array / (f * c * w * h), '.2g'))
#     if index == 326:
#         break
# print('compress_rate:{}'.format(cpra))
test()

