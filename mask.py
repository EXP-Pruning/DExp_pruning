import torch
import torchvision.transforms as transforms  # 可对数据图像做处理
import numpy as np
import pickle
from torch import nn
import random

class CNN(nn.Module):
    def __init__(self,c, k, stride=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, c, kernel_size=k, stride=stride, padding=1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        return x1

feature_result = torch.tensor(0.)
conv_result = torch.tensor(0.)
weight = torch.tensor(0.)
input_1 = torch.tensor(0.)

def self_conv():
    global conv_result
    global weight
    global input_1

    input = input_1[0, 0, :, :].to('cuda:0')
    input = torch.unsqueeze(input, axis=0)
    input = torch.unsqueeze(input, axis=0)
    mod = CNN(c=weight.shape[0]*weight.shape[1],k=weight.shape[3], stride=1)
    W = input.shape[2] - weight.shape[3] + 3
    wei = torch.zeros(weight.shape[0]*weight.shape[1],weight.shape[3],weight.shape[3],device='cuda:0')
    conv_result = torch.zeros(weight.shape[0], weight.shape[1], W, W, device="cuda:0")
    for k in range(weight.shape[0]) :
        wei[k*weight.shape[1] : (k+1)*weight.shape[1]] = weight[k].to('cuda:0')
    wei = torch.unsqueeze(wei,axis=1)
    mod.conv1.weight.data = wei
    conv_result = mod(input)
    conv_result = conv_result.view(weight.shape[0]*weight.shape[1],1,W,W)
    conv_result = conv_result.view(weight.shape[0],weight.shape[1],W,W)

def get_feature():
    global feature_result
    global conv_result
    global input_1

    feature_result = torch.tensor(0.)
    a = conv_result.shape[0]
    b = conv_result.shape[1]
    input_1 = input_1[0, 0, :, :]
    pre_exp = torch.mean(input_1)
    end_exp = torch.tensor([torch.mean(conv_result[i][j]) for i in range(a) for j in range(b)])
    end_exp = end_exp.view(a, -1)
    res = torch.zeros(a, b)
    for i in range(a):
        for j in range(b):
            res[i][j] = end_exp[i][j] / (pre_exp + 0.00001)
    feature_result = res.to('cuda:0')


class mask_vgg_16_bn:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='', device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.cpra = []
        self.job_dir = job_dir
        self.device = device

    def layer_mask(self, epoch, cov_id, trainloader_1, resume=None, param_per_cov=4, arch="vgg_16_bn"):
        params = self.model.parameters()
        params = list(params)
        global feature_result
        global conv_result
        global weight
        global input_1
        for index, item in enumerate(params):
            if index == (cov_id - 1) * param_per_cov:
                weight = item
                break

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume = self.job_dir + '/mask'

        self.param_per_cov = param_per_cov

        if epoch == 0:
            for batch_idx, (inputs, targets) in enumerate(trainloader_1):
                if batch_idx >= 1:
                    break
                input_1 = inputs
            resume_input = self.job_dir + '/input'
            with open(resume_input, "wb") as f:
                pickle.dump(input_1, f)
        else:
            with open(self.job_dir + '/input', 'rb') as f:
                input_1 = pickle.load(f)

        self_conv()
        get_feature()

        for index, item in enumerate(params):
            if epoch == 0:
                resume_deta = self.job_dir + '/feature_result'
                with open(resume_deta, "wb") as f:
                    pickle.dump(feature_result, f)
                break

            if index == cov_id * param_per_cov:
                break

            if index == (cov_id - 1) * param_per_cov:

                with open(self.job_dir + '/feature_result', 'rb') as f:
                    feature_result_old = pickle.load(f)

                f, c, w, h = item.size()
                bn = torch.as_tensor(params[index + 2])
                bn = bn.detach().cpu().numpy()
                pruned_num = int(self.compress_rate[cov_id - 1] * f * c)
                feature_result_sign = torch.sign(feature_result_old)
                feature_result = feature_result - feature_result_old
                ind = np.argsort(abs(bn))[:]
                #  需要观察期望因子的变化大小，以及删除的通道个数
                ones_i = torch.ones(f, c).to(self.device)
                for i in range(len(ind)):
                    if pruned_num == 0:
                        break
                    if bn[ind[i]] > 0:
                        for j in range(len(feature_result[i])):
                            if feature_result_sign[i][j] < 0 and feature_result[i][j] > 0:  # -0.05越小 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                            elif feature_result_sign[i][j] > 0 and feature_result[i][j] < 0:  # 0.05越大 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                    else:
                        for j in range(len(feature_result[i])):
                            if feature_result_sign[i][j] < 0 and feature_result[i][j] < 0:  # 0.05越大 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                            elif feature_result_sign[i][j] > 0 and feature_result[i][j] > 0:  # -0.05越小 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break

                cnt_array = np.sum(ones_i.cpu().numpy() == 0)
                self.cpra.append(format(cnt_array / (f * c), '.2g'))
                ones = torch.ones(f, c, w, h).to(self.device)
                for i in range(f):
                    for j in range(c):
                        for k in range(w):
                            for l in range(h):
                                ones[i, j, k, l] = ones_i[i, j]
                self.mask[index] = ones
                item.data = item.data * self.mask[index]

                with open(resume, "wb") as f:
                    pickle.dump(self.mask, f)
                break
        # for index, item in enumerate(params):
        #
        #     if index == cov_id * param_per_cov:
        #         break
        #
        #     if index == (cov_id - 1) * param_per_cov:
        #
        #         f, c, w, h = item.size()
        #         pruned_num = int(self.compress_rate[cov_id - 1] * f * c)
        #         feature_result = abs(feature_result - 1.0)
        #         feare = np.array(feature_result.cpu()).reshape(f*c)
        #         ones_i = torch.ones(f, c).to(self.device)
        #
        #         ind = np.argsort(abs(feare))[:]
        #         mid = feare[ind[pruned_num]]
        #         ones_i[feature_result <= mid] =0
        #
        #         cnt_array = np.sum(ones_i.cpu().numpy() == 0)
        #         self.cpra.append(format(cnt_array / (f * c), '.2g'))
        #         ones = torch.ones(f, c, w, h).to(self.device)
        #         for i in range(f):
        #             for j in range(c):
        #                 for k in range(w):
        #                     for l in range(h):
        #                         ones[i, j, k, l] = ones_i[i, j]
        #         self.mask[index] = ones
        #         item.data = item.data * self.mask[index]
        #
        #         with open(resume, "wb") as f:
        #             pickle.dump(self.mask, f)
        #         break
    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            if index in range(0, 48, self.param_per_cov):
                item.data = item.data * self.mask[index].to(self.device)

    def pri_comp(self):
        print('compress_rate:{}'.format(self.cpra))

class mask_resnet_56:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='', device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.cpra = []
        self.job_dir = job_dir
        self.device = device

    def layer_mask(self, epoch, cov_id, trainloader_1, resume=None, param_per_cov=3, arch="resnet_56"):
        params = self.model.parameters()
        params = list(params)
        global feature_result
        global conv_result
        global weight
        global input_1
        for index, item in enumerate(params):
            if index == (cov_id - 1) * param_per_cov:
                weight = item
                break

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume = self.job_dir + '/mask'

        self.param_per_cov = param_per_cov

        if epoch == 0:
            for batch_idx, (inputs, targets) in enumerate(trainloader_1):
                if batch_idx >= 1:
                    break
                input_1 = inputs
            resume_input = self.job_dir + '/input'
            with open(resume_input, "wb") as f:
                pickle.dump(input_1, f)
        else:
            with open(self.job_dir + '/input', 'rb') as f:
                input_1 = pickle.load(f)

        self_conv()
        get_feature()

        for index, item in enumerate(params):
            if epoch == 0:
                resume_deta = self.job_dir + '/feature_result'
                with open(resume_deta, "wb") as f:
                    pickle.dump(feature_result, f)
                break

            if index == cov_id * param_per_cov:
                break

            if index == (cov_id - 1) * param_per_cov:

                with open(self.job_dir + '/feature_result', 'rb') as f:
                    feature_result_old = pickle.load(f)

                f, c, w, h = item.size()
                bn = torch.as_tensor(params[index + 1])
                bn = bn.detach().cpu().numpy()
                pruned_num = int(self.compress_rate[cov_id - 1] * f * c)
                feature_result_sign = torch.sign(feature_result_old)
                feature_result = feature_result - feature_result_old
                ind = np.argsort(abs(bn))[:]
                #  需要观察期望因子的变化大小，以及删除的通道个数
                ones_i = torch.ones(f, c).to(self.device)
                for i in range(len(ind)):
                    if pruned_num == 0:
                        break
                    if bn[ind[i]] > 0:
                        for j in range(len(feature_result[i])):
                            if feature_result_sign[i][j] < 0 and feature_result[i][j] > 0:  # -0.05越小 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                            elif feature_result_sign[i][j] > 0 and feature_result[i][j] < 0:  # 0.05越大 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                    else:
                        for j in range(len(feature_result[i])):
                            if feature_result_sign[i][j] < 0 and feature_result[i][j] < 0:  # 0.05越大 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                            elif feature_result_sign[i][j] > 0 and feature_result[i][j] > 0:  # -0.05越小 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break

                cnt_array = np.sum(ones_i.cpu().numpy() == 0)
                self.cpra.append(format(cnt_array / (f * c), '.2g'))
                ones = torch.ones(f, c, w, h).to(self.device)
                for i in range(f):
                    for j in range(c):
                        for k in range(w):
                            for l in range(h):
                                ones[i, j, k, l] = ones_i[i, j]
                self.mask[index] = ones
                item.data = item.data * self.mask[index]

                with open(resume, "wb") as f:
                    pickle.dump(self.mask, f)
                break
        # for index, item in enumerate(params):
        #
        #     if index == cov_id * param_per_cov:
        #         break
        #
        #     if index == (cov_id - 1) * param_per_cov:
        #
        #         f, c, w, h = item.size()
        #         pruned_num = int(self.compress_rate[cov_id - 1] * f * c)
        #         feature_result = abs(feature_result - 1.0)
        #         feare = np.array(feature_result.cpu()).reshape(f*c)
        #         ones_i = torch.ones(f, c).to(self.device)
        #
        #         ind = np.argsort(abs(feare))[:]
        #         mid = feare[ind[pruned_num]]
        #         ones_i[feature_result <= mid] =0
        #
        #         cnt_array = np.sum(ones_i.cpu().numpy() == 0)
        #         self.cpra.append(format(cnt_array / (f * c), '.2g'))
        #         ones = torch.ones(f, c, w, h).to(self.device)
        #         for i in range(f):
        #             for j in range(c):
        #                 for k in range(w):
        #                     for l in range(h):
        #                         ones[i, j, k, l] = ones_i[i, j]
        #         self.mask[index] = ones
        #         item.data = item.data * self.mask[index]
        #
        #         with open(resume, "wb") as f:
        #             pickle.dump(self.mask, f)
        #         break
    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            if index in range(0, 167, self.param_per_cov):
                item.data = item.data * self.mask[index].to(self.device)

    def pri_comp(self):
        print('compress_rate:{}'.format(self.cpra))

class mask_resnet_110:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='', device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.cpra = []
        self.job_dir = job_dir
        self.device = device

    def layer_mask(self, epoch, cov_id, trainloader_1, resume=None, param_per_cov=3, arch="resnet_110"):
        params = self.model.parameters()
        params = list(params)
        global feature_result
        global conv_result
        global weight
        global input_1
        for index, item in enumerate(params):
            if index == (cov_id - 1) * param_per_cov:
                weight = item
                break

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume = self.job_dir + '/mask'

        self.param_per_cov = param_per_cov

        if epoch == 0:
            for batch_idx, (inputs, targets) in enumerate(trainloader_1):
                if batch_idx >= 1:
                    break
                input_1 = inputs
            resume_input = self.job_dir + '/input'
            with open(resume_input, "wb") as f:
                pickle.dump(input_1, f)
        else:
            with open(self.job_dir + '/input', 'rb') as f:
                input_1 = pickle.load(f)

        self_conv()
        get_feature()

        for index, item in enumerate(params):
            if epoch == 0:
                resume_deta = self.job_dir + '/feature_result'
                with open(resume_deta, "wb") as f:
                    pickle.dump(feature_result, f)
                break

            if index == cov_id * param_per_cov:
                break

            if index == (cov_id - 1) * param_per_cov:

                with open(self.job_dir + '/feature_result', 'rb') as f:
                    feature_result_old = pickle.load(f)

                f, c, w, h = item.size()
                bn = torch.as_tensor(params[index + 1])
                bn = bn.detach().cpu().numpy()
                pruned_num = int(self.compress_rate[cov_id - 1] * f * c)
                feature_result_sign = torch.sign(feature_result_old)
                feature_result = feature_result - feature_result_old
                ind = np.argsort(abs(bn))[:]
                #  需要观察期望因子的变化大小，以及删除的通道个数
                ones_i = torch.ones(f, c).to(self.device)
                for i in range(len(ind)):
                    if pruned_num == 0:
                        break
                    if bn[ind[i]] > 0:
                        for j in range(len(feature_result[i])):
                            if feature_result_sign[i][j] < 0 and feature_result[i][j] > 0:  # -0.05越小 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                            elif feature_result_sign[i][j] > 0 and feature_result[i][j] < 0:  # 0.05越大 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                    else:
                        for j in range(len(feature_result[i])):
                            if feature_result_sign[i][j] < 0 and feature_result[i][j] < 0:  # 0.05越大 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                            elif feature_result_sign[i][j] > 0 and feature_result[i][j] > 0:  # -0.05越小 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break

                cnt_array = np.sum(ones_i.cpu().numpy() == 0)
                self.cpra.append(format(cnt_array / (f * c), '.2g'))
                ones = torch.ones(f, c, w, h).to(self.device)
                for i in range(f):
                    for j in range(c):
                        for k in range(w):
                            for l in range(h):
                                ones[i, j, k, l] = ones_i[i, j]
                self.mask[index] = ones
                item.data = item.data * self.mask[index]

                with open(resume, "wb") as f:
                    pickle.dump(self.mask, f)
                break

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            if index in range(0, 326, self.param_per_cov):
                item.data = item.data * self.mask[index].to(self.device)

    def pri_comp(self):
        print('compress_rate:{}'.format(self.cpra))

class mask_googlenet:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.cpra = []
        self.job_dir = job_dir
        self.device = device

    def layer_mask(self, epoch, cov_id, trainloader_1, resume=None, param_per_cov=28, arch="googlenet"):
        params = self.model.parameters()
        params = list(params)
        global feature_result
        global conv_result
        global weight
        global input_1

        for index, item in enumerate(params):
            if index == (cov_id-1) * param_per_cov + 4:
                break
            if (cov_id == 1 and index == 0) \
                    or index == (cov_id - 1) * param_per_cov - 24 \
                    or index == (cov_id - 1) * param_per_cov - 16 \
                    or index == (cov_id - 1) * param_per_cov - 8 \
                    or index == (cov_id - 1) * param_per_cov - 4 \
                    or index == (cov_id - 1) * param_per_cov:
                weight = item
                break

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume = self.job_dir + '/mask'

        self.param_per_cov = param_per_cov

        if epoch == 0:
            for batch_idx, (inputs, targets) in enumerate(trainloader_1):
                if batch_idx >= 1:
                    break
                input_1 = inputs
            resume_input = self.job_dir + '/input'
            with open(resume_input, "wb") as f:
                pickle.dump(input_1, f)
        else:
            with open(self.job_dir + '/input', 'rb') as f:
                input_1 = pickle.load(f)

        self_conv()
        get_feature()


        for index, item in enumerate(params):
            if epoch == 0:
                resume_deta = self.job_dir + '/feature_result'
                with open(resume_deta, "wb") as f:
                    pickle.dump(feature_result, f)
                break
            if index == (cov_id-1) * param_per_cov + 4:
                    break
            if (cov_id == 1 and index == 0) \
                    or index == (cov_id - 1) * param_per_cov - 24 \
                    or index == (cov_id - 1) * param_per_cov - 16 \
                    or index == (cov_id - 1) * param_per_cov - 8 \
                    or index == (cov_id - 1) * param_per_cov - 4 \
                    or index == (cov_id - 1) * param_per_cov:
                with open(self.job_dir + '/feature_result', 'rb') as f:
                    feature_result_old = pickle.load(f)

                f, c, w, h = item.size()
                bn = torch.as_tensor(params[index + 2])
                bn = bn.detach().cpu().numpy()
                pruned_num = int(self.compress_rate[cov_id - 1] * f * c)
                feature_result_sign = torch.sign(feature_result_old)
                feature_result = feature_result - feature_result_old
                ind = np.argsort(abs(bn))[:]
                #  需要观察期望因子的变化大小，以及删除的通道个数
                ones_i = torch.ones(f, c).to(self.device)
                for i in range(len(ind)):
                    if pruned_num == 0:
                        break
                    if bn[ind[i]] > 0:
                        for j in range(len(feature_result[i])):
                            if feature_result_sign[i][j] < 0 and feature_result[i][j] > -0.005:  # -0.05越小 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                            elif feature_result_sign[i][j] > 0 and feature_result[i][j] < 0.005:  # 0.05越大 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                    else:
                        for j in range(len(feature_result[i])):
                            if feature_result_sign[i][j] < 0 and feature_result[i][j] < 0.005:  # 0.05越大 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                            elif feature_result_sign[i][j] > 0 and feature_result[i][j] > -0.005:  # -0.05越小 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break

                cnt_array = np.sum(ones_i.cpu().numpy() == 0)
                self.cpra.append(format(cnt_array / (f * c), '.2g'))
                ones = torch.ones(f, c, w, h).to(self.device)
                for i in range(f):
                    for j in range(c):
                        for k in range(w):
                            for l in range(h):
                                ones[i, j, k, l] = ones_i[i, j]
                self.mask[index] = ones
                item.data = item.data * self.mask[index]

                with open(resume, "wb") as f:
                    pickle.dump(self.mask, f)
                break
        # for index, item in enumerate(params):
        #     if index == (cov_id-1) * param_per_cov + 4:
        #             break
        #     if (cov_id == 1 and index == 0) \
        #             or index == (cov_id - 1) * param_per_cov - 24 \
        #             or index == (cov_id - 1) * param_per_cov - 16 \
        #             or index == (cov_id - 1) * param_per_cov - 8 \
        #             or index == (cov_id - 1) * param_per_cov - 4 \
        #             or index == (cov_id - 1) * param_per_cov:
        #         f, c, w, h = item.size()
        #         pruned_num = int(self.compress_rate[cov_id - 1] * f * c)
        #         feature_result = abs(feature_result - 1.0)
        #         feare = np.array(feature_result.cpu()).reshape(f * c)
        #         ones_i = torch.ones(f, c).to(self.device)
        #
        #         ind = np.argsort(abs(feare))[:]
        #         mid = feare[ind[pruned_num]]
        #         ones_i[feature_result <= mid] = 0
        #
        #         cnt_array = np.sum(ones_i.cpu().numpy() == 0)
        #         self.cpra.append(format(cnt_array / (f * c), '.2g'))
        #         ones = torch.ones(f, c, w, h).to(self.device)
        #         for i in range(f):
        #             for j in range(c):
        #                 for k in range(w):
        #                     for l in range(h):
        #                         ones[i, j, k, l] = ones_i[i, j]
        #         self.mask[index] = ones
        #         item.data = item.data * self.mask[index]
        #
        #         with open(resume, "wb") as f:
        #             pickle.dump(self.mask, f)
        #         break
    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == (cov_id-1) * self.param_per_cov + 4:
                break
            if index not in self.mask:
                continue
            item.data = item.data * self.mask[index].to(self.device)

    def pri_comp(self):
        print('compress_rate:{}'.format(self.cpra))

class mask_resnet_50:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='', device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.cpra = []
        self.job_dir = job_dir
        self.device = device

    def layer_mask(self, epoch, cov_id, trainloader_1, resume=None, param_per_cov=3, arch="resnet_50"):
        params = self.model.parameters()
        params = list(params)
        global feature_result
        global conv_result
        global weight
        global input_1
        for index, item in enumerate(params):
            if index == (cov_id - 1) * param_per_cov:
                weight = item
                break

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume = self.job_dir + '/mask'

        self.param_per_cov = param_per_cov

        if epoch == 0:
            for batch_idx, (inputs, targets) in enumerate(trainloader_1):
                if batch_idx >= 1:
                    break
                inputs = inputs[0, 0, :, :]
                r_ind = random.sample(range(224), 192)
                input_1 = inputs.cpu()
                input_1 = np.delete(inputs, r_ind, axis=0)
                input_1 = torch.tensor(np.delete(input_1, r_ind, axis=1), device='cuda:1')
            resume_input = self.job_dir + '/input'
            with open(resume_input, "wb") as f:
                pickle.dump(input_1, f)
        else:
            with open(self.job_dir + '/input', 'rb') as f:
                input_1 = pickle.load(f)

        self_conv()
        get_feature()

        for index, item in enumerate(params):
            if epoch == 0:
                resume_deta = self.job_dir + '/feature_result'
                with open(resume_deta, "wb") as f:
                    pickle.dump(feature_result, f)
                break

            if index == cov_id * param_per_cov:
                break

            if index == (cov_id - 1) * param_per_cov:

                with open(self.job_dir + '/feature_result', 'rb') as f:
                    feature_result_old = pickle.load(f)

                f, c, w, h = item.size()
                bn = torch.as_tensor(params[index + 1])
                bn = bn.detach().cpu().numpy()
                pruned_num = int(self.compress_rate[cov_id - 1] * f * c)
                feature_result_sign = torch.sign(feature_result_old)
                feature_result = feature_result - feature_result_old
                ind = np.argsort(abs(bn))[:]
                #  需要观察期望因子的变化大小，以及删除的通道个数
                ones_i = torch.ones(f, c).to(self.device)
                for i in range(len(ind)):
                    if pruned_num == 0:
                        break
                    if bn[ind[i]] > 0:
                        for j in range(len(feature_result[i])):
                            if feature_result_sign[i][j] < 0 and feature_result[i][j] > -0.0001:  # -0.05越小 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                            elif feature_result_sign[i][j] > 0 and feature_result[i][j] < 0.0001:  # 0.05越大 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                    else:
                        for j in range(len(feature_result[i])):
                            if feature_result_sign[i][j] < 0 and feature_result[i][j] < 0.0001:  # 0.05越大 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                            elif feature_result_sign[i][j] > 0 and feature_result[i][j] > -0.0001:  # -0.05越小 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break

                cnt_array = np.sum(ones_i.cpu().numpy() == 0)
                self.cpra.append(format(cnt_array / (f * c), '.2g'))
                ones = torch.ones(f, c, w, h).to(self.device)
                for i in range(f):
                    for j in range(c):
                        for k in range(w):
                            for l in range(h):
                                ones[i, j, k, l] = ones_i[i, j]
                self.mask[index] = ones
                item.data = item.data * self.mask[index]

                with open(resume, "wb") as f:
                    pickle.dump(self.mask, f)
                break

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            if index in range(0, 161, self.param_per_cov):
                item.data = item.data * self.mask[index].to(self.device)

    def pri_comp(self):
        print('compress_rate:{}'.format(self.cpra))

class mask_resnet_18:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='', device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.cpra = []
        self.job_dir = job_dir
        self.device = device

    def layer_mask(self, epoch, cov_id,  trainloader_1, resume=None, param_per_cov=3, arch="resnet_56"):
        params = self.model.parameters()
        params = list(params)
        global feature_result
        global conv_result
        global weight
        global input_1
        for index, item in enumerate(params):
            if index == (cov_id - 1) * param_per_cov:
                weight = item
                break

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume = self.job_dir + '/mask'

        self.param_per_cov = param_per_cov

        if epoch == 0:
            for batch_idx, (inputs, targets) in enumerate(trainloader_1):
                if batch_idx >= 1:
                    break
                inputs = inputs[0,0,:,:]
                r_ind = random.sample(range(224), 192)
                input_1 = inputs.cpu()
                input_1 = np.delete(inputs,r_ind,axis=0)
                input_1 = torch.tensor(np.delete(input_1,r_ind, axis=1),device='cuda:1')

            resume_input = self.job_dir + '/input'
            with open(resume_input, "wb") as f:
                pickle.dump(input_1, f)
        else:
            with open(self.job_dir + '/input', 'rb') as f:
                input_1 = pickle.load(f)

        self_conv()
        get_feature()

        for index, item in enumerate(params):
            if epoch == 0:
                resume_deta = self.job_dir + '/feature_result'
                with open(resume_deta, "wb") as f:
                    pickle.dump(feature_result, f)
                break

            if index == cov_id * param_per_cov:
                break

            if index == (cov_id - 1) * param_per_cov:

                with open(self.job_dir + '/feature_result', 'rb') as f:
                    feature_result_old = pickle.load(f)

                f, c, w, h = item.size()
                bn = torch.as_tensor(params[index + 1])
                bn = bn.detach().cpu().numpy()
                pruned_num = int(self.compress_rate[cov_id - 1] * f * c)
                feature_result_sign = torch.sign(feature_result_old)
                feature_result = feature_result - feature_result_old
                ind = np.argsort(abs(bn))[:]
                #  需要观察期望因子的变化大小，以及删除的通道个数
                ones_i = torch.ones(f, c).to(self.device)
                for i in range(len(ind)):
                    if pruned_num == 0:
                        break
                    if bn[ind[i]] > 0:
                        for j in range(len(feature_result[i])):
                            if feature_result_sign[i][j] < 0 and feature_result[i][j] > -0.0005:  # -0.05越小 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                            elif feature_result_sign[i][j] > 0 and feature_result[i][j] < 0.0005:  # 0.05越大 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                    else:
                        for j in range(len(feature_result[i])):
                            if feature_result_sign[i][j] < 0 and feature_result[i][j] < 0.0005:  # 0.05越大 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break
                            elif feature_result_sign[i][j] > 0 and feature_result[i][j] > -0.0005:  # -0.05越小 保留的通道越少
                                ones_i[ind[i]][j] = 0
                                pruned_num = pruned_num - 1
                                if pruned_num == 0:
                                    break

                cnt_array = np.sum(ones_i.cpu().numpy() == 0)
                self.cpra.append(format(cnt_array / (f * c), '.2g'))
                ones = torch.ones(f, c, w, h).to(self.device)
                for i in range(f):
                    for j in range(c):
                        for k in range(w):
                            for l in range(h):
                                ones[i, j, k, l] = ones_i[i, j]
                self.mask[index] = ones
                item.data = item.data * self.mask[index]

                with open(resume, "wb") as f:
                    pickle.dump(self.mask, f)
                break

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            if index in range(0, 59, self.param_per_cov):
                item.data = item.data * self.mask[index].to(self.device)

    def pri_comp(self):
        print('compress_rate:{}'.format(self.cpra))