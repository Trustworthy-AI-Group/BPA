from torch.utils.data import Dataset
import PIL.Image as Image
import os
import csv
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import scipy.stats as st
from torchvision.transforms import InterpolationMode
import math


class SelectedImagenet(Dataset):
    def __init__(self, imagenet_val_dir, selected_images_csv, transform=None):
        super(SelectedImagenet, self).__init__()
        self.imagenet_val_dir = imagenet_val_dir
        self.selected_images_csv = selected_images_csv
        self.transform = transform
        self._load_csv()

    def _load_csv(self):
        reader = csv.reader(open(self.selected_images_csv, 'r'))
        next(reader)
        self.selected_list = list(reader)

    def __getitem__(self, item):
        target, target_name, image_name = self.selected_list[item]
        image = Image.open(os.path.join(
            self.imagenet_val_dir, image_name))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(target)

    def __len__(self):
        return len(self.selected_list)


Normalize = transforms.Normalize(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def input_diversity(img):
    gg = torch.randint(0, 2, (1,)).item()
    if gg == 0:
        return img
    else:
        rnd = torch.randint(224, 257, (1,)).item()
        rescaled = F.interpolate(img, (rnd, rnd), mode='nearest')
        h_rem = 256 - rnd
        w_hem = 256 - rnd
        pad_top = torch.randint(0, h_rem + 1, (1,)).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_hem + 1, (1,)).item()
        pad_right = w_hem - pad_left
        padded = F.pad(rescaled, pad=(
            pad_left, pad_right, pad_top, pad_bottom))
        padded = F.interpolate(padded, (224, 224), mode='nearest')
        return padded


def ila_forw_resnet50(model, x, ila_layer):
    jj = int(ila_layer.split('_')[0])
    kk = int(ila_layer.split('_')[1])
    x = model[0](x)
    x = model[1].conv1(x)
    x = model[1].bn1(x)
    x = model[1].relu(x)
    if jj == 0 and kk == 0:
        return x
    x = model[1].maxpool(x)

    for ind, mm in enumerate(model[1].layer1):
        x = mm(x)
        if jj == 1 and ind == kk:
            return x
    for ind, mm in enumerate(model[1].layer2):
        x = mm(x)
        if jj == 2 and ind == kk:
            return x
    for ind, mm in enumerate(model[1].layer3):
        x = mm(x)
        if jj == 3 and ind == kk:
            return x
    for ind, mm in enumerate(model[1].layer4):
        x = mm(x)
        if jj == 4 and ind == kk:
            return x
    return False


class ILAProjLoss(torch.nn.Module):
    def __init__(self):
        super(ILAProjLoss, self).__init__()

    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        n = old_attack_mid.shape[0]
        x = (old_attack_mid - original_mid).view(n, -1)
        y = (new_mid - original_mid).view(n, -1)
        # x_norm = x / torch.norm(x, dim = 1, keepdim = True)
        proj_loss = torch.sum(y * x) / n
        return proj_loss


class ReLU_SiLU_Function(Function):
    temperature=1.

    @staticmethod
    def forward(ctx, input_):
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        with torch.no_grad():
            output = torch.relu(input_)
        ctx.save_for_backward(input_)
        return output.to(input_.device)

    @staticmethod
    def backward(ctx, grad_output):
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput就是输入的参数grad_output、
        # 因此只需求relu的导数，在乘以grad_output
        input_, = ctx.saved_tensors
        with torch.no_grad():
            grad_input = input_ * \
                torch.sigmoid(input_)*(1-torch.sigmoid(input_)) + \
                torch.sigmoid(input_)
            grad_input = grad_input * grad_output * ReLU_SiLU_Function.temperature
        return grad_input.to(input_.device)


class ReLU_SiLU(nn.Module):
    def __init__(self):
        super(ReLU_SiLU, self).__init__()

    def forward(self, input):
        return ReLU_SiLU_Function.apply(input)


class ReLU_Linear_Function(Function):

    @staticmethod
    def forward(ctx, input_):
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        with torch.no_grad():
            output = torch.relu(input_)
        ctx.save_for_backward(input_)
        return output.to(input_.device)

    @staticmethod
    def backward(ctx, grad_output):
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput就是输入的参数grad_output、
        # 因此只需求relu的导数，在乘以grad_output
        input_, = ctx.saved_tensors
        with torch.no_grad():
            grad_input = grad_output.clone()
        return grad_input.to(input_.device)


class ReLU_Linear(nn.Module):
    def __init__(self):
        super(ReLU_Linear, self).__init__()

    def forward(self, input):
        return ReLU_Linear_Function.apply(input)


'''class MaxPool2dK2S2Function(Function):

    @staticmethod
    def forward(ctx, input_):
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        with torch.no_grad():
            output = F.max_pool2d(input_, 2, 2)
        ctx.save_for_backward(input_, output)         # 将输入保存起来，在backward时使用
        return output.to(input_.device)

    @staticmethod
    def backward(ctx, grad_output):
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput就是输入的参数grad_output、
        # 因此只需求relu的导数，在乘以grad_output
        input_, output = ctx.saved_tensors
        with torch.no_grad():
            #grad_input = torch.nn.functional.interpolate(grad_output, scale_factor=2) * torch.exp(input_) / torch.exp(torch.nn.functional.interpolate(output, scale_factor=2))
            grad_input = torch.nn.functional.interpolate(grad_output, scale_factor=2) * input_ / torch.nn.functional.interpolate(output, scale_factor=2)
        return grad_input.to(input_.device)


class MaxPool2dK2S2(nn.Module):
    def __init__(self):
        super(MaxPool2dK2S2, self).__init__()

    def forward(self, input):
        return MaxPool2dK2S2Function.apply(input)'''


'''def max_pool_2d_k3_s2_p1_hook(grad):
    grad = grad.clone()
    grad_pad = F.avg_pool2d(grad, kernel_size=3, padding=1, stride=1)
    grad = torch.where(grad.abs()>0., grad, grad_pad)
    return grad

class MaxPool2dK3S2P1(nn.Module):
    def __init__(self):       
        super(MaxPool2dK3S2P1, self).__init__()
        self.net=nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    def forward(self, x):
        x.register_hook(max_pool_2d_k3_s2_p1_hook)
        y=self.net(x)
        return y'''


class MaxPool2dK2S2Function(Function):
    temperature=1.
    @staticmethod
    def forward(ctx, input_):
        with torch.no_grad():
            output=F.max_pool2d(input_, 2, 2)
        ctx.save_for_backward(input_, output)
        return output.to(input_.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            input_, output = ctx.saved_tensors
            input_unfold = F.unfold(input_, 2, stride=2).reshape((input_.shape[0],input_.shape[1],2*2,grad_output.shape[2]*grad_output.shape[3]))
            
            # output_unfold=torch.exp(10*output.reshape(output.shape[0],output.shape[1],1,-1).repeat(1,1,9,1))
            output_unfold = torch.exp(MaxPool2dK2S2Function.temperature*input_unfold).sum(dim=2, keepdim=True)
            
            grad_output_unfold=grad_output.reshape(output.shape[0],output.shape[1],1,-1).repeat(1,1,4,1)
            grad_input_unfold=grad_output_unfold*torch.exp(MaxPool2dK2S2Function.temperature*input_unfold)/output_unfold
            grad_input_unfold=grad_input_unfold.reshape(input_.shape[0],-1,output.shape[2]*output.shape[3])
            grad_input=F.fold(grad_input_unfold, input_.shape[2:], 2, stride=2)
            return grad_input.to(input_.device)


class MaxPool2dK2S2(nn.Module):
    def __init__(self):
        super(MaxPool2dK2S2, self).__init__()

    def forward(self, input):
        return MaxPool2dK2S2Function.apply(input)


class MaxPool2dK3S2P1Function(Function):
    temperature=10.
    @staticmethod
    def forward(ctx, input_):
        with torch.no_grad():
            output=F.max_pool2d(input_, 3, 2, 1)
        ctx.save_for_backward(input_, output)
        return output.to(input_.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            input_, output = ctx.saved_tensors
            input_unfold = F.unfold(input_, 3, padding=1, stride=2).reshape((input_.shape[0],input_.shape[1],3*3,grad_output.shape[2]*grad_output.shape[3]))
            
            # output_unfold=torch.exp(10*output.reshape(output.shape[0],output.shape[1],1,-1).repeat(1,1,9,1))
            output_unfold = torch.exp(MaxPool2dK3S2P1Function.temperature*input_unfold).sum(dim=2, keepdim=True)
            
            grad_output_unfold=grad_output.reshape(output.shape[0],output.shape[1],1,-1).repeat(1,1,9,1)
            grad_input_unfold=grad_output_unfold*torch.exp(MaxPool2dK3S2P1Function.temperature*input_unfold)/output_unfold
            grad_input_unfold=grad_input_unfold.reshape(input_.shape[0],-1,output.shape[2]*output.shape[3])
            grad_input=F.fold(grad_input_unfold, input_.shape[2:], 3, padding=1, stride=2)
            return grad_input.to(input_.device)


class MaxPool2dK3S2P1(nn.Module):
    def __init__(self):
        super(MaxPool2dK3S2P1, self).__init__()

    def forward(self, input):
        return MaxPool2dK3S2P1Function.apply(input)


class MyBatchNorm2dFunction(Function):

    @staticmethod
    def forward(ctx, input_, weight, bias, mean, var, eps):
        with torch.no_grad():
            output = torch.batch_norm(
                input_, weight, bias, mean, var, False, 0.1, eps, True)
        ctx.save_for_backward(input_, weight, torch.tensor(eps), var)
        return output.to(input_.device)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, eps, var = ctx.saved_tensors
        eps = eps.item()
        with torch.no_grad():
            running_var = torch.var(input_, dim=[0, 2, 3])
            grad_input = weight / torch.sqrt(running_var + eps)
            grad_input = grad_output * grad_input.unsqueeze(0).unsqueeze(2).unsqueeze(
                3).repeat(input_.shape[0], 1, input_.shape[2], input_.shape[3]).to(input_.device)
        return grad_input, None, None, None, None, None


class MyBatchNorm2d(nn.Module):
    def __init__(self, weight, bias, mean, var, eps):
        super(MyBatchNorm2d, self).__init__()
        self.weight = weight
        self.bias = bias
        self.mean = mean
        self.var = var
        self.eps = eps

    def forward(self, input):
        return MyBatchNorm2dFunction.apply(input, self.weight, self.bias, self.mean, self.var, self.eps)

class GaussianSmoothConv2d3x3(nn.Module):
    def __init__(self, channels):
        super(GaussianSmoothConv2d3x3, self).__init__()
        self.channels=channels
        kernel_backbone=torch.tensor([[1./8, 1./8, 1./8],
                                      [1./8, 0., 1./8],
                                      [1./8, 1./8, 1./8]])*2
        '''kernel_backbone=torch.tensor([[1./16, 1./8, 1./16],
                                      [1./8, 1./4, 1./8],
                                      [1./16, 1./8, 1./16]])'''
        '''kernel_backbone=torch.tensor([[1./24, 1./12, 1./24],
                                      [1./12, 1./2, 1./12],
                                      [1./24, 1./12, 1./24]])'''
        kernel_backbone=kernel_backbone.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.weight = nn.Parameter(kernel_backbone).requires_grad_(False)

    def forward(self, x):
        out = F.conv2d(x, self.weight, stride=1, padding=1, groups=self.channels)
        return out

def gaussian_smooth_backward_hook(module, grad_input, grad_output):
    with torch.no_grad():
        grad_pad = GaussianSmoothConv2d3x3(grad_input[0].shape[1]).to(grad_input[0].device)(grad_input[0])
        new_grad_input = (torch.where(grad_input[0].abs()>0., grad_input[0], grad_pad), )
    return new_grad_input

def resnet_weight_backward_hook(module, grad_input, grad_output):
    with torch.no_grad():
        new_grad_input = (grad_input[0]*0.5,)
    return new_grad_input

def linbp_forw_resnet50(model, x, do_linbp, linbp_layer):
    jj = int(linbp_layer.split('_')[0])
    kk = int(linbp_layer.split('_')[1])
    x = model[0](x)
    x = model[1].conv1(x)
    x = model[1].bn1(x)
    x = model[1].relu(x)
    x = model[1].maxpool(x)
    ori_mask_ls = []
    conv_out_ls = []
    relu_out_ls = []
    conv_input_ls = []
    def layer_forw(jj, kk, jj_now, kk_now, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp):
        if jj < jj_now:
            x, ori_mask, conv_out, relu_out, conv_in = block_func(mm, x, linbp=True)
            ori_mask_ls.append(ori_mask)
            conv_out_ls.append(conv_out)
            relu_out_ls.append(relu_out)
            conv_input_ls.append(conv_in)
        elif jj == jj_now:
            if kk_now >= kk:
                x, ori_mask, conv_out, relu_out, conv_in = block_func(mm, x, linbp=True)
                ori_mask_ls.append(ori_mask)
                conv_out_ls.append(conv_out)
                relu_out_ls.append(relu_out)
                conv_input_ls.append(conv_in)
            else:
                x, _, _, _, _ = block_func(mm, x, linbp=False)
        else:
            x, _, _, _, _ = block_func(mm, x, linbp=False)
        return x, ori_mask_ls
    for ind, mm in enumerate(model[1].layer1):
        x, ori_mask_ls = layer_forw(jj, kk, 1, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer2):
        x, ori_mask_ls = layer_forw(jj, kk, 2, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer3):
        x, ori_mask_ls = layer_forw(jj, kk, 3, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer4):
        x, ori_mask_ls = layer_forw(jj, kk, 4, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    x = model[1].avgpool(x)
    x = torch.flatten(x, 1)
    x = model[1].fc(x)
    return x, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls

def block_func(block, x, linbp):
    identity = x
    conv_in = x+0
    out = block.conv1(conv_in)
    out = block.bn1(out)
    out_0 = out + 0
    if linbp:
        out = linbp_relu(out_0)
    else:
        out = block.relu(out_0)
    ori_mask_0 = out.data.bool().int()

    out = block.conv2(out)
    out = block.bn2(out)
    out_1 = out + 0
    if linbp:
        out = linbp_relu(out_1)
    else:
        out = block.relu(out_1)
    ori_mask_1 = out.data.bool().int()

    out = block.conv3(out)
    out = block.bn3(out)

    if block.downsample is not None:
        identity = block.downsample(identity)
    identity_out = identity + 0
    x_out = out + 0


    out = identity_out + x_out
    out = block.relu(out)
    ori_mask_2 = out.data.bool().int()
    return out, (ori_mask_0, ori_mask_1, ori_mask_2), (identity_out, x_out), (out_0, out_1), (0, conv_in)


def linbp_relu(x):
    x_p = F.relu(-x)
    x = x + x_p.data
    return x

    # return ReLU_SiLU_Function.apply(x)
    


def linbp_backw_resnet50(img, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls, xp):
    for i in range(-1, -len(conv_out_ls)-1, -1):
        if i == -1:
            grads = torch.autograd.grad(loss, conv_out_ls[i])
        else:
            grads = torch.autograd.grad((conv_out_ls[i+1][0], conv_input_ls[i+1][1]), conv_out_ls[i], grad_outputs=(grads[0], main_grad_norm))
        normal_grad_2 = torch.autograd.grad(conv_out_ls[i][1], relu_out_ls[i][1], grads[1]*ori_mask_ls[i][2],retain_graph=True)[0]
        normal_grad_1 = torch.autograd.grad(relu_out_ls[i][1], relu_out_ls[i][0], normal_grad_2 * ori_mask_ls[i][1], retain_graph=True)[0]
        normal_grad_0 = torch.autograd.grad(relu_out_ls[i][0], conv_input_ls[i][1], normal_grad_1 * ori_mask_ls[i][0], retain_graph=True)[0]
        del normal_grad_2, normal_grad_1
        main_grad = torch.autograd.grad(conv_out_ls[i][1], conv_input_ls[i][1], grads[1])[0]
        alpha = normal_grad_0.norm(p=2, dim = (1,2,3), keepdim = True) / main_grad.norm(p=2,dim = (1,2,3), keepdim=True)
        main_grad_norm = xp * alpha * main_grad
    input_grad = torch.autograd.grad((conv_out_ls[0][0], conv_input_ls[0][1]), img, grad_outputs=(grads[0], main_grad_norm))
    return input_grad[0].data

def DI_299(X_in,prob=0.7):
    rnd = np.random.randint(299, 330,size=1)[0]
    h_rem = 330 - rnd
    w_rem = 330 - rnd
    pad_top = np.random.randint(0, h_rem,size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem,size=1)[0]
    pad_right = w_rem - pad_left
    c = np.random.rand(1)
    if c <= prob:
        X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)
        return  X_out 
    else:
        return  X_in

def RDI(x_adv):
    x_di = x_adv 
    di_pad_amount=256-224
    
    di_pad_value=0

    ori_size = x_di.shape[-1]
    rnd = int(torch.rand(1) * di_pad_amount) + ori_size
    x_di = transforms.Resize((rnd, rnd), interpolation=InterpolationMode.NEAREST)(x_di)
    pad_max = ori_size + di_pad_amount - rnd
    pad_left = int(torch.rand(1) * pad_max)
    pad_right = pad_max - pad_left
    pad_top = int(torch.rand(1) * pad_max)
    pad_bottom = pad_max - pad_top
    x_di = F.pad(x_di, (pad_left, pad_right, pad_top, pad_bottom), 'constant', di_pad_value)
    x_di = transforms.Resize((ori_size, ori_size), interpolation=InterpolationMode.NEAREST)(x_di)


    return x_di

def compute_rotation(angles, device):
    """
    Return:
        rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat
    Parameters:
        angles           -- torch.tensor, size (B, 3), radian
    """

    batch_size = angles.shape[0]
    ones = torch.ones([batch_size, 1]).to(device)
    zeros = torch.zeros([batch_size, 1]).to(device)
    x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
    
    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x), 
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([batch_size, 3, 3])
    
    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([batch_size, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)

def rigid_transform( vs, rot, trans):
    vs_r = torch.matmul(vs, rot)
    vs_t = vs_r + trans.view(-1, 1, 3)
    return vs_t

def render_3d_aug_input(x_adv, renderer,prob=0.7):
    c = np.random.rand(1)
    if c <= prob:
        x_ri=x_adv.clone()
        for i in range(x_adv.shape[0]):
            x_ri[i]=renderer.render(x_adv[i].unsqueeze(0))
        return  x_ri 
    else:
        return  x_adv



def calculate_v(model, x_adv_or_nes, y, eps, number_of_v_samples, beta, target_label, attack_type, number_of_si_scales, prob,loss_fn,renderer, device):
    sum_grad_x_i = torch.zeros_like(x_adv_or_nes)
    for i in range(number_of_v_samples):
        x_i = x_adv_or_nes.clone().detach() + (torch.rand(x_adv_or_nes.size()).to(device)*2-1.) * (beta * eps)
        x_i.requires_grad = True
        if 'S' in attack_type: 
            ghat = calculate_si_ghat(model, x_i, y, number_of_si_scales, target_label, attack_type, prob,loss_fn,renderer)
        else:
            if 'D' in attack_type:
                x_i2 = DI(x_i,prob)
            elif 'R' in attack_type:
                x_i2 = RDI(x_i)
            elif 'O' in attack_type:
                x_i2 = render_3d_aug_input(x_i,renderer=renderer,prob=prob)
            else:
                x_i2 = x_i
            output_x_adv_or_nes = model(x_i2)
            loss= loss_fn(output_x_adv_or_nes)
            ghat = torch.autograd.grad(loss, x_i,
                    retain_graph=False, create_graph=False)[0]
        sum_grad_x_i += ghat.detach()
    v = sum_grad_x_i / number_of_v_samples
    return v


def calculate_si_ghat(model, x_adv_or_nes, y, number_of_si_scales, target_label, attack_type, prob, loss_fn,renderer, device):
    x_neighbor = x_adv_or_nes.clone().detach()
    grad_sum = torch.zeros_like(x_neighbor).to(device)
    for si_counter in range(0, number_of_si_scales):
        si_div = 2 ** si_counter
        si_input = (((x_adv_or_nes.clone().detach()-0.5)*2 / si_div)+1)/2 # 0 1 -> -1 1
        si_input.requires_grad = True
        # Diverse-Input
        if 'D' in attack_type:
            si_input2 = DI(si_input,prob)
        elif 'R' in attack_type:
            si_input2 = RDI(si_input)
        elif 'O' in attack_type:
            si_input2 = render_3d_aug_input(si_input,renderer=renderer,prob=prob)
        else:
            si_input2 = si_input
        output_si = model(si_input2)

        loss_si=loss_fn(output_si)
        si_input_grad = torch.autograd.grad(loss_si, si_input,
                retain_graph=False, create_graph=False)[0]
        grad_sum += si_input_grad*(1/si_div)

    ghat = grad_sum
    return ghat


def DI(X_in,prob=0.7):
    rnd = np.random.randint(224, 257,size=1)[0]
    h_rem = 256 - rnd
    w_rem = 256 - rnd
    pad_top = np.random.randint(0, h_rem,size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem,size=1)[0]
    pad_right = w_rem - pad_left
    c = np.random.rand(1)
    if c <= prob:
        X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)
        return  X_out 
    else:
        return  X_in

def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def vgg19_ila_forw(model, x, ila_layer):
    x = model[0](x)
    for ind, mm in enumerate(model[1].features):
        x = mm(x)
        if ind == ila_layer:
            return x