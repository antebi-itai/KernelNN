import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import shave_a2b, resize_tensor_w_kernel, create_penalty_mask, map2tensor
import math


# noinspection PyUnresolvedReferences
class GANLoss(nn.Module):
    """D outputs a [0,1] map of size of the input. This map is compared in a pixel-wise manner to 1/0 according to
    whether the input is real (i.e. from the input image) or fake (i.e. from the Generator)"""

    def __init__(self, d_last_layer_size):
        super(GANLoss, self).__init__()
        print("Using GAN loss!")
        # The loss function is applied after the pixel-wise comparison to the true label (0/1)
        self.loss = nn.L1Loss(reduction='mean')
        # Make a shape
        d_last_layer_shape = [1, 1, d_last_layer_size, d_last_layer_size]
        # The two possible label maps are pre-prepared
        self.label_tensor_fake = Variable(torch.zeros(d_last_layer_shape).cuda(), requires_grad=False)
        self.label_tensor_real = Variable(torch.ones(d_last_layer_shape).cuda(), requires_grad=False)

    def forward(self, d_last_layer, is_d_input_real):
        # Determine label map according to whether current input to discriminator is real or fake
        label_tensor = self.label_tensor_real if is_d_input_real else self.label_tensor_fake
        # Compute the loss
        return self.loss(d_last_layer, label_tensor)


class DownScaleLoss(nn.Module):
    """ Computes the difference between the Generator's downscaling and an ideal (bicubic) downscaling"""

    def __init__(self, scale_factor):
        super(DownScaleLoss, self).__init__()
        self.loss = nn.MSELoss()
        bicubic_k = [[0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [-.0013275146484375, -0.0039825439453130, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0013275146484380, -0.0039825439453125, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625]]
        self.bicubic_kernel = Variable(torch.Tensor(bicubic_k).cuda(), requires_grad=False)
        self.scale_factor = scale_factor

    def forward(self, g_input, g_output):
        downscaled = resize_tensor_w_kernel(im_t=g_input, k=self.bicubic_kernel, sf=self.scale_factor)
        # Shave the downscaled to fit g_output
        return self.loss(g_output, shave_a2b(downscaled, g_output))


class SumOfWeightsLoss(nn.Module):
    """ Encourages the kernel G is imitating to sum to 1 """

    def __init__(self):
        super(SumOfWeightsLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.ones(1).to(kernel.device), torch.sum(kernel))


class CentralizedLoss(nn.Module):
    """ Penalizes distance of center of mass from K's center"""

    def __init__(self, k_size, scale_factor=.5):
        super(CentralizedLoss, self).__init__()
        self.indices = Variable(torch.arange(0., float(k_size)).cuda(), requires_grad=False)
        wanted_center_of_mass = k_size // 2 + 0.5 * (int(1 / scale_factor) - k_size % 2)
        self.center = Variable(torch.FloatTensor([wanted_center_of_mass, wanted_center_of_mass]).cuda(), requires_grad=False)
        self.loss = nn.MSELoss()

    def forward(self, kernel):
        """Return the loss over the distance of center of mass from kernel center """
        r_sum, c_sum = torch.sum(kernel, dim=1).reshape(1, -1), torch.sum(kernel, dim=0).reshape(1, -1)
        return self.loss(torch.stack((torch.matmul(r_sum, self.indices) / torch.sum(kernel),
                                      torch.matmul(c_sum, self.indices) / torch.sum(kernel))), self.center)


class BoundariesLoss(nn.Module):
    """ Encourages sparsity of the boundaries by penalizing non-zeros far from the center """

    def __init__(self, k_size):
        super(BoundariesLoss, self).__init__()
        self.mask = map2tensor(create_penalty_mask(k_size, 30))
        self.zero_label = Variable(torch.zeros(k_size).cuda(), requires_grad=False)
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(kernel * self.mask, self.zero_label)


class SparsityLoss(nn.Module):
    """ Penalizes small values to encourage sparsity """
    def __init__(self):
        super(SparsityLoss, self).__init__()
        self.power = 0.2
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.abs(kernel) ** self.power, torch.zeros_like(kernel))


def calc_dist_l2(X, Y):
    """
    Calculate distances between patches
    :param X: tensor of n patches of size k
    :param Y: tensor of m patches of size k
    :return: l2 distance matrix - tensor of shape n * m
    """
    Y = Y.transpose(0, 1)
    X2 = X.pow(2).sum(1, keepdim=True)
    Y2 = Y.pow(2).sum(0, keepdim=True)
    XY = X @ Y
    return X2 - (2 * XY) + Y2


class NNLoss(nn.Module):
    """ Distance to NN in original image """

    def __init__(self, original_image, patch_size=5):
        super(NNLoss, self).__init__()
        print("Using NN loss!")
        self.patch_size = patch_size
        self.original_patches = F.unfold(original_image, kernel_size=self.patch_size).squeeze().t()
        self.loss = calc_dist_l2

    def forward(self, crop):
        crop_patches = F.unfold(crop, kernel_size=self.patch_size).squeeze().t()
        dists_mat = self.loss(crop_patches, self.original_patches)
        patchNN_dists, patchNN_indices = dists_mat.min(dim=1)
        loss = patchNN_dists.mean()
        return loss, patchNN_indices


class NNTracker:
    def __init__(self, images_shape, crop_size=64, kernel_size=13, scale=2, patch_size=5):
        self.crop_size = crop_size
        self.kernel_size = kernel_size
        self.scale = scale
        self.patch_size = patch_size
        self.lr_crop_size = (self.crop_size - self.kernel_size) // self.scale + 1
        self.patches_size = (self.lr_crop_size - patch_size) + 1

        self.prev_indices = torch.zeros(images_shape) * float('nan')
        self.curr_indices = torch.zeros(images_shape) * float('nan')

        self.num_indices_in_prev = []
        self.num_indices_changed = []

    def update(self, patchNN_indices, top, left):
        # reshape the indices
        patchNN_indices = patchNN_indices.reshape([1, 1, self.patches_size, self.patches_size])

        # shift to distinguish zeros added from true zero indices
        patchNN_indices = patchNN_indices.float()
        patchNN_indices += 0.1

        # dilate the indices by self.scale
        transposed_conv_weight = F.pad(
            torch.ones(1, 1, 1, 1, dtype=patchNN_indices.dtype, device=patchNN_indices.device),
            (self.scale-1, self.scale-1, self.scale-1, self.scale-1))
        patchNN_indices = F.conv_transpose2d(patchNN_indices, transposed_conv_weight, stride=self.scale, padding=self.scale-1).squeeze()
        # pad the dilated indices to align with the original crop
        pad_before = math.floor(self.kernel_size / 2) + self.scale * (self.patch_size // 2)
        pad_after = math.ceil(self.kernel_size / 2) + self.scale * (self.patch_size // 2)
        patchNN_indices = F.pad(patchNN_indices, (pad_before, pad_after, pad_before, pad_after))

        patchNN_indices[patchNN_indices == 0] = float('nan') # all zeros added (dilation / padding) should be nan
        patchNN_indices = patchNN_indices.floor() # restore original indices from shifted indices

        # update the prev and curr indices accordingly
        not_nan_indices = torch.logical_not(patchNN_indices.isnan())
        del self.prev_indices
        self.prev_indices = self.curr_indices.clone()
        self.curr_indices[top:top+self.crop_size, left:left+self.crop_size][not_nan_indices] = patchNN_indices[not_nan_indices]

        self.num_indices_in_prev.append(self.get_num_indices_in_prev())
        self.num_indices_changed.append(self.get_num_indices_changed())

    def get_num_indices_in_prev(self):
        return (torch.logical_not(self.prev_indices.isnan())).count_nonzero()

    def get_num_indices_changed(self):
        # both not nan and are different
        different_indices = self.prev_indices != self.curr_indices
        not_nan_indices = (torch.logical_not(self.prev_indices.isnan()) & torch.logical_not(self.curr_indices.isnan()))
        return (different_indices & not_nan_indices).count_nonzero()


class LossTracker:
    def __init__(self):
        self.losses = []
        self.regs_percentage = []

    def update(self, loss, reg):
        loss, reg = loss.item(), reg.item()
        self.losses.append(loss)
        self.regs_percentage.append(100 * (reg / loss))
