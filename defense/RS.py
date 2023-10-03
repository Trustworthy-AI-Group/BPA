import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os
from typing import *

class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
    
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

def get_model():
    checkpoint = torch.load('checkpoint.pth.tar')
    model=models.resnet50()
    cudnn.benchmark = True
    normalize_layer = NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    base_classifier = torch.nn.Sequential(normalize_layer, model)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    smoothed_classifier = Smooth(base_classifier, 1000, 0.25)
    return smoothed_classifier

def test(save_dir, target_attack, device, if_minus=False):
    model=get_model()
    target = torch.from_numpy(np.load(save_dir + '/labels.npy')).long()
    if target_attack:
        label_switch = torch.tensor(list(range(500, 1000)) + list(range(0, 500))).long()
        target = label_switch[target]
    img_num = 0
    count = 0
    advfile_ls = os.listdir(save_dir)
    for advfile_ind in range(len(advfile_ls)-1):
        adv_batch = torch.from_numpy(np.load(save_dir + '/batch_{}.npy'.format(advfile_ind))).float() / 255
        if advfile_ind == 0:
            adv_batch_size = adv_batch.shape[0]
        img = adv_batch
        img_num += img.shape[0]
        label = target[advfile_ind * adv_batch_size : advfile_ind*adv_batch_size + adv_batch.shape[0]]
        label = label.to(device)
        img = img.to(device)
        with torch.no_grad():
            # pred = torch.argmax(model(trans(img)), dim=1).view(1,-1)
            pred = model.predict(img, 100, 0.001, 50)
            if if_minus:
                pred = pred -torch.tensor(1, dtype=torch.long)
        count += (label != pred.squeeze(0)).sum().item()
        del pred, img
        del adv_batch
    return round(100. - 100. * count / img_num, 2) if target_attack else round(100. * count / img_num, 2)
