# BPA

This repository contains code to reproduce results from the paper:

[Rethinking the Backward Propagation for Adversarial Transferability](https://arxiv.org/abs/2306.12685) (NeurIPS 2023)

Xiaosen Wang, Kangheng Tong, Kun He

> We also include the code in the framework [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack).

## Environments

* Python 3.10.9
* PyTorch 2.0.0
* Torchvision 0.15.0
* Timm 0.6.13
* Pillow 9.4.0
* Numpy 1.23.5

## Datasets(ImageNet)

Following [LinBP](https://github.com/qizhangli/linbp-attack), we randomly sample 5,000 images pertaining to the 1,000 categories
from ILSVRC 2012 validation set, which could be classified correctly by all the victim models. The corresponding CSV file are saved as **`data/imagenet/selected_imagenet_resnet50.csv`** and **`data/imagenet/selected_imagenet_vgg19_bn.csv`**, and you can refer to it to select other images for testing.

Before running, you should prepare the ILSVRC 2012 validation set and specify the directory using parameters `--imagenet_val_dir`. If your directory structure is as follows:

```
imagenet
└───val
    ├── ILSVRC2012_val_00001822.JPEG
    └── ILSVRC2012_val_00011685.JPEG
    └── ...
```

No modifications should be made. But if your directory structure is as follows:

```
imagenet-val
    ├── n07880968
    |   ├── ILSVRC2012_val_00011685.JPEG
    |   └── ...
    ├── n02927161
    |   ├── ILSVRC2012_val_00011685.JPEG
    |   └── ...
    └── ...
```

Line 32of file `utils.py` should be modified to `self.imagenet_val_dir, target_name, image_name))`

## Models

Please download pretrained models at [here](https://drive.google.com/file/d/1L2BW1w40oia_-l1AZ6KytwK5b7EKoqag/view?usp=drive_link), then extract them to `ckpt/`.

## Attack & Test

Untargeted attack using **`BPA+PGD`**:

```bash
python attack_eval_imagenet.py --epsilon 0.03 --sgm_lambda 1.0 --niters 10 --method max_relu_silu_pgd --batch_size 25 --save_dir dat
a/imagenet/0 --device_id 0 --imagenet_val_dir ../datasets/imagenet-val --model_name resnet50 --alpha 0.006
```

Meaning of parameters are as follows:

* --epsilon: The maximum magnitude of perturbation $\epsilon$;
* --niters: The iterations of attack;
* --method: The method of attack. You can combine the following tricks and base methods. For some examples, `max_relu_silu_pgd`, `max_relu_linear_ssa_mifgsm` and so on.
  * `max` means our modification of max-pooling layers
  * `relu_silu` means our modification of ReLU layers
  * `relu_linear` means the method of baseline LinBP
  * `ghost` means the method of baseline Ghost
  * `sgm` means the method of baseline SGM
  * `pgd` or `mifgsm` or `vmifgsm` or `ila_ifgsm` or `ssa_mifgsm` are our selected base attack methods.
* --linbp_layer: The first layer to be modified by `LinBP` and `BPA`
* --batch_size: The batch size for attack&test
* --save_dir: the directory to save generated adversarial images
* --imagenet_val_dir: the directory of you imagenet validation set
* --model_name: `resnet50` or `vgg19_bn`
* --alpha: step size $\alpha$

To reproduce the results of the first line of Table 1, you can run the following commands (Actual results may vary slightly):

```bash
python attack_eval_imagenet.py --epsilon 0.03 --sgm_lambda 1.0 --niters 10 --method pgd --batch_size 25 --save_dir data/imagenet/0 --device_id 0 --imagenet_val_dir ../datasets/imagenet-val --model_name resnet50 --alpha 0.006
python attack_eval_imagenet.py --epsilon 0.03 --sgm_lambda 1.0 --niters 10 --method sgm_pgd --batch_size 25 --save_dir data/imagenet/0 --device_id 0 --imagenet_val_dir ../datasets/imagenet-val --model_name resnet50 --alpha 0.006
python attack_eval_imagenet.py --epsilon 0.03 --sgm_lambda 1.0 --niters 10 --method ghost_pgd --batch_size 25 --save_dir data/imagenet/0 --device_id 0 --imagenet_val_dir ../datasets/imagenet-val --model_name resnet50 --alpha 0.006
python attack_eval_imagenet.py --epsilon 0.03 --sgm_lambda 1.0 --niters 10 --method relu_linear_pgd --batch_size 25 --save_dir data/imagenet/0 --device_id 0 --imagenet_val_dir ../datasets/imagenet-val --model_name resnet50 --alpha 0.006
python attack_eval_imagenet.py --epsilon 0.03 --sgm_lambda 1.0 --niters 10 --method max_relu_silu_pgd --batch_size 25 --save_dir data/imagenet/0 --device_id 0 --imagenet_val_dir ../datasets/imagenet-val --model_name resnet50 --alpha 0.006
```

## Citation

```tex
@inproceedings{wang2023rethinkinga,
     title={{Rethinking the Backward Propagation for Adversarial Transferability}},
     author={Xiaosen Wang and Kangheng Tong and Kun He},
     booktitle={Proceedings of the Advances in Neural Information Processing Systems},
     year={2023}
}
```

