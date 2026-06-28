"""
33. RepVGG  (Ding, Zhang, Ma, Han, Ding & Sun, 2021 — "Making VGG-style ConvNets Great Again")
==============================================================================================

RepVGG's trick is **structural re-parameterization**: train with a rich,
multi-branch block, then algebraically fuse it into a single 3x3 convolution for
inference. You get the trainability of a ResNet-like block and the blazing speed
of a plain VGG-style stack — same weights, identical outputs.

  * **Training-time block** = 3x3 conv(+BN)  +  1x1 conv(+BN)  +  identity(BN),
    summed, then ReLU. The parallel paths behave like residuals and help
    optimization.
  * **Inference-time block** = one 3x3 conv + ReLU. Fusion folds each BN into its
    conv, pads the 1x1 kernel to 3x3, turns the identity branch into a 3x3
    identity kernel, and adds them all up.

This script trains the multi-branch model, reports test metrics, then calls
`switch_to_deploy()` and re-evaluates — printing the max output difference (≈0) to
prove the re-parameterization is exact while the model is now a plain conv stack.

    --variant a   width 0.75x       --variant b   width 1.0x

Run:
    python "33.repvgg.py" --variant a --epochs 5
    python "33.repvgg.py" --variant b --limit 2000
"""

import copy
import os

import torch
import torch.nn as nn

import mnist_common as mc

# variant -> per-stage width multiplier applied to base widths [16,32,64,128]
VARIANTS = {"a": 0.75, "b": 1.0}
BASE_WIDTHS = [16, 32, 64, 128]
STAGE_DEPTHS = [2, 4, 6, 1]


def _conv_bn(in_ch, out_ch, kernel, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
    )


class RepVGGBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride
        self.relu = nn.ReLU(inplace=True)
        self.deploy = False
        # identity branch only when shape is preserved
        self.bn_identity = nn.BatchNorm2d(in_ch) if (in_ch == out_ch and stride == 1) else None
        self.conv3 = _conv_bn(in_ch, out_ch, 3, stride, 1)
        self.conv1 = _conv_bn(in_ch, out_ch, 1, stride, 0)
        self.reparam = None                          # the fused 3x3 conv (after deploy)

    def forward(self, x):
        if self.deploy:
            return self.relu(self.reparam(x))
        idn = 0 if self.bn_identity is None else self.bn_identity(x)
        return self.relu(self.conv3(x) + self.conv1(x) + idn)

    # ----- re-parameterization -----
    def _fuse_conv_bn(self, branch):
        if branch is None:
            return 0.0, 0.0
        if isinstance(branch, nn.Sequential):        # conv + BN
            conv, bn = branch[0], branch[1]
            kernel = conv.weight
        else:                                        # identity BN: build an identity kernel
            bn = branch
            kernel = torch.zeros(self.in_ch, self.in_ch, 3, 3, device=bn.weight.device)
            for i in range(self.in_ch):
                kernel[i, i, 1, 1] = 1.0
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    @staticmethod
    def _pad_1x1_to_3x3(k):
        return 0.0 if isinstance(k, float) else nn.functional.pad(k, [1, 1, 1, 1])

    def switch_to_deploy(self):
        k3, b3 = self._fuse_conv_bn(self.conv3)
        k1, b1 = self._fuse_conv_bn(self.conv1)
        kid, bid = self._fuse_conv_bn(self.bn_identity)
        kernel = k3 + self._pad_1x1_to_3x3(k1) + kid
        bias = b3 + b1 + bid
        self.reparam = nn.Conv2d(self.in_ch, self.out_ch, 3, stride=self.stride, padding=1, bias=True)
        self.reparam.weight.data = kernel
        self.reparam.bias.data = bias
        self.deploy = True
        for attr in ("conv3", "conv1", "bn_identity"):
            if hasattr(self, attr):
                self.__delattr__(attr)


class RepVGG(nn.Module):
    def __init__(self, width_mult, num_classes=10):
        super().__init__()
        widths = [max(8, int(w * width_mult)) for w in BASE_WIDTHS]
        layers = []
        in_ch = 1
        for stage, (w, depth) in enumerate(zip(widths, STAGE_DEPTHS)):
            for i in range(depth):
                stride = 2 if (i == 0 and stage > 0) else 1     # downsample at each stage start
                layers.append(RepVGGBlock(in_ch, w, stride))
                in_ch = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(in_ch, num_classes))

    def forward(self, x):
        return self.head(self.features(x))

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()


def main():
    p = mc.build_argparser("RepVGG on MNIST")
    p.add_argument("--variant", choices=list(VARIANTS), default="a")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = RepVGG(VARIANTS[args.variant])

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"RepVGG-{args.variant.upper()}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))

    # Demonstrate exact re-parameterization: fuse to a plain 3x3 stack, re-check.
    print("Re-parameterizing multi-branch blocks -> single 3x3 convs ...")
    deploy_model = copy.deepcopy(model).to(device)
    deploy_model.switch_to_deploy()
    deploy_model.eval()
    model.eval()
    with torch.no_grad():
        x = next(iter(test_loader))[0].to(device)
        max_diff = (model(x) - deploy_model(x)).abs().max().item()
    train_p = sum(p.numel() for p in model.parameters())
    deploy_p = sum(p.numel() for p in deploy_model.parameters())
    print(f"Max output diff train vs deploy: {max_diff:.2e}  (should be ~0)")
    print(f"Params: train {train_p:,} -> deploy {deploy_p:,}")


if __name__ == "__main__":
    main()
