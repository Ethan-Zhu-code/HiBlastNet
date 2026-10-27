# Third-party notices

Parts of this repository are adapted from the open-source projects below. Each remains under its own license; the full texts are in [`licenses/`](licenses/).

| Component | Files | Upstream | License |
| --- | --- | --- | --- |
| GCNet (backbone of HiBlastNet) | `blastseg/models/gcnet.py` (GCNet-S backbone; the BC-MSRR branch and auxiliary heads are new) | [gyyang23/GCNet](https://github.com/gyyang23/GCNet) | MIT, [`licenses/GCNET_LICENSE.txt`](licenses/GCNET_LICENSE.txt) |
| PIDNet | `blastseg/models/pidnet.py` | [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet) | MIT, [`licenses/PIDNET_LICENSE.txt`](licenses/PIDNET_LICENSE.txt) |
| SCTNet | `blastseg/models/sctnet.py` | [xzz2/SCTNet](https://github.com/xzz2/SCTNet) | MIT, [`licenses/SCTNET_LICENSE.txt`](licenses/SCTNET_LICENSE.txt) |
| EfficientViT | `blastseg/models/efficientvit.py` | [mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit) | Apache 2.0, [`licenses/EFFICIENTVIT_LICENSE.txt`](licenses/EFFICIENTVIT_LICENSE.txt) |
| SegFormer | `blastseg/models/segformer*.py` | [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer) | NVIDIA Source Code License, non-commercial research and evaluation use only, [`licenses/SEGFORMER_LICENSE.txt`](licenses/SEGFORMER_LICENSE.txt) |
| DeepLabV3-ResNet50 | provided by torchvision | [pytorch/vision](https://github.com/pytorch/vision) | BSD 3-Clause (installed package) |

The SegFormer files may be used only for non-commercial research or evaluation. Users who need commercial use should remove the SegFormer comparison model.

The U-Net comparison model (`blastseg/models/unet.py`) is an independent implementation of the architecture of Ronneberger et al. (2015) written for this repository.

## SCTNet

`blastseg/models/sctnet.py` is adapted from the official SCTNet implementation, Copyright (c) 2024 Zhengze Xu, MIT License.

## EfficientViT

`blastseg/models/efficientvit.py` is adapted from the official EfficientViT implementation, Copyright 2023 Han Cai, licensed under the Apache License, Version 2.0.
