
# Semi-Supervised Active Learning for Camera Trap Data

## Author: Amy Andrews
## Supervisor: Professor Gabriel Brostow
## Submission date: September 2021

---------

This repository documents the code used for the thesis Semi-Supervised Active Learning for Camera Trap Data, which was submitted as part requirement for the MSc Data Science and Machine Learning at UCL. The thesis can be shared on request. Please note that this work and code has been used only in an academic research setting but has not been quality assured or subject to a peer review process. 

The code implements FixMatch, a semi supervised learning model, in an active learning procedure. It also documents experiments run under different label-scarcity settings with different query strategies, one of which is novel. Finally it also documents the creation of a proof-of-concept dash application to visualise the active learning process to non-technical users, allowing interactive labelling during a human-in-the-loop retraining system.  

Please note that code in this repository includes code accessed from other repositories and this is cited in context. The files `ema.py` and `wideresnet.py` are composed entirely of code authored by others, as detailed in the files themselves. Where code from other work is used elsewhere it is referenced in the appropriate place. The creation of this codebase was also supported by Omiros Pantazis and Gabriel Brostow. Each file is headed with the purpose of the file and resources used.

`requirements.txt` is included to recreate the necessary python environment and references the packages used throughout the code.

The data for which this code was optimised has not yet been publically released, however there is a potential future avenue to update the code to allow it to work with open-source datasets, and I may continue to add to this project.

The file `set_args.py` is used to set the arguments to run the code base, but can also be used as a guide to run `main.py` from the command line using `python main.py --[args]`. A config file which is not contained in this release specifies internal paths to models etc.. 

### Reference repositories for this code (also referenced in-situ)

- Official FixMatch implementation at https://github.com/google-research/fixmatch/tree/d4985a158065947dba803e626ee9a6721709c570
- Unofficial PyTorch FixMatch implementations at https://github.com/kekmodel/FixMatch-pytorch/tree/ced359021e0b48bf3be48504f98ea483579c6e71/models and https://github.com/Celiali/FixMatch
- Data loading structures at https://github.com/omipan/camera_traps_self_supervised/blob/main/datasets.py
- Code in RandAugment paper https://arxiv.org/pdf/1909.13719.pdf                          
- Pytorch documentation https://pytorch.org/vision/stable/transforms.html             
- Unofficial pytorch implementation of randaugment https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
- official implementation of FixMatch reference to Tensorflow authors at https://github.com/google-research/fixmatch/blob/d4985a158065947dba803e626ee9a6721709c570/third_party/auto_augment/augmentations.py
- Kcenter greedy references https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py
- Query strategy references https://github.com/microsoft/CameraTraps/tree/f69de8339bf19806fbe943963e2b385b948c35b2/research/active_learning
- Exponential moving average implementation https://github.com/YUE-FAN/FixMatch-PyTorch/blob/master/utils/ema.py

### Development status

As of September 2021 not under active development.