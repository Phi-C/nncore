# nncore
nncore is a light-weight core library that proides the most common and essential functionality shared in neural network research developed with Python/PyTorch stack. All components in this library are type-annotated, tested, and benchmarked.

# Features
Besides some basic utilities, nncore includes the following features:
* PyTorch memory profiler in `nncore.memory_profiler`
* PyTorch flops counter in `nncore.flops`

# Install
## 1. Install from Github
```shell
pip install -U git+https://github.com/Phi-C/nncore.git
```

## 2. Install from a local clone
```shell
git clone https://github.com/Phi-C/nncore.git
pip install -e nncore
```

# Acknowlegements
* This project is inspired by [fvcore](https://github.com/facebookresearch/fvcore)
* PyTorch memory profiler is adapted from [subclass_zoo](https://github.com/albanD/subclass_zoo)
* Flops counter is adapted from [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch)

# License
This library is released undder the [Apache 2.0 licence](https://github.com/Phi-C/nncore/blob/main/LICENSE)

