# MSVTNet: Multi-Scale Vision Transformer Neural Network for EEG-Based Motor Imagery Decoding

# Architecture

![MSVTNet](MSVTNet_Arch.png)

The proposed MSVTNet network comprises three main blocks: multi-scale spatio-temporal convolutional (MSST) block, cross-scale global temporal 
encoder (CSGT) block, and auxiliary branch loss (ABL) block. Each branch in the MSST block extracts local spatiotemporal feature representations 
from the MI-EEG signals. By using multiple independent branches across different scales, more informative encoded representations can be 
extracted. Before entering the CSGT block, the encoded features from different scales of the MSST block and a class token are concatenated along 
the feature dimension to create an integrated global spatiotemporal representation enriched with multi-scale features. Internally within the CSGT 
block, cross-scale global temporal correlations are modeled through a multi-head self-attention (MHSA) mechanism, and the embedded class token is 
further input into the classifier (CLS) for final decoding. The ABL block, serving as intermediate supervision, addresses the parameter imbalance 
problem between the MSST and CSGT blocks to prevent overfitting. At the same time, it enhances the feature extraction capabilities of each branch.

# Development environment

All models were trained and tested by a single GPU, Nvidia GeForce RTX 3090 ([Driver 530.41.03](https://www.nvidia.com/Download/driverResults.aspx/200481/), [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)) on [Ubuntu 22.04.2 LTS](https://releases.ubuntu.com/jammy/).
The main following packages are required:

- [Python 3.10.13](https://www.python.org/downloads/release/python-31013/)
- [NumPy 1.23.5](https://numpy.org/doc/stable/release/1.23.5-notes.html)
- [PyTorch 2.0.1 + PyTorch CUDA 11.8](https://pytorch.org/get-started/previous-versions/#v201)
- [dpeeg 0.3.6](https://pypi.org/project/dpeeg/0.3.6/)

more detailed dependencies are in [environment.yml](https://github.com/SheepTAO/MSVTNet/tree/main/out/environment.yml). Build the development
environment using [Miniconda](https://docs.anaconda.com/free/miniconda/):

```shell
conda env create -f environment.yml
```

# Implementation

All the core codes are placed in the [dpeeg 0.3.6](https://github.com/SheepTAO/dpeeg/tree/6085816cbeca376d8d2f5c5b5d2d0b40cf757089) in the form 
of package functions, which provides some convenient interface functions to support the experiments in this paper. This repositories only 
provides the top-level training code. For details about the training code and related experimental methods, please check **dpeeg**.

After installing **dpeeg**, you should be able to run the code in the repositories correctly. If you want to run a certain analysis experiment, comment out the corresponding experiment code and run it from the beginning. All training details of the algorithms are also 
provided in the [out](https://github.com/SheepTAO/MSVTNet/tree/main/out) folder for reference. The file tree of `out` is as follows:

```Shell
out
|- decoding algorithm
   |- KFold
      |- datasets_SD
         |- ...
      |- datasets_SI
   |- LOSO_HO
      |- datasets
|- ...
```

where `decoding algorithm` folder contains all experimental analysis results for the corresponding algorithm. Inside, `KFold` stores the 
subject-dependent analysis results (`datasets_SD` and `datasets_SI` are the session-dependent and session-independent analysis results for the 
corresponding dataset, respectively), and `LOSO_HO` stores the subject-independent analysis results for the corresponding dataset. Note: Since the EEG Conformer results exceeded 7G, they were not uploaded.

# Results

The overall classification results for MSVTNet and other competing architectures are as follows:

![Results](MSVTNet_Results.png)

For easy reference, detailed results of all analyses are reorganized in the [supporting document](https://github.com/SheepTAO/MSVTNet/tree/main/Supplement_document_of_MSVTNet.pdf).

# References
