# Requirement
The codes are implemented in tensorflow--1.14 or 1.15 under the interpreter python3.6 or python3.7.  Additionally, if the codes are runned on a Server, one should use the miniconda3 for python 3.7 or 3.6. However, if you dowmload the latest version of miniconda3 from https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh, you will get a miniconda3 based on python 3.8.  Hence, you should redirect to the https://docs.conda.io/en/latest/miniconda.html, then download the miniconda3 based on python3.7.

## Requirement for packages:
import tkinter as TK
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import platform
import shutil
import time

# Corresponding Papers

## A multi-scale DNN algorithm for nonlinear elliptic equations with multiple scales  
created by Xi-An Li, Zhi-Qin John, Xu and Lei Zhang

[[Paper]](https://arxiv.org/pdf/2009.14597.pdf)

### Ideas
This work exploited the technique of shifting the input data in narrow-range into large-range, then fed the transformed data into the DNN pipline.
