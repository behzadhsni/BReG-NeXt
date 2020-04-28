# BReG-NeXt
Implementation of the paper **BReG-NeXt: Facial Affect Computing Using Adaptive Residual Networks With Bounded Gradient**


BReG-NeXt paper can be found on 
[IEEE Xplore](https://ieeexplore.ieee.org/document/9064942) and
[arXiv](https://arxiv.org/abs/2004.08495)

![overview](overview_modular3.png)

# Requirements

Tensorflow 1.14.0 is suggested to run the code. For installing the rest of the required packages, run the following command:
```
pip install -r requirements.txt
```
# Content
* tfrecords: Sample tfrecords for training and validation from FER2013 database
* Snapshots: Example model trained on AffectNet database on BReG-NeXt-50
* Logs: Log report of the BReG-NeXt-50 trained model on AffectNet database

# How to run
Simply run the `BReG-NeXt.py` file:
```
codes/>> python BReG-NeXt.py
```
# Citation
All submitted papers (or any publically available text) that uses the entire or parts of this code, must cite the following paper:

**B. Hasani, P. S. Negi and M. Mahoor, "BReG-NeXt: Facial affect computing using adaptive residual networks with bounded gradient," in IEEE Transactions on Affective Computing, 2020.**

BibTex:

```
@ARTICLE{9064942,  author={B. {Hasani} and P. S. {Negi} and M. {Mahoor}},  journal={IEEE Transactions on Affective Computing},  title={BReG-NeXt: Facial affect computing using adaptive residual networks with bounded gradient},   year={2020},  volume={},  number={},  pages={1-1},}
```
