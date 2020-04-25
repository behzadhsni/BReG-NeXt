# BReG-NeXt
Implementation of the paper **BReG-NeXt: Facial Affect Computing Using Adaptive Residual Networks With Bounded Gradient**


BReG-NeXt paper can be found on 
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
