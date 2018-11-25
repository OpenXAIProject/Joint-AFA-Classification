Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding
==

Python implementation of the model and learning algorithm proposed by Shim et
al., 2018

## Result

This is a deep learning method to solve multi-class classification problems in a
cost-sensitive setting. Given dataset whose feature acquisition costs are
defined, our model tries to find the optimal feature acquisition policy and
the classifier working with the policy. After training is done, this code prints
out the test results (classification performance, # of times that each feature is
acquired, etc.) of training/validation/test dataset.

<p align="center">
<img
src="https://github.com/OpenXAIProject/Joint-AFA-Classification/blob/master/dfs_result.png"  width="800">
</p>

## Dataset

CUBE dataset (used in the paper) is generated as default. You can also use your own dataset in the form of
the csv file whose first column is label followed by features values. Set
`data_type` argument as 'csv' and pass csv filename as a keyword argument named
`csv_filename` to `data_load` function. You should define
feature acquisition cost and pass it to `r_cost` argument.

## Installation

**1. Fork & Clone** : Fork this project to your repository and clone to your work directory.

  ```bash
  $ git clone https://github.com/OpenXAIProject/Joint-AFA-Classification.git
  ```

**2. Run** : Run python3 main.py --data_type=csv or cube_[feature
dimension]_[sigma]. You can check additional options by following command.

```bash
$ python3 main.py --help
```

## Requirements
+ python 3.5
+ pytorch (0.4.1)
+ numpy (1.15.0)
+ matplotlib (2.2.2)
+ scikit-learn (0.19.1)

## Reference
If you found the provided code useful, please cite our work.

```
@inproceedings{shim2018jointAFA,
    author    = {Hajin Shim and Sung Ju Hwangand Eunho Yang and },
    title     = {Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding},
    booktitle = {NIPS},
    year      = {2018}
              }
```

<br/>


## Contacts
If you have any question, please contact Hajin Shim(shimazing@kaist.ac.kr).

<br />
<br />

# XAI Project

**This work was supported by Institute for Information & Communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.2017-0-01779, A machine learning and statistical inference framework for explainable artificial intelligence)**

+ Project Name : A machine learning and statistical inference framework for explainable artificial intelligence(의사결정 이유를 설명할 수 있는 인간 수준의 학습·추론 프레임워크 개발)

+ Managed by Ministry of Science and ICT/XAIC <img align="right" src="http://xai.unist.ac.kr/static/img/logos/XAIC_logo.png" width=300px>

+ Participated Affiliation : UNIST, Korea Univ., Yonsei Univ., KAIST, AItrics

+ Web Site : <http://openXai.org>

