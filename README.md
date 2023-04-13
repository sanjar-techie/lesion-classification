### DLeasion classification
Resnet50 is benchmarked by adding cbam attention blocks

- Built in setup.py
- Built in requirements
- Examples with Dataset

#### Goals  
The goal of this seed is to see how effective cbam module is for leasion classification task    
 
---

<div align="center">    
 
# Skin Lesion Classification     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/abs/1807.06521)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
What it does   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/sanjar-techie/lesion-classification.git

# install project   
cd lesion-classification
# pip install -e .   
# pip install -r requirements.txt
 ```   
 Next, you can open your code editor or open resnet50_benchmark.ipynb in Google Colab
 <!-- ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
``` -->

## Dataset
Download the dataset like so:
```python
! pip install opendatasets --upgrade
import opendatasets as od

dic = {"username":"sanjartechie","key":"82c494b2e40d1e481393dcf1d0e797d8"} # kaggle.jason

data_dir = '/content'
data_url = 'https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/download'
if not os.path.exists(data_dir + '/ham1000'):
  od.download(data_url, data_dir=data_dir)
else:
  print('\n -> dataset exists')
```

<!-- ### Citation   
```
@article{Sanjar,
  title={Mr},
  author={Sanjar},
  year={2020}
}
```    -->
