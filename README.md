# NeighConsensus
NCNet Re-implementation on **Pytorch 1.0 + Python3.6** (Original implementation : https://github.com/ignacio-rocco/ncnet)
For more information check out their project [website](https://www.di.ens.fr/willow/research/ncnet/) and their paper on [arXiv](https://arxiv.org/abs/1810.10510).

## Table of Content
* [Installation](#installation)
* [Functions Quick Search](https://github.com/XiSHEN0220/NeighConsensus/blob/master/model/README.md)
* [Train](#train)



## Installation

To download Pre-trained feature extractor (ResNet 18 ): 

``` Bash
cd model/FeatureExtractor
bash download.sh
```

To download PF Pascal Dataset : 

``` Bash
cd data/pf-pascal/
bash download.sh
```


### Functions Quick Search

For important functions, we provide a quick search [here](https://github.com/XiSHEN0220/NeighConsensus/blob/master/model/README.md)

### Train 

To train on PF-Pascal : 
``` Bash
bash demo_train.sh
``` 
