<!-- PROJECT LOGO -->
<br />
  <h3 align="center">Spatial information guided Convolution for Real-Time 
  RGBD Semantic Segmentation</h3>

  <p align="center">
    Lin-Zhuo Chen, Zheng Lin, Ziqin Wang, Yong-Liang Yang and Ming-Ming Cheng
    <br />
    <a href="http://linzhuo.xyz/sgnet.html"><strong>⭐ Project Home »</strong></a>
    <br />
    <!-- <a href="https://arxiv.org/pdf/2004.04534.pdf" target="_black">[PDF]</a>
    <a href="#" target="_black">[Code]</a>
    <a href="http://linzhuo.xyz/papers/SGNet/translation.pdf" target="_black">[中译版]</a>
    <br />
    <br /> -->
  </p>
</p>
<p align="center">
  <a href="https://arxiv.org/pdf/2004.04534.pdf">
    <img src="https://img.shields.io/badge/PDF-%F0%9F%93%83-green" target="_blank" />
  </a>
  <a href="http://zhaozhang.net/papers/20_GICD/translation.pdf">
    <img src="https://img.shields.io/badge/%E4%B8%AD%E8%AF%91%E7%89%88-%F0%9F%90%BC-red">
  </a>
</p>


***
The official repo of the TIP 2021 paper ``
[Spatial information guided Convolution for Real-Time 
  RGBD Semantic Segmentation](https://arxiv.org/pdf/2004.04534.pdf).

More details can be found at our [project home.](http://linzhuo.xyz/sgnet.html)



## Prerequisites
#### Environments
* PyTorch == 0.4.1
* tqdm
* CUDA==9.2
* CUDNN=7.1.4
* pillow
* numpy
#### Pretrained model
Download [dataset](https://1drv.ms/u/s!AlDxLjilJDZoj2FrwVV9o8K8rhmI?e=AZ1POE]) 
and pretrained model: [SGNet](https://1drv.ms/u/s!AlDxLjilJDZoj18iaFYjVMLsS5U5?e=8TZSKC) 

<!-- USAGE EXAMPLES -->
## Usage
1. Put the pretrained model into `pretrained_weights` folder and unzip the dataset into `dataset` folder.

2. To compile the InPlace-ABN and S-Conv operation, please run:
    ```bash
    ## compile InPlace-ABN 
    cd graphs/ops/libs
    sh build.sh
    python build.py
    ## compile S-Conv
    cd ..
    sh make.sh
    ```
3. Modify the config in `configs/sgnet_nyud_test.json` (mainly check "trained_model_path"). 
To test the model, please run:

    ```bash
    ## SGNet
    python main.py ./configs/sgnet_nyud_test.json
    ```

## Citation
If you find this work is useful for your research, please cite our paper:
```
@ARTICLE{chen2021sconv,
 title={Spatial Information Guided Convolution for Real-Time RGBD Semantic Segmentation},
 author={Chen, Lin-Zhuo and Lin, Zheng and Wang, Ziqin and Yang, Yong-Liang and Cheng, Ming-Ming},
 journal={IEEE Transactions on Image Processing},
 year={2021}
}
```

## Contact
If you have any questions, feel free to contact me via `linzhuochen🥳foxmail😲com`