<!-- PROJECT LOGO -->
<br />
  <h3 align="center">Spatial information guided Convolution for Real-Time 
  RGBD Semantic Segmentation</h3>

  <p align="center">
    Lin-Zhuo Chen, Zheng Lin, Ziqin Wang, Yong-Liang Yang and Ming-Ming Cheng
    <br />
    <a href="http://linzhuo.xyz/sgnet.html"><strong>⭐ Project Home »</strong></a>
    <br />
    <!-- <a href="http://mftp.mmcheng.net/Papers/21TIP-SGNet.pdf" target="_black">[PDF]</a>
    <a href="#" target="_black">[Code]</a>
    <a href="http://linzhuo.xyz/papers/SGNet/translation.pdf" target="_black">[中译版]</a>
    <br />
    <br /> -->
  </p>
</p>
<p align="center">
  <a href="http://mftp.mmcheng.net/Papers/21TIP-SGNet.pdf">
    <img src="https://img.shields.io/badge/PDF-%F0%9F%93%83-green" target="_blank" />
  </a>
  <a href="http://zhaozhang.net/papers/20_GICD/translation.pdf">
    <img src="https://img.shields.io/badge/%E4%B8%AD%E8%AF%91%E7%89%88-%F0%9F%90%BC-red">
  </a>
</p>

***
The official repo of the TIP 2021 paper ``
[Spatial information guided Convolution for Real-Time RGBD Semantic Segmentation](http://mftp.mmcheng.net/Papers/21TIP-SGNet.pdf).


## Prerequisites
#### Environments
* PyTorch == 0.4.1
* tqdm
* CUDA==8.0
* CUDNN=7.1.4
* pillow
* numpy
* tensorboardX
* tqdm
#### Trained model and dataset
Download NYUDv2 dataset and trained model: 

|                          | Dataset                                                      | model                                                        | model                                                        |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| OneDrive                 | [NYUDv2](https://1drv.ms/u/s!AlDxLjilJDZoj2FrwVV9o8K8rhmI?e=AZ1POE]) | [SGNet](https://1drv.ms/u/s!AlDxLjilJDZokRF-0oJUVr21lYzP?e=0NEVW1) | [SGNet_ASPP](https://1drv.ms/u/s!AlDxLjilJDZokRLcX9uMQFz1FuzP?e=Yq6G6K) |
| BaiduDrive(passwd: scon) | [NYUDv2](https://pan.baidu.com/s/1lCrMu10IBepXXyGq3Vqphw)    | [SGNet](https://pan.baidu.com/s/1shzbcPjIKdq99Ji39OHIMg)     | [SGNet_ASPP](https://pan.baidu.com/s/1HeiJfHpIjSQKmFtYJhBrng) |

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
To test the model with imput size $480 \times 640$, please run:

    ```bash
    ## SGNet
    python main.py ./configs/sgnet_nyud_test.json

    ## SGNet_ASPP
    python main.py ./configs/sgnet_aspp_nyud_test.json
    ```
    You will get the following results:
   
   |   Model    |   mIoU    | mIoU(multi-scale) |
   | :--------: | :-------: | :---------------: |
   |   SGNet    |   49.8%   |       51.1%       |
   | SGNet_ASPP | **50.2%** |       51.1%       |
   
4. You can run the follow command to 
    test the model inference speed, input the image size such as 480 x 640:

   ```bash
   ## SGNet
   python main.py ./configs/sgnet_nyud_fps.json
    
   ## SGNet_ASPP
   python main.py ./configs/sgnet_aspp_nyud_fps.json
   ```
   
   Speed is related to the hardware spec (e.g. CPU, GPU, RAM, etc), so it is hard to make an equal comparison. 
   
   I  get the following results under NVIDIA 1080TI GPU, Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz:
   
   |   Model    | FPS(480 x 640) | FPS(425 x 560) |
   | :--------: | :------------: | :------------: |
   |   SGNet    |       26       |       29       |
   | SGNet_ASPP |       24       |       26       |

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