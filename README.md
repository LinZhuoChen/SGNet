<!-- PROJECT LOGO -->
<br />
  <h3 align="center">Spatial information guided Convolution for Real-Time 
  RGBD Semantic Segmentation</h3>
  <p align="center">
    Lin-Zhuo Chen, Zheng Lin, Ziqin Wang, Yong-Liang Yang and Ming-Ming Cheng
    <br />
    <a href="https://mmcheng.net/sgnet/"><strong>⭐ Project Home »</strong></a>
    <br />
    <!-- <a href="https://arxiv.org/pdf/2004.04534.pdf" target="_black">[PDF]</a>
    <a href="#" target="_black">[Code]</a>
    <br />
    <br /> -->
  </p>

</p>
<p align="center">
  <a href="https://arxiv.org/pdf/2004.04534.pdf">
    <img src="https://img.shields.io/badge/PDF-%F0%9F%93%83-green" target="_blank" />
  </a>
</p>

***
The official repo of the TIP 2021 paper ``
[Spatial information guided Convolution for Real-Time RGBD Semantic Segmentation](https://arxiv.org/pdf/2004.04534.pdf).

## Results on NYUDv2 Dataset

 Speed is related to the hardware spec (e.g. CPU, GPU, RAM, etc), so it is hard to make an equal comparison. 

 I  get the following results under NVIDIA 1080TI GPU, Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz:

|    Model     | mIoU(480x640) | mIoU(MS) | FPS(480x640) | FPS(425x560) |
| :----------: | :-----------: | :------: | :----------: | :----------: |
| SGNet(Res50) |     47.7%     |  48.6%   |      35      |      39      |
|    SGNet     |     49.8%     |  51.1%   |      26      |      28      |
|  SGNet_ASPP  |     50.2%     |  51.1%   |      24      |      26      |

If you want to measure speed on more advanced graphics card (such as 2080ti),  you can use the environment of pytorch 0.4.1 CUDA 9.2 to measure inference speed.

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

|                          |                           Dataset                            |                            model                             |                            model                             |                            model                             |
| ------------------------ | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| OneDrive                 | [NYUDv2](https://1drv.ms/u/s!AlDxLjilJDZoj2FrwVV9o8K8rhmI?e=AZ1POE]) | [SGNet_Res50](https://1drv.ms/u/s!AlDxLjilJDZokRMM62SCR3iOI_xk?e=00gLqJ) | [SGNet](https://1drv.ms/u/s!AlDxLjilJDZokRF-0oJUVr21lYzP?e=0NEVW1) | [SGNet_ASPP](https://1drv.ms/u/s!AlDxLjilJDZokRLcX9uMQFz1FuzP?e=Yq6G6K) |
| BaiduDrive(passwd: scon) |  [NYUDv2](https://pan.baidu.com/s/1lCrMu10IBepXXyGq3Vqphw)   | [SGNet_Res50](https://pan.baidu.com/s/1yj3llVf14uT17HzqTi6pjw) |   [SGNet](https://pan.baidu.com/s/1shzbcPjIKdq99Ji39OHIMg)   | [SGNet_ASPP](https://pan.baidu.com/s/1HeiJfHpIjSQKmFtYJhBrng) |

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
    
   ## SGNet_Res50
   python main.py ./configs/sgnet_res50_nyud_test.json
    ```
4. You can run the follow command to 
    test the model inference speed, input the image size such as 480 x 640:

   ```bash
   ## SGNet
   python main.py ./configs/sgnet_nyud_fps.json
    
   ## SGNet_ASPP
   python main.py ./configs/sgnet_aspp_nyud_fps.json
   
   ## SGNet_Res50
   python main.py ./configs/sgnet_res50_nyud_fps.json
   ```


## Citation

If you find this work is useful for your research, please cite our paper:
```
@article{21TIP-SGNet,   
  author={Lin-Zhuo Chen and Zheng Lin and Ziqin Wang and Yong-Liang Yang and Ming-Ming Cheng},   
  journal={IEEE Transactions on Image Processing},    
  title={Spatial Information Guided Convolution for Real-Time RGBD Semantic Segmentation}, 
  year={2021},   
  volume={30},
  pages={2313-2324},  
  doi={10.1109/TIP.2021.3049332} 
}
```

### Thanks to the Third Party Libs
[inplace_abn](https://github.com/mapillary/inplace_abn) - 
[Pytorch-Deeplab](https://github.com/speedinghzl/Pytorch-Deeplab) - 
[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
[PyTorch-segmentation-toolbox](https://github.com/speedinghzl/pytorch-segmentation-toolbox)
[Depth2HHA-python](https://github.com/charlesCXK/Depth2HHA-python)
[Pytorch-Project-Template](https://github.com/moemen95/Pytorch-Project-Template)
[Deformable-Convolution-V2-PyTorch](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch)
If you have any questions, feel free to contact me via `linzhuochen🥳foxmail😲com`