# SUNet: Symmetric Undistortion Network for Rolling Shutter Correction

This repository contains the source code for the paper: [SUNet: Symmetric Undistortion Network for Rolling Shutter Correction (ICCV2021)](http://openaccess.thecvf.com/content/ICCV2021/papers/Fan_SUNet_Symmetric_Undistortion_Network_for_Rolling_Shutter_Correction_ICCV_2021_paper.pdf)

<img src="result_demo/rs.gif" height="280px"/> <img src="result_demo/our.gif" height="280px"/>

<table>
  <thead>
    <tr>
      <td>Input rolling shutter image&nbsp;&nbsp;&nbsp;&nbsp;</td>
      <td>Recovered global shutter image</td>
    </tr>
  </thead>
  <tr>
    <td colspan="2">
      <img src="result_demo/rs.gif">
        <img src="result_demo/our.gif">
        </img>
      </a>
    </td>
  </tr>
</table>

## Installation
Install the dependent packages:
```
pip install -r requirements.txt
```
The code is tested with PyTorch 1.6.0 with CUDA 10.2.89.

Note that in our implementation, we borrowed some modules from [DeepUnrollNet](https://github.com/ethliup/DeepUnrollNet).

#### Install correlation package
```
cd ./package_correlation
python setup.py install
```
#### Install differentiable forward warping package
```
cd ./package_forward_warp
python setup.py install
```
#### Install core package
```
cd ./package_core
python setup.py install
```
## Demo with our pretrained model
You can now test our model with the provided images in the `demo` folder. 
To do this, simply run
```
sh demo.sh
```
The visualization results will be stored in the `experiments` folder. Other examples in the dataset can be tested similarly.

## Datasets
- **Carla-RS** and **Fastec-RS:** Download them to your local computer from [here](https://github.com/ethliup/DeepUnrollNet).

## Training and evaluating
You can run following commands to re-train the network.
```
# !! Please update the corresponding paths in 'train.sh' with  #
# !! your own local paths, before run following command!!      #

sh train.sh
```

You can run following commands to obtain the quantitative evaluations.
```
# !! Please update the path to test data in 'inference.sh'
# !! with your own local path, before run following command!!

sh inference.sh
```

## Citations
Please cite our paper if necessary:
```
@inproceedings{fan_SUNet_ICCV21,
  title={SUNet: Symmetric Undistortion Network for Rolling Shutter Correction},
  author={Fan, Bin and Dai, Yuchao and He, Mingyi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4541--4550},
  year={2021}
}
```

## Contact
Please drop me an email for further problems or discussion: binfan@mail.nwpu.edu.cn
