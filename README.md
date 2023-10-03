# CCFENet-PyTorch (TCSVT-2022)

IEEE TCSVT 2022: [Cross-Collaborative Fusion-Encoder Network for Robust RGB-Thermal Salient Object Detection.](https://ieeexplore.ieee.org/document/9801871)


## Requirements
* Pytorch 1.3.0+   

* Torchvision   

* PIL   

* Numpy 


## Results
* RGB-Thermal saliency maps mentioned in the paper can be download from [here](https://pan.baidu.com/s/1v6CwfPIdWzQWiCEoq5gceg) [code: gprv]  

* RGB-Depth saliency maps mentioned in the paper can be download from [here](https://pan.baidu.com/s/1DEjxz9C1muJaJsIcG5Kzjg) [code: qoc7]  

* The results of challenging scenarios mentioned in the paper can be download from [here](https://pan.baidu.com/s/1EFQygrVPARYEQVjk2OhirQ) [code: fteh]  

* The saliency results can be evaluated by using the tool in [Matlab](http://dpfan.net/d3netbenchmark/)  


## Testing
* Download the trained model weight from [here](https://pan.baidu.com/s/1ogxwapJK8bvY4BFsVm0VFQ) [code: ij0a]

* Modify your `test_root` in test.py

* Test the CCFENet: `python test.py`


## Citation
Please consider citing our work if you use this repository in your reseach.
```
@ARTICLE{CCFENet_TCSVT22,
  author={Liao, Guibiao and Gao, Wei and Li, Ge and Wang, Junle and Kwong, Sam},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Cross-Collaborative Fusion-Encoder Network for Robust RGB-Thermal Salient Object Detection}, 
  year={2022},
  volume={32},
  number={11},
  pages={7646-7661},
  }
```


## Acknowlogdement
This repository is built under the help of the following projects for academic use only:

* [PyTorch](https://github.com/pytorch/pytorch)

* [JL-DCF](https://github.com/jiangyao-scu/JL-DCF-pytorch) 
