# Sign Language Transformers (CVPR'20)

This repo contains the training and evaluation code for the paper [Sign Language Transformers: Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation](https://www.cihancamgoz.com/pub/camgoz2020cvpr.pdf). 

This code is based on [Joey NMT](https://github.com/joeynmt/joeynmt) but modified to realize joint continuous sign language recognition and translation. For text-to-text translation experiments, you can use the original Joey NMT framework.
 
## Requirements
* Download the feature files using the `data/download.sh` script.

* [Optional] Create a conda or python virtual environment.

* Install required packages using the `requirements.txt` file.

    `pip install -r requirements.txt`

## Usage

  `python -m signjoey train configs/sign.yaml` 

## Weights and data 
通过网盘分享的文件：SLT-DATA
链接: https://pan.baidu.com/s/1x_SDXYPHwTzlEJwrKlHr3A?pwd=xhys 提取码: xhys


! Note that the default data directory is `./data`. If you download them to somewhere else, you need to update the `data_path` parameters in your config file.   
## ToDo:

- [X] *Initial code release.*
- [X] *Release image features for Phoenix2014T.*
- [ ] Share extensive qualitative and quantitative results & config files to generate them.
- [ ] (Nice to have) - Guide to set up conda environment and docker image.

## Reference

Please cite the paper below if you use this code in your research:

     @inproceedings{tang2025sign,
         title={Sign-IDD: Iconicity Disentangled Diffusion for Sign Language Production},
         author={Tang, Shengeng and He, Jiayi and Guo, Dan and Wei, Yanyan and Li, Feng and Hong, Richang},
         booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
         volume={39},
         number={7},
         pages={7266--7274},
         year={2025}
      }
    
    @article{tang2024discrete,
      title={Discrete to Continuous: Generating Smooth Transition Poses from Sign Language Observation},
      author={Tang, Shengeng and He, Jiayi and Cheng, Lechao and Wu, Jingjing and Guo, Dan and Hong, Richang},
      journal={arXiv preprint arXiv:2411.16810},
      year={2024}
    }
    
    @article{tang2024GCDM,
      title={Gloss-Driven Conditional Diffusion Models for Sign Language Production},
      author={Tang, Shengeng and Xue, Feng and Wu, Jingjing and Wang, Shuo and Hong, Richang},
      journal={ACM Transactions on Multimedia Computing, Communications, and Applications},
      issn = {1551-6857},
      year={2024},
    }

    @inproceedings{camgoz2020sign,
      author = {Necati Cihan Camgoz and Oscar Koller and Simon Hadfield and Richard Bowden},
      title = {Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation},
      booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2020}
    }

## Acknowledgements
<sub>This work was funded by the SNSF Sinergia project "Scalable Multimodal Sign Language Technology for Sign Language Learning and Assessment" (SMILE) grant agreement number CRSII2 160811 and the European Union’s Horizon2020 research and innovation programme under grant agreement no. 762021 (Content4All). This work reflects only the author’s view and the Commission is not responsible for any use that may be made of the information it contains. We would also like to thank NVIDIA Corporation for their GPU grant. </sub>
