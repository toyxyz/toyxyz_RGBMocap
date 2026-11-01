# Research citations

RGB MoCap integrates the research projects below. Please credit the methods used
in your work and follow their upstream citation and license requirements.
These references do not change the separate licenses or permissions for code,
model weights, body data or input footage. Supporting models retain their own
upstream acknowledgements; see [third-party notices](../THIRD_PARTY_NOTICES.md).

## PEAR

[Paper](https://doi.org/10.1145/3799902.3811096) · [Official citation](https://github.com/Pixel-Talk/PEAR/blob/8e915f8c3dfae4063d66763e6296b2b0b02eb303/README.md#-citation)

```bibtex
@inproceedings{wu2026pear,
  author    = {Wu, Jiahao and Liu, Yunfei and Lin, Lijian and Zhu, Ye and Zhu, Lei and Li, Jingyi and Li, Yu},
  title     = {PEAR: Pixel-aligned Expressive humAn mesh Recovery},
  booktitle = {Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
  series    = {SIGGRAPH '26 Conference Papers},
  year      = {2026},
  pages     = {46:1--46:12},
  doi       = {10.1145/3799902.3811096},
  publisher = {Association for Computing Machinery}
}
```

RGB MoCap uses PEAR for live bone preview; its fixed-length display rig and
filtering are application features, not a claim to reproduce the paper's full
mesh workflow or reported throughput.

## ExMR

[Paper](https://arxiv.org/abs/2610.26213) · [Official citation](https://github.com/twehrbein/ExMR/blob/main/README.md)

```bibtex
@Article{wehrbein26exmr,
    author    = {Wehrbein, Tom and Rosenhahn, Bodo},
    title     = {{ExMR}: Extremity Motion Refinement for Monocular Human Motion Capture},
    journal   = {arXiv preprint arXiv:2610.26213},
    year      = {2026},
}
```

## FlowHMR

[Paper](https://arxiv.org/abs/2610.03691) · [Official citation](https://github.com/flowhmr/flowhmr/blob/main/README.md)

```bibtex
@misc{wang2026flowhmr,
      title={FlowHMR: Physically Plausible Motion Capture from Video},
      author={Zhanke Wang and Chengfeng Zhao and Qing Shuai and Jingzhong Lin and Heng Li and Zeyu Ling and Yuxin Wen and Jing Li and Di Kang and Chunchao Guo and Linchao Bao},
      year={2026},
      eprint={2610.03691},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.03691},
}
```

## GVHMR

[Paper](https://arxiv.org/abs/2409.06662) · [Official citation](https://github.com/zju3dv/GVHMR/blob/main/README.md)

```bibtex
@inproceedings{shen2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2024}
}
```

## MAMMA

[Paper](https://arxiv.org/abs/2506.13040) · [Official citation](https://github.com/cuevhv/mamma/blob/main/README.md)

```bibtex
@inproceedings{cuevas2026mamma,
  title     = {{MAMMA}: {Markerless Accurate Multi-person Motion Acquisition}},
  author    = {Cuevas Velasquez, Hanz and Yiannakidis, Anastasios and Shin, Soyong and Becherini, Giorgio and H{\"o}schle, Markus and Tesch, Joachim and Obersat, Taylor and Alexiadis, Tsvetelina and Halilaj, Eni and Black, Michael J.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## PromptHMR

[Paper](https://arxiv.org/abs/2504.06397) · [Official citation](https://github.com/yufu-wang/PromptHMR/blob/main/README.md)

```bibtex
@article{wang2025prompthmr,
  title={PromptHMR: Promptable Human Mesh Recovery},
  author={Wang, Yufu and Sun, Yu and Patel, Priyanka and Daniilidis, Kostas and Black, Michael J and Kocabas, Muhammed},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## FootMR

[Paper](https://doi.org/10.1109/3DV69130.2026.00088) · [Official citation](https://github.com/twehrbein/FootMR/blob/main/README.md)

```bibtex
@InProceedings{wehrbein26footmr,
    author    = {Wehrbein, Tom and Rosenhahn, Bodo},
    title     = {Improving 3D Foot Motion Reconstruction in Markerless Monocular Human Motion Capture},
    booktitle = {International Conference on 3D Vision (3DV)},
    year      = {2026},
    pages     = {871--881},
    doi       = {10.1109/3DV69130.2026.00088},
}
```

## HTD-Refine

[Paper](https://arxiv.org/abs/2605.26879) · [Official citation](https://github.com/ant-research/HTD-Refine/blob/main/README.md)

```bibtex
@inproceedings{wei2026htdrefine,
  title     = {Natural Human Motion Recovery by Aligning High-Order Temporal Dynamics from Monocular Videos},
  author    = {Wei, Dingkun and Shen, Zehong and Xia, Yan and Pavlakos, Georgios and Shen, Yujun and Zhou, Xiaowei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## HandFlow

[Paper](https://arxiv.org/abs/2607.11221) · [Official citation](https://github.com/mxxu00/HandFlow/blob/main/README.md)

```bibtex
@misc{xu2026handflow,
      title={HandFlow: Fully Generative 4D Hand Recovery with Flow Matching},
      author={Mingxi Xu and Bowen Duan and Yi Gu and Zhengyang Shen and Renjing Xu and Yutao Yue},
      year={2026},
      eprint={2607.11221},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.11221}
}
```

When using ExMR, also cite FootMR and the original GVHMR work.
Cite optional HTD-Refine and HandFlow when using those add-ons. The
application's use of Kimodo MotionCorrection does not imply use of the Kimodo
text-to-motion generator.


## GeoCalib (PEAR Camera Align and MAMMA background gravity)

[Original project and citation](https://github.com/cvg/GeoCalib) ·
[Code license](https://github.com/cvg/GeoCalib/blob/1dedb29f94fc11b896c5f572005978aeebccd67f/LICENSE) ·
[Weight license](https://github.com/cvg/GeoCalib/blob/1dedb29f94fc11b896c5f572005978aeebccd67f/LICENSE-WEIGHTS)

```bibtex
@inproceedings{veicht2024geocalib,
  author = {Alexander Veicht and Paul-Edouard Sarlin and Philipp Lindenberger and Marc Pollefeys},
  title = {{GeoCalib: Single-image Calibration with Geometric Optimization}},
  booktitle = {ECCV},
  year = {2024}
}
```

Camera Align also uses VGGT-Omega with its existing source/model acknowledgements.
Its background-floor estimation and fixed-rig multiview fusion are application
features, not an official PEAR multiview inference mode.
