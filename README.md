# AQuA: Learn 3D VQA Better with Active Selection and Reannotation

<p align="center">
    <a href='https://arxiv.org/pdf/2507.04630'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=googledocs&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://arxiv.org/html/2507.04630'>
      <img src='https://img.shields.io/badge/Paper-HTML-blue?style=plastic&logo=html5&logoColor=blue' alt='Paper HTML'>
    </a>
    <a href='https://arxiv.org/abs/2507.04630'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://drive.google.com/drive/folders/148fTyjCJW_fT3cs4_ju0pKrhZPV0YViu?usp=sharing'>
      <img src='https://img.shields.io/badge/Model-Checkpoints-orange?style=plastic&logo=Google%20Drive&logoColor=orange' alt='Checkpoints'>
    </a>
</p>


<p align="center">
  <a href='https://fz-zsl.github.io/'>Shengli Zhou</a>,
  <a href='http://www.csyangliu.com/'>Yang Liu</a>,
  <a href='https://www.sustech.edu.cn/en/faculties/zhengfeng.html'>Feng Zheng</a>📧
</p>

This repository is the official implementation of the ACM MM 2025 paper "Learn 3D VQA Better with Active Selection and Reannotation".

In our paper, we conduct comparative experiments (i.e., the "Lazy Oracle Experiment" and the "Diligent Oracle Experiment") and an ablation study to validate our methods. This repository contains the code for experiments.

## ScanQA

For ScanQA, we modify the code from the [official implementation of ScanQA](https://github.com/ATR-DBI/ScanQA). Please refer to the official repository for dependency installation and data preparation.

### Model Training

```sh
python scripts/train.py --use_color --tag <tag_name> --AL_mode <AL_strategy> [--AL_oracle]
```

Options:

- `--AL_mode` sets the strategy used for active learning, which includes `[random, entropy, infogain, variance]`.
- Adding `--AL_oracle` enables the usage of Hierarchical Reannotation Strategy; otherwise, the "lazy oracle" is applied.
- For more training options, please run `python scripts/train.py -h`.

### Model Evaluation

- Evaluation of trained ScanQA models with the val dataset:

  ```sh
  python scripts/eval.py --folder <folder_name> --qa --force
  ```

  `<folder_name>` corresponds to the folder under `outputs/` with the `timestamp + <tag_name>`.

- Scoring with the val dataset:

  ```sh
  python scripts/score.py --folder <folder_name>
  ```

- Prediction with the test dataset:

  ```sh
  python scripts/predict.py --folder <folder_name> --test_type <test_type>
  ```

  `<test_type>` includes `test_w_obj` and `test_wo_obj`.

## 3D-VisTA

For 3D-VisTA, we modify the code from the [official implementation of 3D-VisTA](https://github.com/3d-vista/3D-VisTA). Please refer to the official repository for dependency installation and data preparation. Before running the model, path configurations in line 3 of `./dataset/path_config.py` and line 5 of `./model/language/lang_encoder.py` needs to be modified.

### Model Training

```sh
python3 run.py --config project/vista/train_scanqa_config.yml
```

Options: in `train_scanqa_config.yml`,

- `AL_mode` sets the strategy used for active learning, which includes `[random, variance]`.
- `AL_oracle` represents the usage of Hierarchical Reannotation Strategy.

### Model Evaluation

```sh
python3 run.py --config project/vista/eval_scanqa_config.yml
```

## Acknowledgement

We would like to thank the authors of [ScanQA](https://github.com/ATR-DBI/ScanQA) and [3D-VisTA](https://github.com/3d-vista/3D-VisTA) for their open-source release.

## Citation

If you find this project useful in your research, please consider citing:

```bib
@misc{zhou2025learn3dvqabetter,
      title={Learn 3D VQA Better with Active Selection and Reannotation}, 
      author={Shengli Zhou and Yang Liu and Feng Zheng},
      year={2025},
      eprint={2507.04630},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.04630}, 
}
```
