# Fairness Analysis of CLIP-Based Foundation Models for X-Ray Image Classification

This repository contains the implementation for the ISBI 2025 paper *"Fairness Analysis of CLIP-Based Foundation Models for X-Ray Image Classification"* ([arXiv:2501.19086](https://arxiv.org/abs/2501.19086)). The project evaluates the fairness and performance of CLIP-based foundation models (CLIP, MedCLIP, and BiomedCLIP) on X-ray image classification tasks. We implement three fine-tuning approaches: **Linear Probe**, **MLP**, and **LoRA** (Low-Rank Adaptation).

## Setup Instructions

### 1. Create the Environment

```bash
conda env create -f environment.yaml
conda activate base_clip

# Install CLIP and MedCLIP
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/RyanWangZf/MedCLIP.git
```

### 2. Running the Fine-Tuning Experiments
an example command to run LoRA fine-tuning with CLIP (B/16 variant):

```bash
python main.py --model_type clip --variant B16 --mode lora
```

You can modify these arguments according to `parse_arguments` in `utils.py`.

### 3. Calculating Metrics

```bash
python calculate_metrics.py
```

## Citation
If you use this code in your research, please cite our paper:

```bibtex
@article{sun2025fairness,
  title={Fairness Analysis of CLIP-Based Foundation Models for X-Ray Image Classification},
  author={Sun, Xiangyu and Zou, Xiaoguang and Wu, Yuanquan and Wang, Guotai and Zhang, Shaoting},
  journal={arXiv preprint arXiv:2501.19086},
  year={2025}
}
```


## License
This project is licensed under the [MIT License](LICENSE) (or specify your preferred license).

---
