
<div align="center">

# 🏔️ OptiGDM

**Optical Imagery-Guided Diffusion Model for Structure-Preserving DEM Void Filling**


[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

</div>

---

## Overview

Digital Elevation Models (DEMs) are critical for geoscience and engineering applications, but **voids caused by data acquisition limitations** often compromise their completeness and usability.

- Traditional interpolation methods tend to **oversmooth** terrain
- GAN-based methods may suffer from **artifacts** and **unstable training**

**OptiGDM** addresses these limitations by introducing optical imagery as cross-modal guidance into the denoising diffusion process. By leveraging the strong spatial-structural correspondence between optical imagery and terrain morphology, the model translates visual semantic cues into coherent elevation structures — enabling **high-quality reconstruction** of missing DEM regions.

---

## Getting Started

### Training

To train OptiGDM from scratch, run:

```bash
python train.py --conf_path confs/train_config.yml
```

### Testing

To test the trained model or perform DEM void filling, run:

```bash
python test.py --conf_path confs/test_config.yml
```

---

## Acknowledgements

This project is inspired by and builds upon the following excellent works:

### RePaint

> **RePaint: Inpainting using Denoising Diffusion Probabilistic Models**  
> Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, Luc Van Gool  
> *Computer Vision Lab, ETH Zürich, Switzerland*

Paper: [https://arxiv.org/abs/2201.09865](https://arxiv.org/abs/2201.09865)

### OpenAI Guided Diffusion

Repository: [https://github.com/openai/guided-diffusion](https://github.com/openai/guided-diffusion)

---

We sincerely thank the authors for their excellent contributions to diffusion-based image generation, guided diffusion, and image inpainting.
```

