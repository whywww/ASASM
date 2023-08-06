# LS-ASM
This repository provides the official open-source code of the following paper:

**Modeling off-axis diffraction with least-sampling angular spectrum method**\
Haoyu Wei*, Xin Liu*, Xiang Hao, Edmund Y. Lam, Yifan Peng\
[<ins>Paper</ins>](https://doi.org/10.1364/OPTICA.490223), [<ins>Project page</ins>](https://whywww.github.io/LSASM_page/) \
Correspondence: [Dr. Peng](https://www.eee.hku.hk/~evanpeng/) and [Prof. Lam](https://www.eee.hku.hk/~elam/). For implementation and experiment details please contact Haoyu (haoyu.wei97@gmail.com).

<img src="documents/principles.png" alt="principle" width="300"/>

## Quick start
This repository contains implementations of LS-ASM and Rayleigh-Sommerfeld algorithms, with spherical wave input and thin lens and diffuser modulations.

### Prerequisites
Create a conda environment from yml file:
```
conda env create -f environment.yml
```
If you are running on a GPU, please install a PyTorch version that matches the Cuda version on your machine.

### Config and Run
Configurations are in `main.py`.\
Run and find results in the `results` folder.
```
python main.py
```

## Performance
We display LS-ASM speedup along 0 - 20 degrees of incident angles.\
<img src="documents/results.png" alt="results" width="800">

Diffuser results closely resemble RS.\
<img src="documents/uniform-diffuser.png" alt="diffuser" width="800">

## Citation

If you use this code and find our work valuable, please cite our paper.
```
@article{Wei:23,
title       = {Modeling Off-Axis Diffraction with the Least-Sampling Angular Spectrum Method},
author      = {Haoyu Wei and Xin Liu and Xiang Hao and Edmund Y. Lam and Yifan Peng},
journal     = {Optica},
volume      = {10}, number = {7}, pages = {959--962},
publisher   = {Optica Publishing Group},
year        = {2023}, 
month       = {Jul}, 
doi         = {10.1364/OPTICA.490223}
}
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
