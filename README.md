# HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis

Official experiment code for **HELIX** (ICML 2026).

> **HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation**
>
> Fengming Zhang, Wenjie Du, Huan Zhang, Ke Yu, Shen Qu

## Model Availability

The HELIX model is integrated into [PyPOTS](https://github.com/WenjieDu/PyPOTS):

```python
pip install pypots
from pypots.imputation import HELIX
```

Ablation variants (w/o FeatID, w/o Fusion, w/o Hybrid, w/o Sinusoidal PE) are available at: https://github.com/milaogou/PyPOTS/tree/add-helix-model

## Reproducing Experiments

All experiments were conducted on a Linux aarch64 HPC cluster with NVIDIA A100 GPUs. Tasks were submitted in parallel via SLURM. The shell scripts (`.sh`) in this repo reflect this cluster setup — adapt them to your own environment as needed.

Training logs are omitted due to their large size. Each script is self-explanatory; the overall workflow is described below.

See [`benchmark_code/`](benchmark_code/) for full details.

## Citation

```bibtex
@inproceedings{zhang2026helix,
  title={HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation},
  author={Zhang, Fengming and Du, Wenjie and Zhang, Huan and Yu, Ke and Qu, Shen},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

MIT