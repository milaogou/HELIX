<a id="HELIX"></a>

<p align="center">
    <b> <a href="https://arxiv.org/abs/xxxx.xxxxx">HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation</a></b>
</p>

<p align="center">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-v3-E97040?logo=python&logoColor=white" alt="Python v3">
    </a>
    <a>
        <img src="https://img.shields.io/badge/PyTorch-❤️-F8C6B5?logo=pytorch&logoColor=white" alt="powered by PyTorch">
    </a>
    <a href="https://github.com/milaogou/HELIX/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-3C7699?logo=opensourceinitiative&logoColor=white" alt="MIT License">
    </a>
    <a href="https://icml.cc/virtual/2026/">
        <img src="https://img.shields.io/badge/ICML-2026-blue" alt="ICML 2026">
    </a>
</p>

The official experiment code for the paper
**HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation**,
which has been accepted by the
*[International Conference on Machine Learning (ICML)](https://icml.cc/) 2026*.

> Fengming Zhang, Wenjie Du, Huan Zhang, Ke Yu, Shen Qu.
> HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation.
> In Proceedings of the International Conference on Machine Learning (ICML), 2026.

🤗 If you find HELIX helpful to your work, please [cite it](#-citing-helix) and ⭐️star this repository to help others discover it. Thank you!

## ❖ Motivation and Highlights

⦿ **`Motivation`**: Existing time series imputation methods often treat temporal and feature dimensions in isolation,
limiting their ability to capture the complex inter-dependencies in multivariate incomplete data. HELIX is proposed
to bridge this gap by jointly encoding temporal position, feature identity, and cross-dimensional interactions
within a unified framework.

⦿ **`Highlights`**:
- **Learnable Feature Identity (FeatID)**: A learnable embedding module that assigns each feature a unique identity
  representation, enabling the model to distinguish and specialize across heterogeneous variables.
- **Hybrid Encoding**: Combines sinusoidal positional encoding with learnable identity embeddings, capturing both
  absolute temporal positions and feature-specific characteristics simultaneously.
- **Cross-dimensional Synthesis**: A fusion mechanism that synthesizes information across the temporal and feature
  dimensions, allowing bidirectional information flow for more accurate imputation.
- **State-of-the-art Performance**: HELIX achieves competitive or superior results on multiple benchmark datasets
  for time series imputation.

## ❖ Model Availability

> [!IMPORTANT]
> <a href="https://github.com/WenjieDu/PyPOTS"><img src="https://pypots.com/figs/pypots_logos/PyPOTS/logo_FFBG.svg" width="128" align="right"></a>
> **📣 HELIX is now available in [PyPOTS](https://github.com/WenjieDu/PyPOTS)!**
>
> You can easily train and use HELIX for time series imputation with just a few lines of code:

```bash
pip install pypots
```

```python
from pypots.imputation import HELIX

# initialize the model
helix = HELIX(
    n_steps=48,        # number of time steps
    n_features=37,     # number of features
    # ... (refer to PyPOTS documentation for full parameter list)
    epochs=100,
)

# train
helix.fit(train_set, val_set)

# impute
imputation = helix.impute(test_set)
```

**Ablation variants** (w/o FeatID, w/o Fusion, w/o Hybrid, w/o Sinusoidal PE) are available at:
👉 https://github.com/milaogou/PyPOTS/tree/add-helix-model

## ❖ Reproducing Experiments

All experiments were conducted on a **Linux aarch64 HPC cluster** with **NVIDIA A100 GPUs**.
Tasks were submitted in parallel via **SLURM**. The shell scripts (`.sh`) in this repo reflect this cluster setup —
please adapt them to your own environment as needed.

Training logs are omitted due to their large size. Each script is self-explanatory;
the overall workflow is described below.

### Step-by-step Workflow

1. **Prepare datasets**: Follow the data preprocessing instructions in `benchmark_code/`.
2. **Configure experiments**: Adjust hyperparameters and paths in the config/script files as needed.
3. **Run training**: Submit the `.sh` scripts via SLURM or run them directly.
4. **Evaluate**: Results will be saved automatically. Compare with baselines using the provided evaluation scripts.

See [`benchmark_code/`](benchmark_code/) for full details.

## ❖ Repository Structure

```
HELIX/
├── benchmark_code/       # Full experiment code, configs, and evaluation scripts
│   ├── ...
│   └── ...
├── LICENSE               # MIT License
└── README.md             # This file
```

## ❖ Citing HELIX

If HELIX is helpful to your research, please cite our paper as below and ⭐️star this repository.
🤗 Thank you!

```bibtex
@inproceedings{zhang2026helix,
    title   = {{HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation}},
    author  = {Zhang, Fengming and Du, Wenjie and Zhang, Huan and Yu, Ke and Qu, Shen},
    booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
    year    = {2026}
}
```

or

> Fengming Zhang, Wenjie Du, Huan Zhang, Ke Yu, and Shen Qu.
> HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation.
> In Proceedings of the International Conference on Machine Learning (ICML), 2026.

### 🔥 If you use PyPOTS in your research, please also cite:

```bibtex
@article{du2023pypots,
    title   = {{PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series}},
    author  = {Wenjie Du},
    journal = {arXiv preprint arXiv:2305.18811},
    year    = {2023}
}
```

## ❖ Acknowledgments

Thanks to all co-authors for their valuable contributions.
Thanks to the [PyPOTS](https://github.com/WenjieDu/PyPOTS) community for providing the ecosystem and infrastructure.
Thanks to all reviewers for helping improve the quality of this paper.

### ✨ Stars/forks/issues/PRs are all welcome!

## ❖ License

This project is released under the [MIT License](LICENSE).

## ❖ Contact

If you have any questions or are interested in collaboration,
please feel free to open an issue or contact us. 😊
