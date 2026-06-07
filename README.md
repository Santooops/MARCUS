# MARCUS
MARCUS: Missing-Aware Region Representation with Contextual Urban Signals for Rent Prediction

# MARCUS — NYC (KDD2026)

Environment and hardware used to train and evaluate MARCUS on the NYC dataset.

## Environment

Conda env: **`marcusbase`** (`/scratch/Renee/miniconda3/envs/marcusbase`)

| Package | Version |
|---|---|
| Python | 3.10.19 |
| PyTorch | 2.10.0 (+cu128) |
| CUDA (torch build) | 12.8 |
| cuDNN | 9.10.2 |
| torch-geometric | 2.7.0 |
| torch-scatter / sparse / cluster | 2.1.2 / 0.6.18 / 1.6.3 |
| numpy | 2.2.5 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| scipy | 1.15.3 |
| geopandas / shapely | 1.1.2 / 2.1.2 |
| matplotlib | 3.10.8 |
| networkx | 3.4.2 |

## Hardware

| | |
|---|---|
| GPU | 2× NVIDIA Tesla V100-PCIE-32GB (training uses `cuda:0`) |
| NVIDIA driver | 550.100 |
| OS | Linux (RHEL 8.8, kernel 4.18) |

## Reproducibility note

 Results are not guaranteed to match across different GPU architectures or PyTorch versions.
GPU： Tesla V100-PCIE-32GB
Python 3.10.19



Overview Framework:

<img width="836" height="474" alt="截屏2026-06-07 19 52 19" src="https://github.com/user-attachments/assets/2b81661b-d4f2-4bee-97df-8d09df93bba6" />

Results:

<img width="429" height="288" alt="截屏2026-06-07 19 53 06" src="https://github.com/user-attachments/assets/2ac26023-4e1f-48d5-84f8-0b0cba8d80f5" />

Ablation Study:

<img width="449" height="271" alt="截屏2026-06-07 19 53 34" src="https://github.com/user-attachments/assets/993a5bfd-378c-4f68-bef2-1026a5b57e05" />
<img width="398" height="211" alt="截屏2026-06-07 19 53 44" src="https://github.com/user-attachments/assets/7a87b250-7e2b-40db-a285-890aa5e39dd0" />

