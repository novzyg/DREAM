<div align="center">

<img src="fig/drug.png" width="80"/>

# DREAM: A Unified Benchmark for Drug Recommendation

**DREAM** (**D**rug **R**ecommendation **E**valuation **A**cross **M**ultiple settings) unifies **22 medication recommendation models** under a single training, testing, and comparison framework.

Given a patient's diagnoses, procedures, and historical medication records, each model predicts the drug combination for the current visit — DREAM lets you train, evaluate, and compare them all with one consistent interface.

<img src="fig/DREAM.png" width="450"/>

[![Models](https://img.shields.io/badge/models-22-blue)](#models)
[![Datasets](https://img.shields.io/badge/datasets-MIMIC--III%20%7C%20MIMIC--IV%20%7C%20eICU-green)](#datasets)
[![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](#installation)
[![PyTorch](https://img.shields.io/badge/pytorch-2.12-EE4C2C?logo=pytorch&logoColor=white)](#installation)
[![Leaderboard](https://img.shields.io/badge/leaderboard-coming%20soon-orange)](#leaderboard)

</div>

## <div id="news">📰 News</div>

- **[2026-09]** DREAM v1 released with 22 models, 3 datasets (×3 drug-granularity variants each), unified CLI, and multi-GPU batch scheduling.

## <div id="highlights"><img src="fig/shi.png" width="60"/>Highlights</div>

- **22 models, one interface** — from RETAIN (2016) to MR-DTR (2025), every model shares the same train/test/run commands and config format.
- **3 benchmark datasets** — MIMIC-III, MIMIC-IV, and eICU, each processed at three drug-granularity levels (**all-level / ATC-3 / ATC-4**).
- **Unified CLI** — `train`, `test`, and `run` subcommands work for any model × dataset combination.
- **Multi-GPU batch scheduling** — queue dozens of (model, dataset) jobs; the scheduler dispatches them across GPUs based on live memory/utilization from `nvidia-smi`, with a real-time terminal dashboard.
- **Standardized evaluation** — Jaccard, F1, PRAUC, and DDI-rate metrics, multi-seed runs, and cross-seed summary reports out of the box.

## <div id="leaderboard"><img src="fig/link.png" width="40"/>Leaderboard</div>

🚧 The interactive leaderboard (all models × all datasets × all metrics) is under construction and will be published on GitHub Pages: **https://novzyg.github.io/DREAM**

## <div id="toc"><img src="fig/issac.gif" width="40"/> Table of Contents</div>

- [Highlights](#highlights)
- [Leaderboard](#leaderboard)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
- [Models](#models)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## <div id="installation"><img src="fig/env.png" width="30"/>Installation</div>

```bash
git clone https://github.com/novzyg/DREAM.git
```

Tested environment:

| Dependency | Version |
|:---|:---|
| Python | 3.12 |
| PyTorch | 2.12.0+cu130 |
| dill | 0.4.1 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| tqdm | 4.68.1 |

> **Package name note:** the code imports itself as the `drugrec_benchmark` package. Make the repo importable under that name, e.g. rename or symlink the clone, and run all commands from the **parent directory**:
>
> ```bash
> ln -s DREAM drugrec_benchmark     # or: mv DREAM drugrec_benchmark
> cd ..                              # parent of drugrec_benchmark/
> ```

Datasets are expected under `drugrec_benchmark/data/<dataset_name>/` (see [Datasets](#datasets)).

## <div id="quick-start"><img src="fig/use.png" width="40"/>Quick Start</div>

> All commands below run from the parent directory of `drugrec_benchmark/`.

**Train** a model (runs all seeds from its config, default `[1203, 2024, 42, 1234, 5678]`):

```bash
python -m drugrec_benchmark.scripts.train \
  --model safedrug \
  --dataset eicu_atc3 \
  --cuda 0
```

**Test** with automatic checkpoint discovery:

```bash
python -m drugrec_benchmark.scripts.test \
  --model safedrug \
  --dataset eicu_atc3 \
  --cuda 0
```

or point to a specific checkpoint:

```bash
python -m drugrec_benchmark.scripts.test \
  --model safedrug \
  --dataset eicu_atc3 \
  --cuda 0 \
  --checkpoint drugrec_benchmark/results/safedrug/eicu_atc3/seed_1203/run_xxx/models/epoch_xxx_jaccard_xxxx.pth
```

**Train + test + cross-seed summary** in one go:

```bash
python -m drugrec_benchmark.scripts.run \
  --model safedrug \
  --dataset eicu_atc3 \
  --cuda 0
# → drugrec_benchmark/results/safedrug/eicu_atc3/reports/cross_seed_summary_<timestamp>.json
```

Outputs are organized as:

```text
results/<model>/<dataset>/seed_<seed>/run_<time>/     # weights (.pth), logs, per-run report JSON
results/<model>/<dataset>/reports/cross_seed_summary_<timestamp>.json   # mean ± std across seeds
```

## <div id="advanced-usage"><img src="fig/commend.png" width="35"/>Advanced Usage</div>

The unified entry point `main.py` wraps the scripts above and adds batch execution and GPU-aware scheduling.

<details>
<summary><b>Single task via the unified entry</b></summary>

```bash
python -m drugrec_benchmark.main train --model safedrug --dataset eicu_atc3 --cuda 0
python -m drugrec_benchmark.main test   --model safedrug --dataset eicu_atc3 --cuda 0
python -m drugrec_benchmark.main run    --model safedrug --dataset eicu_atc3 --cuda 0
```
</details>

<details>
<summary><b>Batch: multiple models × datasets across GPUs</b></summary>

```bash
python -m drugrec_benchmark.main run \
  --models safedrug,gamenet,retain \
  --datasets eicu_atc3,mimic-iii_atc3 \
  --cuda-list 0,1 \
  --parallel 2
```

This creates 6 tasks (3 models × 2 datasets) and runs 2 of them concurrently on GPU 0 and GPU 1.
</details>

<details>
<summary><b>GPU-aware scheduling</b></summary>

```bash
python -m drugrec_benchmark.main run \
  --models cognet,molerec \
  --datasets eicu_all,eicu_atc3,eicu_atc4 \
  --cuda-list 0,1,2,3 \
  --parallel 0 \
  --max-procs-per-gpu 3 \
  --worker-threads 1 \
  --gpu-launch-interval 5 \
  --gpu-mem-threshold 0.9 \
  --gpu-util-threshold 100 \
  --poll-interval 5 \
  --no-dashboard
```

Behavior: a new task is launched on a GPU only when its memory usage is below **90%** and utilization below **100%**, with at least **5 s** between consecutive launches on the same GPU; when no GPU qualifies, the scheduler re-checks every **5 s**. Each worker process is limited to **1 thread**, and the live terminal dashboard is disabled.

Key scheduler options (see `python -m drugrec_benchmark.main run --help` for the full list):

| Option | Meaning |
|:---|:---|
| `--parallel N` | Global max concurrent tasks (`0` = auto: `#GPUs × max-procs-per-gpu`) |
| `--max-procs-per-gpu N` | Max concurrent tasks per GPU (`<=0` = no cap) |
| `--gpu-mem-threshold R` | Launch only if GPU memory ratio ≤ R |
| `--gpu-util-threshold U` | Launch only if GPU utilization ≤ U% |
| `--gpu-launch-interval S` | Min seconds between launches on the same GPU |
| `--max-starting-tasks N` | Max tasks in STARTING warmup at once (`<=0` = #GPUs) |
| `--task-warmup-seconds S` | How long a STARTING task counts against the quota above |
| `--worker-threads N` | CPU/BLAS/PyTorch threads per worker (`<=0` = unchanged) |
| `--no-dashboard` | Disable the rich live dashboard (plain heartbeat logs instead) |
| `--scheduler-dir DIR` | Where scheduler state/logs are written |
</details>

<details>
<summary><b>Summarize all experimental results</b></summary>

```bash
python -m drugrec_benchmark.scripts.export_results_summary
```
</details>

Model hyperparameters live in `configs/<model>.yaml` (with `base_config.yaml` as the shared base) — edit these to tune a model.

## <div id="models"><img src="fig/models.png" width="30"/>Models</div>

DREAM integrates **22 models** spanning a decade of medication recommendation research. The **Config name** column is the value to pass to `--model`.

| Year | Model | Config name | Title | Venue | Links |
|:----:|:---|:---|:---|:---|:---|
| 2016 | RETAIN | `retain` | RETAIN: An Interpretable Predictive Model for Healthcare Using Reverse Time Attention Mechanism | NeurIPS | [Paper](https://arxiv.org/abs/1608.05745) · [Code](https://github.com/mp2893/retain) |
| 2017 | Leap | `leap` | LEAP: Learning to Prescribe Effective and Safe Treatment Combinations for Multimorbidity | KDD | [Paper](https://dl.acm.org/doi/abs/10.1145/3097983.3098109) · [Code](https://github.com/neozhangthe1/AutoPrescribe) |
| 2019 | GAMENet | `gamenet` | GAMENet: Graph Augmented Memory Networks for Recommending Medication Combination | AAAI | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/3905) · [Code](https://github.com/sjy1203/GAMENet) |
| 2019 | CompNet | `compnet` | Order-free Medicine Combination Prediction with Graph Convolutional Reinforcement Learning | CIKM | [Paper](https://dl.acm.org/doi/abs/10.1145/3357384.3357965) · [Code](https://github.com/irlab-sdu/CompNet) |
| 2021 | MICRON | `micron` | Change Matters: Medication Change Prediction with Recurrent Residual Networks | IJCAI | [Paper](https://arxiv.org/abs/2105.01876) · [Code](https://github.com/ycq091044/MICRON) |
| 2021 | ARMR | `armr` | Adversarially Regularized Medication Recommendation Model with Multi-hop Memory Network | KAIS | [Paper](https://dl.acm.org/doi/10.1007/s10115-020-01513-9) · [Code](https://github.com/yanda-wang/ARMR) |
| 2021 | SafeDrug | `safedrug` | Dual Molecular Graph Encoders for Recommending Effective and Safe Drug Combinations | IJCAI | [Paper](https://arxiv.org/abs/2105.02711) · [Code](https://github.com/ycq091044/SafeDrug) |
| 2021 | PREMIER | `premier` | Personalizing Medication Recommendation with a Graph-based Approach | ACM TOIS | [Paper](https://dl.acm.org/doi/abs/10.1145/3488668) |
| 2022 | COGNet | `cognet` | COGNet: Conditional Generation Net for Medication Recommendation | TheWebConf | [Paper](https://dl.acm.org/doi/abs/10.1145/3485447.3511936) · [Code](https://github.com/BarryRun/COGNet) |
| 2022 | 4SDrug | `4sdrug` | 4SDrug: Symptom-based Set-to-set Small and Safe Drug Recommendation | KDD | [Paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539089) · [Code](https://github.com/Melinda315/4SDrug) |
| 2022 | DrugRec | `drugrec_all` / `drugrec_nosym` | Debiased, Longitudinal and Coordinated Drug Recommendation through Multi-Visit Clinic Records | NeurIPS | [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b295b3a940706f431076c86b78907757-Abstract-Conference.html) · [Code](https://github.com/ssshddd/DrugRec) |
| 2023 | DRecHGR | `drechgr` | Enhancing Drug Recommendations via Heterogeneous Graph Representation Learning in EHR Networks | IEEE TKDE | [Paper](https://ieeexplore.ieee.org/abstract/document/10302298/) · [Code](https://github.com/HjZ1998/DRecHGR-master) |
| 2023 | MoleRec | `molerec` | MoleRec: Combinatorial Drug Recommendation with Substructure-Aware Molecular Representation Learning | TheWebConf | [Paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583872) · [Code](https://github.com/yangnianzu0515/MoleRec) |
| 2023 | MedRec | `kamtl_medrec` | Knowledge-Enhanced Attributed Multi-Task Learning for Medicine Recommendation | ACM TOIS | [Paper](https://dl.acm.org/doi/abs/10.1145/3527662) |
| 2023 | OntoPath | `ontopath` | Ontology-Aware Prescription Recommendation in Treatment Pathways Using Multi-Evidence Healthcare Data | ACM TOIS | [Paper](https://dl.acm.org/doi/abs/10.1145/3579994) · [Code](https://github.com/zyao237/ontopath) |
| 2023 | Carmen | `carmen` | Context-Aware Safe Medication Recommendations with Molecular Graph and DDI Graph Embedding | AAAI | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/25861) · [Code](https://github.com/bit1029public/Carmen) |
| 2023 | REFINE | `refine` | REFINE: A Fine-Grained Medication Recommendation System Using Deep Learning and Personalized Drug Interaction Modeling | NeurIPS | [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4b7439a4ab0b8e4bcb4e2412c6a10a58-Abstract-Conference.html) |
| 2024 | VITA | `vita` | VITA: 'Carefully Chosen and Weighted Less' Is Better in Medication Recommendation | AAAI | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28704) · [Code](https://github.com/jhheo0123/VITA) |
| 2024 | RAREMed | `raremed` | Leave No Patient Behind: Enhancing Medication Recommendation for Rare Disease Patients | SIGIR | [Paper](https://dl.acm.org/doi/abs/10.1145/3626772.3657785) · [Code](https://github.com/zzhUSTC2016/RAREMed) |
| 2025 | SSPNet | `sspnet` | SSPNet: Leveraging Robust Medication Recommendation with History and Knowledge | IJCAI | [Paper](https://www.ijcai.org/proceedings/2025/1052) · [Code](https://github.com/ResearchGroupHdZhang/SSPNet) |
| 2025 | TEMPT | `tempt` | A Contrastive Pretrain Model with Prompt Tuning for Multi-center Medication Recommendation | ACM TOIS | [Paper](https://dl.acm.org/doi/abs/10.1145/3706631) · [Code](https://github.com/liuqidong07/TEMPT) |
| 2025 | MR-DTR | `mrdtr` | Time-aware Medication Recommendation via Intervention of Dynamic Treatment Regimes | TheWebConf | [Paper](https://dl.acm.org/doi/abs/10.1145/3696410.3714533) · [Code](https://github.com/liyifo/MR-DTR) |

> **Note:** MR-DTR uses staged training — run it via `python -m drugrec_benchmark.scripts.train_mrdtr` instead of `scripts.train`.

## <div id="datasets"><img src="fig/datasets.png" width="30"/>Datasets</div>

Three public EHR datasets are used, each processed into patient-level longitudinal sequences (diagnoses, procedures, medications) at three drug-granularity levels:

| Dataset | Variants | Description |
|:---|:---|:---|
| [MIMIC-III](https://mimic.mit.edu/docs/iii/) | `mimic-iii_all` / `mimic-iii_atc3` / `mimic-iii_atc4` | Critical care EHR, 2001–2012 |
| [MIMIC-IV](https://mimic.mit.edu/docs/iv/) | `mimic-iv_all` / `mimic-iv_atc3` / `mimic-iv_atc4` | Updated MIMIC, 2008–2019 |
| [eICU](https://eicu.mit.edu/) | `eicu_all` / `eicu_atc3` / `eicu_atc4` | Multi-center ICU EHR |

Granularity levels:

- **all-level** — exact standardized drug codes (finest)
- **ATC-4** — intermediate drug categories
- **ATC-3** — broader therapeutic categories (coarsest)

Each dataset directory under `data/` contains shared files — `records_final.pkl` (patient sequences), `voc_final.pkl` (vocabularies), `ddi_A_final.pkl` (DDI adjacency), `ehr_adj_final.pkl` (EHR co-occurrence) — plus model-specific extras such as `ddi_mask_H.pkl` (DDI mask) and `SMILES.pkl` (molecular structures).

## <div id="project-structure"><img src="fig/structure.png" width="40"/>Project Structure</div>

```text
DREAM/
├── main.py                # Unified CLI entry: batch execution + GPU-aware scheduler
├── configs/               # Per-model YAML configs (+ shared base_config.yaml)
├── core/                  # Training loop, evaluation, metrics, I/O
├── models/                # Model implementations + registry (@register_model)
│   └── base_model.py      #   shared model interface (forward + compute_loss)
├── scripts/               # train.py / test.py / run.py / train_mrdtr.py
│                          # + export_results_summary.py
├── utils/                 # Config loading, data processing, logging, metrics
│   └── modules/           #   reusable modules (e.g., graph neural networks)
├── data/                  # Processed datasets: data/<dataset_name>/*.pkl
└── results/               # Checkpoints, logs, and report JSONs per run
```

Execution flow:

```text
parse CLI args → load YAML config → load dataset → build model (registry)
→ train / evaluate / test → compute metrics → save logs, weights, reports
```

<div align="center">
<img src="fig/framework1.png"/>
</div>

## <div id="known-limitations">⚠️ Known Limitations</div>

- The `--config` argument is declared in `scripts/train.py` and `scripts/test.py` but is **not currently wired into `load_config`**, so custom config overrides do not take effect yet. Model hyperparameters come from `configs/*.yaml` and `base_config.yaml`.

## <div id="roadmap">🗺️ Roadmap</div>

- [ ] Interactive leaderboard on GitHub Pages (22 models × 9 dataset variants)
- [ ] Wire up `--config` override support
- [ ] More models and datasets

## <div id="citation">📖 Citation</div>

If you use DREAM in your research, please cite:

```bibtex
@misc{zhen2026dream,
  title  = {EHR-Based Medication Recommendation: A Stage-Oriented Survey and Unified Benchmark},
  author = {Zhen, Yeguang and Wang, Shoujin and Li, Yishuo and Hu, Liang and Lian, Defu and Wang, Cong and Lu, Wenpeng},
  year   = {2026},
  url    = {https://github.com/novzyg/DREAM}
}
```

## <div id="acknowledgements">🙏 Acknowledgements</div>

DREAM builds on the excellent work of the medication recommendation community — we thank the authors of all [integrated models](#models) for open-sourcing their code, and the [MIT-LCP](https://mimic.mit.edu/) team for maintaining the MIMIC/eICU datasets.
