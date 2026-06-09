# DREAM 项目说明

## 一、功能

DREAM 是一个用于药物推荐任务的深度学习项目。项目根据患者的诊断信息、手术信息和历史用药记录，预测当前就诊中可能需要使用的药物组合。

本项目主要功能：

1. 支持药物推荐模型的训练。
2. 支持已训练模型的测试。
3. 支持不同模型实验结果的对比。
4. 支持在多个 eICU 数据集版本上运行实验。
5. 支持计算 Jaccard、F1、PRAUC、DDI Rate 等评价指标。
6. 支持保存训练日志、测试结果和模型参数文件。
7. 通过 `main.py` 统一管理训练、测试和完整实验流程。

项目包含的主要模型：

```text
SafeDrug
GAMENet
Retain
LEAP
MICRON
MoleRec
CogNet
VITA
4SDrug
Carmen
CompNet
```

项目包含的数据集：

```text
eicu_all
eicu_atc3
eicu_atc4
```

---

## 二、实现方法

项目由数据文件、模型文件、配置文件、训练模块和测试模块组成。程序运行时，先读取数据和配置参数，再创建对应模型，最后完成训练或测试任务。
### 1. 项目结构

DREAM-main/
├── README.md
├── RUN_COMMANDS.md
├── main.py
│
├── configs/
│   ├── base_config.yaml
│   ├── 4sdrug.yaml
│   ├── armr.yaml
│   ├── carmen.yaml
│   ├── cognet.yaml
│   ├── compnet.yaml
│   ├── drechgr.yaml
│   ├── gamenet.yaml
│   ├── kamtl_medrec.yaml
│   ├── leap.yaml
│   ├── micron.yaml
│   ├── molerec.yaml
│   ├── ontopath.yaml
│   ├── premier.yaml
│   ├── raremed.yaml
│   ├── refine.yaml
│   ├── retain.yaml
│   ├── safedrug.yaml
│   ├── sspnet.yaml
│   └── vita.yaml
│
├── core/
│   ├── evaluator.py
│   ├── io.py
│   └── trainer.py
│
├── data
│   ├── eicu_all/
│   ├── eicu_atc3/
│   └── eicu_atc4/
│
├── models/
│   ├── base_model.py
│   ├── registry.py
│   ├── armr.py
│   ├── carmen.py
│   ├── cognet.py
│   ├── compnet.py
│   ├── drechgr.py
│   ├── foursdrug.py
│   ├── gamenet.py
│   ├── kamtl_medrec.py
│   ├── leap.py
│   ├── micron.py
│   ├── molerec.py
│   ├── ontopath.py
│   ├── premier.py
│   ├── raremed.py
│   ├── refine.py
│   ├── retain.py
│   ├── safedrug.py
│   ├── sspnet.py
│   └── vita.py
│
├── scripts/
│   ├── train.py
│   ├── test.py
│   ├── run.py
│   └── export_results_summary.py
│
└── utils/
    ├── build_mpnn.py
    ├── dataset_utils.py
    ├── load_config.py
    ├── logs.py
    ├── metrics.py
    └── modules/
        ├── GNNConv.py
        └── GNNs.py

### 2. 数据读取

数据存放在 `data` 文件夹中，主要为已经处理好的 `.pkl` 文件。

```text
data/
├── eicu_all/
├── eicu_atc3/
└── eicu_atc4/
```

主要数据文件：

```text
records_final.pkl      患者就诊记录
voc_final.pkl          诊断、手术、药物词表
ddi_A_final.pkl        药物相互作用矩阵
ehr_adj_final.pkl      药物共现关系
SMILES.pkl             药物分子信息
```

程序会读取这些文件，并将患者记录整理成模型可以接收的输入格式。数据处理相关代码位置：

```text
utils/dataset_utils.py
core/io.py
```

### 3. 模型构建

模型代码集中在 `models` 文件夹中，不同模型对应不同的 Python 文件。

主要模型文件：

```text
models/safedrug.py
models/gamenet.py
models/retain.py
models/leap.py
models/micron.py
models/molerec.py
```

模型的输入主要包括患者诊断、手术和历史用药信息，输出为当前就诊的药物推荐结果。

模型中常用函数：

```text
forward()       完成模型前向计算
compute_loss()  计算训练损失
save_model()    保存模型参数
load_model()    读取模型参数
```

项目使用 `models/registry.py` 统一管理模型名称和模型类。运行时输入模型名称后，程序会根据名称调用对应模型。

### 4. 配置管理

配置文件存放在 `configs` 文件夹中，文件格式为 YAML。

主要配置文件：

```text
configs/base_config.yaml
configs/safedrug.yaml
configs/gamenet.yaml
configs/retain.yaml
configs/micron.yaml
```

配置文件主要记录以下内容：

```text
学习率
batch size
epoch 数量
模型参数
数据集路径
结果保存路径
```

### 5. 模型训练

模型训练可以通过 `main.py` 或 `scripts/train.py` 启动。

训练命令如下：

```bash
python -m drugrec_benchmark.main train --model safedrug --dataset eicu_atc3 --cuda 0
```

训练过程：

```text
读取配置文件
读取数据集
创建模型
输入训练数据
计算损失
反向传播
更新参数
保存模型和日志
```

训练相关代码位于：

```text
core/trainer.py
```

### 6. 模型测试

模型测试可以通过 `main.py` 或 `scripts/test.py` 启动。

测试命：

```bash
python -m drugrec_benchmark.main test --model safedrug --dataset eicu_atc3 --cuda 0
```

测试过程：

```text
加载测试数据
加载模型参数
进行药物预测
计算评价指标
保存测试结果
```

测试和评价相关代码位于：

```text
core/evaluator.py
utils/metrics.py
```

### 7. 训练与测试连续运行

`run` 命令可以先完成训练，再自动进行测试。

```bash
python -m drugrec_benchmark.main run --model safedrug --dataset eicu_atc3 --cuda 0
```

## 三、环境

项目需要 Python 环境，并依赖 PyTorch 以及常用的数据处理库。

### 1. Python 版本

建议使用以下版本：

```text
Python 3.10
```

创建 Conda 环境：

```bash
conda create -n drugrec python=3.10 -y
conda activate drugrec
```

### 2. 依赖库

安装基础依赖：

```bash
pip install numpy scipy pandas scikit-learn pyyaml dill rich rdkit-pypi
```

安装 PyTorch：

```bash
pip install torch torchvision torchaudio
```

使用 NVIDIA 显卡时，需要安装与 CUDA 版本匹配的 PyTorch。

部分模型可能还需要图神经网络相关库：

```bash
pip install torch-geometric ogb dgl torch-scatter
```

这些库对 PyTorch 和 CUDA 版本有要求，安装时需要注意版本对应关系。

### 3. 推荐环境

```text
系统：Windows 或 Linux
Python：3.10
深度学习框架：PyTorch
显卡：NVIDIA GPU
数据集：eicu_all、eicu_atc3、eicu_atc4
```
