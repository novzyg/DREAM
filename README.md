# DREAM 项目说明

## 一、项目功能

DREAM 是一个用于药物推荐任务的深度学习项目。项目根据患者的诊断信息、手术信息和历史用药记录，预测当前就诊中可能使用的药物组合。

项目主要完成以下功能：

1. 训练药物推荐模型。
2. 测试已经训练好的模型。
3. 对比不同模型的实验结果。
4. 在不同版本的 eICU 数据集上运行实验。
5. 计算 Jaccard、F1、PRAUC、DDI Rate 等评价指标。
6. 保存训练日志、测试结果和模型参数文件。
7. 通过 `main.py` 统一管理训练、测试和完整实验流程。

## 二、项目框架

DREAM 的整体框架可以理解为五个部分：数据、配置、模型、训练测试流程、工具函数。

程序运行时，一般按照下面的顺序执行：

```text
读取命令参数
↓
读取配置文件
↓
读取数据集
↓
创建指定模型
↓
训练模型或加载模型
↓
计算评价指标
↓
保存日志、模型参数和测试结果
```

其中，`main.py` 是项目的统一入口。用户通过命令指定模型名称、数据集名称和运行模式，程序会自动调用对应的配置文件、数据文件和模型文件。

## 三、各文件夹实现的功能

| 文件或文件夹           | 主要功能   | 说明                                           |
| ---------------- | ------ | -------------------------------------------- |
| `main.py`        | 项目统一入口 | 接收命令参数，并根据参数启动训练、测试或完整实验流程                   |
| `configs/`       | 配置管理   | 保存不同模型的参数配置，包括学习率、batch size、epoch、数据路径和结果路径 |
| `core/`          | 核心流程   | 实现数据读取、模型训练、模型测试和评价流程                        |
| `data/`          | 数据存放   | 存放已经处理好的 eICU 数据集文件                          |
| `models/`        | 模型实现   | 存放不同药物推荐模型的具体代码                              |
| `scripts/`       | 运行脚本   | 提供训练、测试、完整运行和结果汇总脚本                          |
| `utils/`         | 工具函数   | 提供数据处理、配置读取、日志记录、指标计算等辅助功能                   |
| `utils/modules/` | 模型基础模块 | 存放图神经网络等可复用模块                                |

## 四、使用框架的命令

进入项目目录：

```bash
cd DREAM-main
```

训练模型：

```bash
python main.py train --model safedrug --dataset eicu_atc3 --cuda 0
```

测试模型：

```bash
python main.py test --model safedrug --dataset eicu_atc3 --cuda 0
```

训练后自动测试：

```bash
python main.py run --model safedrug --dataset eicu_atc3 --cuda 0
```

更换模型：

```bash
python main.py run --model gamenet --dataset eicu_atc3 --cuda 0
python main.py run --model retain --dataset eicu_atc3 --cuda 0
python main.py run --model micron --dataset eicu_atc3 --cuda 0
python main.py run --model molerec --dataset eicu_atc3 --cuda 0
```

更换数据集：

```bash
python main.py run --model safedrug --dataset eicu_all --cuda 0
python main.py run --model safedrug --dataset eicu_atc4 --cuda 0
```

汇总实验结果：

```bash
python scripts/export_results_summary.py
```

## 五、项目总结

DREAM 项目的核心思路是：用统一的实验框架管理多个药物推荐模型。`configs` 负责配置，`data` 负责数据，`models` 负责模型实现，`core` 负责训练和测试流程，`utils` 负责辅助函数，`scripts` 负责脚本化运行。

通过这种结构，可以在同一套框架下快速切换模型和数据集，完成训练、测试和结果对比。
