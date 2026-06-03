# DrugRec Benchmark 运行命令示例

建议在项目父目录运行以下命令，否则 `drugrec_benchmark` 包可能导入不到：

```bash
cd /home/qluai/zyg/mr
```

## 可用模型

配置名通常使用小写：

```text
safedrug
leap
gamenet
retain
cognet
molerec
micron
4sdrug
armr
```

对应模型：

```text
SafeDrug
Leap
GAMENet
Retain
COGNet
MoleRec
MICRON
4SDrug
ARMR
```

## 可用数据集

```text
eicu_all
eicu_atc3
eicu_atc4
mimic-iii_all
mimic-iii_atc3
mimic-iii_atc4
mimic-iv_all
mimic-iv_atc3
mimic-iv_atc4
```

## 单模型训练

```bash
python -m drugrec_benchmark.scripts.train \
  --model safedrug \
  --dataset eicu_atc3 \
  --cuda 0
```

更多示例：

```bash
python -m drugrec_benchmark.scripts.train --model gamenet --dataset mimic-iii_atc3 --cuda 0
python -m drugrec_benchmark.scripts.train --model leap --dataset mimic-iv_atc4 --cuda 0
python -m drugrec_benchmark.scripts.train --model cognet --dataset eicu_atc3 --cuda 0
```

训练会按配置里的 seeds 跑，默认是：

```yaml
[1203, 2024, 42, 1234, 5678]
```

结果目录大致为：

```text
drugrec_benchmark/results/<model>/<dataset>/seed_<seed>/run_<time>/
```

## 单模型测试，自动查找 checkpoint

```bash
python -m drugrec_benchmark.scripts.test \
  --model safedrug \
  --dataset eicu_atc3 \
  --cuda 0
```

## 单模型测试，指定 checkpoint

```bash
python -m drugrec_benchmark.scripts.test \
  --model safedrug \
  --dataset eicu_atc3 \
  --checkpoint drugrec_benchmark/results/safedrug/eicu_atc3/seed_1203/run_xxx/models/epoch_xxx_jaccard_xxxx.pth \
  --cuda 0
```

## 训练后立即测试，并生成跨 seed 汇总

```bash
python -m drugrec_benchmark.scripts.run \
  --model safedrug \
  --dataset eicu_atc3 \
  --cuda 0
```

会输出类似：

```text
drugrec_benchmark/results/safedrug/eicu_atc3/reports/cross_seed_summary.json
```

## 使用统一入口运行单任务

统一入口是 `drugrec_benchmark/main.py`，支持 `train`、`test`、`run` 三个子命令。

训练：

```bash
python -m drugrec_benchmark.main train \
  --model safedrug \
  --dataset eicu_atc3 \
  --cuda 0
```

测试：

```bash
python -m drugrec_benchmark.main test \
  --model safedrug \
  --dataset eicu_atc3 \
  --cuda 0
```

训练加测试：

```bash
python -m drugrec_benchmark.main run \
  --model safedrug \
  --dataset eicu_atc3 \
  --cuda 0
```

## 批量运行多个模型和数据集

例如 3 个模型乘 2 个数据集，共 6 个任务：

```bash
python -m drugrec_benchmark.main run \
  --models safedrug,gamenet,retain \
  --datasets eicu_atc3,mimic-iii_atc3 \
  --cuda-list 0,1 \
  --parallel 2
```

只批量训练：

```bash
python -m drugrec_benchmark.main train \
  --models safedrug,leap,gamenet,retain,cognet,micron,4sdrug,armr \
  --datasets eicu_atc3,mimic-iii_atc3,mimic-iv_atc3 \
  --cuda-list 0,1 \
  --parallel 2
```

只批量测试：

```bash
python -m drugrec_benchmark.main test \
  --models safedrug,gamenet,retain \
  --datasets eicu_atc3,mimic-iii_atc3 \
  --cuda-list 0,1 \
  --parallel 2
```

## GPU 调度参数示例

统一入口会用 `nvidia-smi` 检查 GPU 占用，再启动任务：

```bash
python -m drugrec_benchmark.main run \
  --models safedrug,gamenet,retain,cognet \
  --datasets eicu_atc3,mimic-iii_atc3 \
  --cuda-list 0,1,2,3 \
  --parallel 4 \
  --gpu-mem-threshold 0.5 \
  --gpu-util-threshold 50 \
  --max-procs-per-gpu 1 \
  --poll-interval 10
```

关闭实时 dashboard：

```bash
python -m drugrec_benchmark.main run \
  --models safedrug,gamenet \
  --datasets eicu_atc3 \
  --cuda-list 0,1 \
  --parallel 2 \
  --no-dashboard
```

```
/home/qluai/miniconda3/envs/zyg_mr/bin/python -m drugrec_benchmark.main run \
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

```
/home/qluai/miniconda3/envs/zyg_mr/bin/python -m drugrec_benchmark.main run \
  --models molerec,cognet \
  --datasets eicu_all,eicu_atc3,eicu_atc4 \
  --cuda-list 0,1,2,3 \
  --parallel 0 \
  --max-procs-per-gpu 3 \
  --max-starting-tasks 4 \
  --task-warmup-seconds 30 \
  --worker-threads 1 \
  --gpu-launch-interval 10 \
  --gpu-mem-threshold 0.9 \
  --gpu-util-threshold 100 \
  --poll-interval 5
```

```
/home/qluai/miniconda3/envs/zyg_mr/bin/python -m drugrec_benchmark.main run \
  --models molerec,cognet \
  --datasets eicu_all,eicu_atc3,eicu_atc4 \
  --cuda-list 0,1,2,3 \
  --parallel 0 \
  --max-procs-per-gpu 3 \
  --batch-size 4 \
  --worker-threads 1 \
  --max-starting-tasks 8 \
  --task-warmup-seconds 20 \
  --gpu-launch-interval 5 \
  --gpu-mem-threshold 0.95 \
  --gpu-util-threshold 100 \
  --poll-interval 5
```

## 注意事项

`--config` 参数在 `scripts/train.py` 和 `scripts/test.py` 里虽然声明了，但当前代码没有实际传给 `load_config`，所以自定义 config 覆盖目前不会生效。模型超参主要来自 `drugrec_benchmark/configs/*.yaml` 和 `base_config.yaml`。
