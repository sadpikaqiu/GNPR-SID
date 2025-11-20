# QWEN.md - 生成式下一个POI推荐与语义ID

## 项目概述

此仓库包含王东升等人在KDD 2025上发表的论文《Generative Next POI Recommendation with Semantic ID》的官方实现。该项目专注于开发一种使用从残差向量量化变分自编码器（RQ-VAE）派生的语义ID进行下一个兴趣点（POI）推荐的生成模型。

### 主要组件

1. **RQ-VAE模块**: 从POI特征生成语义ID的核心实现
2. **示例数据集**: 基于纽约签到数据用于演示和评估
3. **数据预处理**: 用于数据清理和格式转换的Jupyter笔记本
4. **模型微调**: 使用LLaMA-Factory框架

## 文件结构

```
GNPR-SID/
├── code/                 # 核心实现文件
│   ├── codebook.py       # 生成从离散标记ID到语义ID的映射
│   ├── POIdataset.py     # POI数据的数据集类
│   ├── train_rqvae.py    # RQ-VAE的训练脚本
│   ├── trainer.py        # 训练循环实现
│   ├── utils.py          # 工具函数
│   └── RQVAE/            # RQ-VAE模块实现
│       ├── layers.py
│       ├── rq.py         # 残差量化器
│       ├── rqvae.py      # 完整的RQ-VAE模型
│       ├── vq.py         # 向量量化器
│       └── vq.txt
├── datasets/             # 数据集目录
│   └── NYC/              # 示例纽约数据集
│       └── codebooks_0.25.csv  # 生成的语义ID
├── data2json.ipynb       # 数据转换为JSON
├── dataprocess.ipynb     # 原始数据预处理
├── test.ipynb            # 测试笔记本
├── Readme.md             # 项目文档
└── QWEN.md              # 本文档
```

## 核心概念

### RQ-VAE (残差向量量化变分自编码器)
- 在残差配置中使用多个向量量化器
- 将连续POI嵌入转换为离散语义标记
- 由编码器、残差量化器（RQ）和解码器组件组成

### 语义ID生成
1. 在POI特征上训练RQ-VAE（类别、区域、时间、用户邻居）
2. 提取每个POI的离散标记作为语义ID
3. 使用这些语义ID进行生成式推荐模型

## 构建和运行

### 前提条件
- Python 3.x
- PyTorch
- Pandas
- NumPy
- LLaMA-Factory（用于微调阶段）

### 训练RQ-VAE

1. **准备数据**:
   - 将POI数据格式化为CSV，列包括：Pid,Uid,Catname,Region,Time,neighbors,forward_neighbors
   - 放在`datasets/{data_mode}/`目录中

2. **训练RQ-VAE模型**:
   ```bash
   cd code
   python train_rqvae.py
   ```

3. **生成语义ID**:
   ```bash
   cd code
   python codebook.py
   ```

### 数据预处理
- 使用`dataprocess.ipynb`进行原始数据清理和格式化
- 使用`data2json.ipynb`将处理后的数据转换为模型就绪的JSON格式

### 模型微调
- 论文使用LLaMA-Factory进行生成模型的微调
- 按照LLaMA-Factory的说明使用处理后的数据集

## 关键文件和参数

### `train_rqvae.py`
- RQ-VAE模型的主训练脚本
- 关键参数：
  - `--num_emb_list`: 每个VQ层的嵌入数量（例如，[32,32,32]）
  - `--e_dim`: VQ码本嵌入大小（例如，64）
  - `--beta`: 承诺损失的Beta值（例如，0.25）
  - `--lamda`: 多样性损失的Lambda值（例如，0）

### `codebook.py`
- 从训练模型生成语义ID
- 将语义ID和向量输出到CSV

### `POIdataset.py`
- 处理分类特征的独热编码（类别、区域、时间、用户邻居）
- 支持NYC、TKY和CA数据集，具有不同的类别数量

### `RQVAE/rqvae.py`
- 主RQ-VAE模型实现
- 结合编码器、残差量化器和解码器

## 研究用途

1. 要重现结果：
   - 使用提供的纽约数据集或准备自己的数据集
   - 使用`train_rqvae.py`训练RQ-VAE
   - 使用`codebook.py`生成语义ID
   - 使用LLaMA-Factory进行语义ID的微调

2. 对于自定义数据集：
   - 修改`POIdataset.py`以处理您的数据格式
   - 相应更新`POIdataset.py`中的类别/区域/时间计数
   - 根据需要在训练脚本中调整模型参数

## 开发约定

- 代码库使用PyTorch进行深度学习实现
- 配置参数通过argparse管理
- 训练包括评估碰撞率以衡量码本质量
- 模型检查点使用评估指标保存
- 代码遵循标准的PyTorch训练模式并包含日志记录

## 引用

对于研究论文：
```
@inproceedings{wang2025generative,
  title={Generative Next POI Recommendation with Semantic ID},
  author={Wang, Dongsheng and Huang, Yuxi and Gao, Shen and Wang, Yifan and Huang, Chengrui and Shang, Shuo},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={2904--2914},
  year={2025}
}
```