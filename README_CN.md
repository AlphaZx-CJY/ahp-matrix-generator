# AHP 判断矩阵生成器

一个基于 Typer 的 Python 命令行工具，用于生成 AHP（层次分析法）的 1-9 标度法判断矩阵，并计算权重向量和一致性比率。

## 功能特性

- 🧮 生成符合 AHP 1-9 标度法的随机判断矩阵
- ⚖️ 计算各因素的权重向量
- ✅ 进行一致性检验（CR 值计算）
- 📊 格式化输出判断矩阵和权重结果
- 🚦 一致性比率可视化（绿色/红色提示）

## 安装

### 前提条件

- Python 3.11+
- pip 包管理器

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/AlphaZx-CJY/ahp-matrix-generator.git
cd ahp-matrix-generator 
```

2. 安装依赖：
```bash
pip install -r requirements.txt 
```

或者直接安装依赖：
```bash
pip install typer numpy 
```

## 使用说明

### 基本用法

生成一个 n×n 的判断矩阵：
```bash
python ahp_generator.py [维度] 
# 或
./aph-generator[.exe] [纬度]
```

示例（生成 4×4 矩阵）：
```bash
python ahp_generator.py 4 
```

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --max-trials | 最大尝试次数以满足一致性要求 | 100 |
| --cr-threshold | 可接受的一致性比率阈值 | 0.1 |

示例（生成 5×5 矩阵，最大尝试 200 次，CR 阈值 0.15）：
```bash
python ahp_generator.py 5 --max-trials 200 --cr-threshold 0.15 
```

### 输出示例

```text
判断矩阵:
1.0000  7.0000  9.0000  4.0000
0.1429  1.0000  1.0000  0.1667
0.1111  1.0000  1.0000  0.5000
0.2500  6.0000  2.0000  1.0000

权重向量:
因素 1: 0.6324
因素 2: 0.0670
因素 3: 0.0764
因素 4: 0.2241

最大特征值对应的特征向量:
1.0617  1.0117  1.0656  1.0452

最大特征值 (λ_max): 4.1842

一致性比率 (CR): 0.0682
```

## 算法说明

### 判断矩阵生成

- 使用 1-9 标度法及其倒数（共 17 个值）
- 对角线元素为 1
- 满足互反性：a<sub>ji</sub> = 1/a<sub>ij</sub>

### 权重计算

1. 列归一化：将判断矩阵按列归一化
2. 行求和：将归一化后的矩阵按行求和得到 V 矩阵
3. 归一化 V 矩阵：将 V 矩阵归一化得到权重向量

### 一致性检验

1. 计算最大特征值 λ_max：
- 计算 AW = A × W（判断矩阵 × 权重向量）
- λ_max = Σ[AW[i] / (n × W[i])]

2. 计算一致性指标 CI：
   CI = (λ_max - n) / (n - 1)   

3. 计算一致性比率 CR：
   CR = CI / RI   
（RI 为随机一致性指标，使用 Saaty 的标准 RI 表）

4. 一致性判断：
- CR ≤ 0.1：一致性可接受（绿色显示）
- CR > 0.1：一致性不可接受（红色警告）

## 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 项目仓库
2. 创建您的特性分支 (git checkout -b feature/AmazingFeature)
3. 提交您的更改 (git commit -m 'Add some AmazingFeature')
4. 推送到分支 (git push origin feature/AmazingFeature)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](./LICENSE) 文件

---

提示：对于重要决策，建议人工验证生成的判断矩阵的一致性。本工具适用于层次分析法(AHP)的初步研究。