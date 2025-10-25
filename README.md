# AHP Judgment Matrix Generator

[中文介绍](./README_CN.md)

A Python-based command-line tool for generating AHP (Analytic Hierarchy Process) judgment matrices using the 1-9 scale method, and calculating weight vectors and consistency ratios.

## Features

- 🧮 Generates random AHP judgment matrices using the 1-9 scale method
- ⚖️ Calculates weight vectors for criteria
- ✅ Performs consistency checks (CR value calculation)
- 📊 Formatted output of judgment matrices and weight results
- 🚦 Visual consistency ratio indicators (green/red prompts)

## Installation

### Prerequisites

- Python 3.6+
- pip package manager

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/AlphaZx-CJY/ahp-matrix-generator.git
cd ahp-matrix-generator 
```

2. Install dependencies:
```bash
pip install -r requirements.txt 
```

Or install dependencies directly:
```bash
pip install typer numpy 
```

## Usage

### Basic Usage

Generate an n×n judgment matrix:
bash python ahp_generator.py [DIMENSION] 

Example (generate a 4×4 matrix):
bash python ahp_generator.py 4 

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| --max-trials | Maximum trials to satisfy consistency requirement | 100 |
| --cr-threshold | Acceptable consistency ratio threshold | 0.1 |

Example (generate a 5×5 matrix, max 200 trials, CR threshold 0.15):
bash python ahp_generator.py 5 --max-trials 200 --cr-threshold 0.15 

### Output Example

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

## Algorithm Explanation

### Judgment Matrix Generation

- Uses the 1-9 scale method and their reciprocals (17 values total)
- Diagonal elements are 1
- Satisfies reciprocity: a<sub>ji</sub> = 1/a<sub>ij</sub>

### Weight Calculation

1. Column Normalization: Normalize the judgment matrix by columns
2. Row Summation: Sum the normalized matrix by rows to get the V matrix
3. V Matrix Normalization: Normalize the V matrix to obtain the weight vector

### Consistency Check

1. Calculate maximum eigenvalue λ_max:
- Compute AW = A × W (judgment matrix × weight vector)
- λ_max = Σ[AW[i] / (n × W[i])]

2. Calculate Consistency Index (CI):
   CI = (λ_max - n) / (n - 1)   

3. Calculate Consistency Ratio (CR):
   CR = CI / RI   
(RI is the random index using Saaty's standard RI table)

4. Consistency Judgment:
- CR ≤ 0.1: Consistency acceptable (green indicator)
- CR > 0.1: Consistency unacceptable (red warning)

## Contribution Guidelines

Contributions are welcome! Please follow these steps:

1. Fork the project repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

Note: For important decision-making processes, it's recommended to manually verify the consistency of generated judgment matrices. This tool is suitable for preliminary AHP research.