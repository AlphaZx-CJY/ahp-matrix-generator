from typing import List, Tuple
import random
import typer
import numpy as np

app = typer.Typer()

# 随机一致性指标 RI 表 (Saaty, 1980)
RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
    11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59
}

# AHP 1-9 标度法值及其倒数
SCALE_VALUES = [1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1,
                2, 3, 4, 5, 6, 7, 8, 9]


def generate_ahp_matrix(n: int) -> List[List[float]]:
    """生成随机的 AHP 判断矩阵"""
    matrix = [[1.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i+1, n):
            # 随机选择标度值
            value = random.choice(SCALE_VALUES)
            matrix[i][j] = value
            matrix[j][i] = 1.0 / value

    return matrix


def calculate_weights_and_lambda_max(
    matrix: List[List[float]],
    weights: List[float]
) -> Tuple[List[float], float]:
    """
    计算权重和最大特征值 λ_max

    1. 让判断矩阵A和权重矩阵W相乘得到向量AW
    2. 计算 AW[i] / (n * W[i]) 对每个i求和
    3. 最大特征值 λ_max = Σ[AW[i] / (n * W[i])]
    """
    n = len(matrix)
    matrix_np = np.array(matrix, dtype=float)
    weights_np = np.array(weights, dtype=float)

    # 1. 计算 AW = A × W
    AW = matrix_np @ weights_np

    # 2. 计算 λ_max = Σ[AW[i] / (n * W[i])]
    lambda_list = list()
    for i in range(n):
        # 避免除以0的情况
        if abs(weights_np[i]) > 1e-6:
            lambda_list.append(AW[i] / (n * weights_np[i]))

    lambda_max = sum(lambda_list, 0)
    return lambda_list, lambda_max


def calculate_weights(matrix: List[List[float]]) -> Tuple[List[float], List[float], float, float]:
    """计算权重、最大特征值和一致性指标"""
    n = len(matrix)
    matrix_np = np.array(matrix)

    # 1. 列归一化
    column_sums = matrix_np.sum(axis=0)
    normalized_matrix = matrix_np / column_sums

    # 2. 行求和得到V矩阵
    v_matrix = normalized_matrix.sum(axis=1)

    # 3. 归一化V矩阵得到权重
    total_sum = v_matrix.sum()
    weights = v_matrix / total_sum

    # 4. 计算λ_max
    lambda_list, lambda_max = calculate_weights_and_lambda_max(matrix, weights)

    # 计算一致性指标 CI
    ci = (lambda_max - n) / (n - 1)

    # 计算一致性比率 CR
    ri = RI_TABLE.get(n, 1.49)  # 对于 n>15 使用近似值
    cr = ci / ri if ri > 0 else 0.0

    return weights.tolist(), lambda_list, lambda_max, cr


def format_matrix(matrix: List[List[float]]) -> str:
    """格式化矩阵为字符串"""
    return '\n'.join(['\t'.join(f"{x:.4f}" for x in row) for row in matrix])

def format_vector(vector: List[float]) -> str:
    """格式化矩阵为字符串"""
    return '\n'.join(['\t'.join(f"{x:.4f}" for x in vector)])


@app.command()
def generate(
    n: int = typer.Argument(..., help="判断矩阵的维度", min=1, max=15),
    max_trials: int = typer.Option(100, help="最大尝试次数以满足一致性要求"),
    cr_threshold: float = typer.Option(0.1, help="可接受的一致性比率阈值")
):
    """
    生成 AHP 判断矩阵并计算权重

    示例: python ahp_generator.py 4
    """
    if n > 15:
        typer.echo("警告: 维度大于15时RI值将使用近似值，结果可能不够准确")

    best_matrix = None
    best_weights = None
    best_cr = float('inf')
    best_lambda_list = None

    # 尝试生成满足一致性要求的矩阵
    for _ in range(max_trials):
        matrix = generate_ahp_matrix(n)
        weights, lambda_list, labmda_max, cr = calculate_weights(matrix)

        if cr < best_cr:
            best_cr = cr
            best_matrix = matrix
            best_weights = weights
            best_lambda_list = lambda_list
            best_lambda_max = labmda_max
        if cr <= cr_threshold:
            break

    # 输出结果
    typer.echo("\n判断矩阵:")
    typer.echo(format_matrix(best_matrix))

    typer.echo("\n权重向量:")
    for i, weight in enumerate(best_weights):
        typer.echo(f"因素 {i+1}: {weight:.4f}")

    typer.echo("\n最大特征值对应的特征向量:")
    typer.echo(format_vector(best_lambda_list))
    typer.echo(f"\n最大特征值 (λ_max): {best_lambda_max:.4f}")

    typer.echo(f"\n一致性比率 (CR): {best_cr:.4f}")

    if best_cr > cr_threshold:
        typer.echo(typer.style(
            f"警告: CR > {cr_threshold}，矩阵的一致性不可接受!",
            fg=typer.colors.RED, bold=True
        ))
    else:
        typer.echo(typer.style(
            f"CR <= {cr_threshold}，矩阵的一致性可接受",
            fg=typer.colors.GREEN
        ))



# 新增命令：检查输入矩阵的一致性
@app.command()
def check_matrix(
    matrix: str = typer.Option(..., help="输入判断矩阵，格式如 '1,2,3;0.5,1,4;0.333,0.25,1' 或文件路径"),
    cr_threshold: float = typer.Option(0.1, help="可接受的一致性比率阈值")
):
    """
    检查输入的判断矩阵是否符合AHP一致性要求
    示例: python ahp_generator.py check-matrix --matrix "1,2,3;0.5,1,4;0.333,0.25,1"
    """
    import os
    # 判断输入是文件还是字符串
    if os.path.isfile(matrix):
        with open(matrix, 'r', encoding='utf-8') as f:
            content = f.read().strip()
    else:
        content = matrix.strip()

    # 解析矩阵
    try:
        rows = content.split(';')
        parsed_matrix = [list(map(float, row.split(','))) for row in rows]
    except Exception as e:
        typer.echo(typer.style(f"矩阵解析失败: {e}", fg=typer.colors.RED, bold=True))
        raise typer.Exit(code=1)

    n = len(parsed_matrix)
    for row in parsed_matrix:
        if len(row) != n:
            typer.echo(typer.style("矩阵必须为 n x n 方阵！", fg=typer.colors.RED, bold=True))
            raise typer.Exit(code=1)

    # 计算一致性
    weights, lambda_list, lambda_max, cr = calculate_weights(parsed_matrix)

    typer.echo("\n输入的判断矩阵:")
    typer.echo(format_matrix(parsed_matrix))

    typer.echo("\n权重向量:")
    for i, weight in enumerate(weights):
        typer.echo(f"因素 {i+1}: {weight:.4f}")

    typer.echo("\n最大特征值对应的特征向量:")
    typer.echo(format_vector(lambda_list))
    typer.echo(f"\n最大特征值 (λ_max): {lambda_max:.4f}")

    typer.echo(f"\n一致性比率 (CR): {cr:.4f}")

    if cr > cr_threshold:
        typer.echo(typer.style(
            f"警告: CR > {cr_threshold}，矩阵的一致性不可接受!",
            fg=typer.colors.RED, bold=True
        ))
    else:
        typer.echo(typer.style(
            f"CR <= {cr_threshold}，矩阵的一致性可接受",
            fg=typer.colors.GREEN
        ))



# 新增命令：根据用户输入因素顺序生成判断矩阵
@app.command()
def generate_by_factors(
    factors: str = typer.Option(..., help="输入因素顺序，逗号分隔，如 '价格,质量,服务'"),
    max_trials: int = typer.Option(100, help="最大尝试次数以满足一致性要求"),
    cr_threshold: float = typer.Option(0.1, help="可接受的一致性比率阈值")
):
    """
    根据用户输入的因素顺序生成AHP判断矩阵
    示例: python ahp_generator.py generate-by-factors --factors "价格,质量,服务"
    """
    factor_list = [f.strip() for f in factors.split(',') if f.strip()]
    n = len(factor_list)
    if n < 2:
        typer.echo(typer.style("因素数量必须大于1！", fg=typer.colors.RED, bold=True))
        raise typer.Exit(code=1)
    if n > 15:
        typer.echo("警告: 维度大于15时RI值将使用近似值，结果可能不够准确")

    best_matrix = None
    best_weights = None
    best_cr = float('inf')
    best_lambda_list = None

    # 尝试生成满足一致性要求的矩阵
    for _ in range(max_trials):
        matrix = generate_ahp_matrix(n)
        weights, lambda_list, labmda_max, cr = calculate_weights(matrix)
        if cr < best_cr:
            best_cr = cr
            best_matrix = matrix
            best_weights = weights
            best_lambda_list = lambda_list
            best_lambda_max = labmda_max
        if cr <= cr_threshold:
            break

    # 输出结果
    typer.echo("\n因素顺序:")
    for i, factor in enumerate(factor_list):
        typer.echo(f"{i+1}. {factor}")

    typer.echo("\n判断矩阵:")
    typer.echo(format_matrix(best_matrix))

    typer.echo("\n权重向量:")
    for i, (factor, weight) in enumerate(zip(factor_list, best_weights)):
        typer.echo(f"{factor}: {weight:.4f}")

    typer.echo("\n最大特征值对应的特征向量:")
    typer.echo(format_vector(best_lambda_list))
    typer.echo(f"\n最大特征值 (λ_max): {best_lambda_max:.4f}")

    typer.echo(f"\n一致性比率 (CR): {best_cr:.4f}")

    if best_cr > cr_threshold:
        typer.echo(typer.style(
            f"警告: CR > {cr_threshold}，矩阵的一致性不可接受!",
            fg=typer.colors.RED, bold=True
        ))
    else:
        typer.echo(typer.style(
            f"CR <= {cr_threshold}，矩阵的一致性可接受",
            fg=typer.colors.GREEN
        ))


if __name__ == "__main__":
    app()
