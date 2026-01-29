def generate_identity_mtx(n, file_path):
    """
    生成n阶单位阵并保存为.mtx格式文件
    :param n: 单位阵的维度（n阶，n>0的整数）
    :param file_path: 保存的mtx文件路径，如'identity_5.mtx'
    """
    # 校验输入维度是否合法
    if not isinstance(n, int) or n <= 0:
        raise ValueError("单位阵维度必须是正整数！")
    
    # 单位阵的非零元素个数 = 维度n（对角线全1，其余为0）
    non_zero = n
    
    # 打开文件并写入内容（utf-8编码，覆盖写入）
    with open(file_path, 'w', encoding='utf-8') as f:
        # 1. 写入MTX文件头（稀疏矩阵、实数、通用矩阵、坐标存储）
        f.write("%%MatrixMarket matrix real general coordinate\n")
        # 2. 写入总行数、总列数、非零元素个数
        f.write(f"{n} {n} {non_zero}\n")
        # 3. 写入每个非零元素的行号、列号、值（单位阵对角线都是1，行号=列号）
        for i in range(1, n+1):
            f.write(f"{i} {i} 1.0\n")

# 测试：生成5阶单位阵，保存为identity_5.mtx
if __name__ == "__main__":
    dim=0
    dim_str=input("input the Identity matrix: ")
    if dim_str.isdigit():
       dim=int(dim_str)
    generate_identity_mtx(dim, f"identity_{dim}.mtx")
    print(f"{dim}阶单位阵已成功保存为identity_{dim}.mtx")