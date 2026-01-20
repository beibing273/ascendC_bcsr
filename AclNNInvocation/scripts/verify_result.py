#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import sys
import numpy as np

# for float32
relative_tol = 1e-4
absolute_tol = 1e-6
error_tol = 1e-4


def verify_result(output, golden,m,n, output_path, matrix_name):
    output = np.fromfile(output, dtype=np.float32).reshape(-1)
    golden = np.fromfile(golden, dtype=np.float32).reshape(-1)
    different_element_results = np.isclose(output,
                                           golden,
                                           rtol=relative_tol,
                                           atol=absolute_tol,
                                           equal_nan=True)
    different_element_indexes = np.where(different_element_results == False)[0]
    # for index in range(len(different_element_indexes)):
    #     real_index = different_element_indexes[index]
    #     golden_data = golden[real_index]
    #     output_data = output[real_index]
    #     print(
    #         "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
    #         (real_index, golden_data, output_data,
    #          abs(output_data - golden_data) / golden_data))
    #     if index == 100:
    #         break
    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        if golden_data == 0:
            continue
        print(
                "[%06d] expected: %-.9f, actual: %-.9f, rdiff: %-.6f" % 
                (real_index, golden_data, output_data, 
                abs(output_data - golden_data) / golden_data))

    error_ratio = float(different_element_indexes.size) / golden.size
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, error_tol))
    # 如果验证失败,打印两个矩阵进行对比，按照二维格式输出到两个文件
    if error_ratio > error_tol:
        # 为输出矩阵组合路径和名字
        output_matrix_path = "{}/{}_output_matrix.txt".format(output_path,matrix_name)
        golden_matrix_path = "{}/{}_golden_matrix.txt".format(output_path,matrix_name)
        print("Output Matrix:")
        print_2d_matrix(output,output_matrix_path, int(m), int(n))
        print("Golden Matrix:")
        print_2d_matrix(golden,golden_matrix_path, int(m), int(n))
    return error_ratio <= error_tol

# 打印二维的矩阵
def print_2d_matrix(data,data_path, rows, cols):
    with open(data_path, 'w') as f:
        for i in range(rows):
            for j in range(cols):
                f.write("%6.1f\t" % data[i * cols + j])
            f.write("\n")
        f.write("\n")

if __name__ == '__main__':
    try:
        # 新增m，n参数，方便打印二维矩阵进行调试
        # 新增输出路径目录和矩阵名字信息参数
        res = verify_result(sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4], sys.argv[5], sys.argv[6])
        if not res:
            raise ValueError("[ERROR] result error")
        else:
            print("test pass")
    except Exception as e:
        print(e)
        sys.exit(1)
