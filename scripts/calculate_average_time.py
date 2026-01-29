import re

def calculate_average_time_for_item(file_path, item_name):
    """
    从给定的文件中解析样本数据，并计算指定项的平均耗时。

    Args:
        file_path (str): 数据文件的路径。
        item_name (str): 要计算平均耗时的项的名称 (e.g., "opRunner.RunOp")。

    Returns:
        tuple: 包含样本数量、总耗时和平均耗时的元组。
               如果没有找到样本，则返回 (0, 0, 0)。
    """
    total_time = 0.0
    sample_count = 0
    
    # 用于从 "Run 1: " 行中提取时间的正则表达式
    time_pattern = re.compile(r"Run 1:\s+([\d.]+)\s+ms")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # 检查当前行是否包含 item_name
                if item_name in line:
                    # 如果是，则检查下一行是否存在并尝试匹配时间
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        match = time_pattern.search(next_line)
                        if match:
                            time_str = match.group(1)
                            try:
                                total_time += float(time_str)
                                sample_count += 1
                            except ValueError:
                                print(f"警告：无法将 '{time_str}' 转换为浮点数。跳过此数据点。")

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
        return 0, 0.0, 0.0
    except Exception as e:
        print(f"处理文件时发生错误：{e}")
        return 0, 0.0, 0.0

    # 计算平均值
    average_time = total_time / sample_count if sample_count > 0 else 0.0

    return sample_count, total_time, average_time

if __name__ == "__main__":
    # 要分析的文件列表
    files_to_analyze = ['Bai_reorder.txt', 'Gset_reorder.txt', 'HB_reorder.txt', 'JGD_Homology_reorder.txt', 'Pajek_reorder.txt', 'VDOL_reorder.txt','Bai_default.txt', 'Gset_default.txt', 'HB_default.txt', 'JGD_Homology_default.txt', 'Pajek_default.txt', 'VDOL_default.txt']
    
    # 要计算的项目列表
    items_to_calculate = ["opRunner.RunOp", "aclnnBcsrSpmmCustom"]

    for file_to_analyze in files_to_analyze:
        print(f"'{file_to_analyze}'\n")
        
        for item in items_to_calculate:
            count, total, average = calculate_average_time_for_item(file_to_analyze, item)

            if count > 0:
                print(f"项目 '{item}' 的分析结果：")
                print(f"  共找到 {count} 个样本。")
                print(f"  总耗时: {total:.4f} ms")
                print(f"  平均耗时: {average:.4f} ms\n")
            else:
                print(f"未在文件中找到 '{item}' 的有效样本数据。\n")

