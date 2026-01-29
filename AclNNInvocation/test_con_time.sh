#!/bin/bash
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
source $_ASCEND_INSTALL_PATH/bin/setenv.bash
export DDK_PATH=$_ASCEND_INSTALL_PATH
export NPU_HOST_LIB=$_ASCEND_INSTALL_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/devlib

function main {
    # 初始化性能统计关联数组（总和/样本数，无bc依赖，用awk计算）
    declare -A perf_sum perf_count

    # 1. 编译acl可执行文件
    cd $CURRENT_DIR
    rm -rf build
    mkdir -p build
    cd build
    cmake ../src -DCMAKE_SKIP_RPATH=TRUE
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Cmake failed!"
        return 1
    fi
    echo "[INFO]: Cmake success!"
    make
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Make failed!"
        return 1
    fi
    echo "[INFO]: Make success!"
    cd $CURRENT_DIR

    # 定义输入输出目录
    INPUTS_DIR="../temp_input_ggt"
    OUTPUT_DIR="../output_all"
    MODE="condense"

    # 清理并创建输出目录
    rm -rf $OUTPUT_DIR
    mkdir -p $OUTPUT_DIR

    FAILURE_LOG="$OUTPUT_DIR/failed_samples.log"
    rm -f $FAILURE_LOG

    # 2. 查找所有测试用例并运行
    for mtx_file in $(find $INPUTS_DIR -name "*.mtx"); do
        sample_name=$(basename $mtx_file .mtx)
        category_dir=$(dirname $mtx_file)
        sample_dir="$category_dir/${sample_name}_colcondense"
        
        echo "==================== Running test for $sample_name ===================="

        # 3. 解析矩阵维度
        dims=$(python3 scripts/parse_matrix.py $mtx_file)
        if [ $? -ne 0 ]; then
            echo "[ERROR]: Failed to parse matrix dimensions for $mtx_file"
            continue
        fi
        read -r m k n nnz window_num block_num mean_nnz con_thres<<< "$dims"
        echo "[INFO]: Matrix dimensions (M, K, N, NNZ): $m, $k, $n, $nnz"
        echo "[INFO]: Block info (WindowNum, BlockNum, Fill_rate,Con_thres): $window_num, $block_num, $mean_nnz $con_thres"

        # 4. 定义输入输出文件路径
        input_row_ptr="$sample_dir/rw_ptr.bin"
        input_col="$sample_dir/TC_col_ref.bin"
        input_values="$sample_dir/values.bin"
        input_b="$sample_dir/x2_gm.bin"
        output_c="$OUTPUT_DIR/${sample_name}_output_c.bin"

        # 5. 运行可执行文件并计时
        export LD_LIBRARY_PATH=$_ASCEND_INSTALL_PATH/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/Ascend910B2/lib:$LD_LIBRARY_PATH 
        category=$(basename $category_dir)
        msprof op simulator --output=./msprof_out_default  ./output/execute_spmm_op $m $k $n $window_num $block_num $input_row_ptr $input_col $input_values $input_b $output_c $category $sample_name $MODE $con_thres
        if [ $? -ne 0 ]; then
            echo "[ERROR]: Acl executable run failed for sample $sample_name!"
            continue
        fi

        # 提取性能数据 + 删除execute_spmm_op文件夹
        perf_value="0"
        op_basic_info_file=$(find ./msprof_out_default -name OpBasicInfo.csv 2>/dev/null | head -n 1)
        if [ -n "$op_basic_info_file" ] && [ -f "$op_basic_info_file" ]; then
            # 提取第二行、逗号分隔的第三个字段（清理非数字字符）
            perf_value=$(sed -n '2p' "$op_basic_info_file" | cut -d',' -f3 | sed 's/[^0-9.]//g')
            echo "[INFO]: Extracted performance value for $sample_name: $perf_value"
            # 强制删除msprof_out文件夹
            # if [ -d "./msprof_out_default" ]; then
            #     rm -rf ./msprof_out_default && echo "[INFO]: Deleted ./msprof_out_default folder"
            # fi
        else
            echo "[WARN]: OpBasicInfo.csv not found for sample $sample_name, use default value 0"
        fi

        # 统计分类性能数据（awk替代bc，浮点累加，无外部依赖）
        if [[ "$perf_value" =~ ^[0-9.]+$ ]]; then
            # 浮点累加：用awk计算当前总和+新值，保留8位小数
            current_sum=${perf_sum["$category_dir"]:-0.0}
            new_sum=$(echo "$current_sum $perf_value" | awk '{printf "%.8f", $1 + $2}')
            perf_sum["$category_dir"]=$new_sum
            # 样本数整数累加（bash原生支持）
            perf_count["$category_dir"]=$(( ${perf_count["$category_dir"]:-0} + 1 ))
        else
            echo "[WARN]: Invalid performance value '$perf_value' for $sample_name, skip statistics"
        fi

        # 6. 比较真值文件
        golden_bin="$sample_dir/golden.bin"
        if [ -f "$golden_bin" ]; then
            python3 scripts/verify_result.py $output_c $golden_bin $m $n $OUTPUT_DIR $sample_name > "$OUTPUT_DIR/${sample_name}_wrong_indices"
            if [ $? -ne 0 ]; then
                echo "[ERROR]: Verify result failed for sample $sample_name!"
                echo "[$sample_name] (M, K, N, NNZ): $m, $k, $n, $nnz" >> $FAILURE_LOG
            else
                echo "[INFO]: Verify result success for sample $sample_name!"
            fi
        else
            echo "[WARN]: golden.bin not found for sample $sample_name. Skipping verification."
        fi

        echo "==================== Finished test for $sample_name ===================="
        echo ""
    done

    # 汇总统计并写入result.txt（awk计算平均值，保留6位小数）
    echo -e "\n==================== Performance Statistics ===================="
    # 追加时间戳，便于追溯
    echo -e "\n[$(date +%Y-%m-%d\ %H:%M:%S)] Performance Statistics" >> ./result.txt
    # 遍历所有分类目录的统计数据
    for cat_dir in "${!perf_sum[@]}"; do
        sum=${perf_sum[$cat_dir]}
        count=${perf_count[$cat_dir]}
        # awk计算平均值，避免除0，保留6位小数
        if [ "$count" -eq 0 ]; then
            avg="0.000000"
        else
            avg=$(echo "$sum $count" | awk '{printf "%.6f", $1 / $2}')
        fi
        cat_name=$(basename "$cat_dir")
        # 控制台输出
        echo "[INFO]: Category: $cat_name | Total Samples: $count | Total Performance: $sum | Average: $avg"
        # 追加写入result.txt（格式清晰，可直接解析）
        echo "Category: $cat_name | Directory: $cat_dir | Total Samples: $count | Total Performance: $sum | Average: $avg" >> ./result.txt
    done
    echo "[INFO]: Performance statistics have been written to ./result.txt"
}

main