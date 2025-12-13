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
    INPUTS_DIR="../temp_input"
    OUTPUT_DIR="../output"
    # INPUTS_DIR="/root/autodl-tmp/MatmulInvocationNeo_v1/inputs"
    # OUTPUT_DIR="../output"
    # 时间测试记录在 '../output' 目录下，详情见 './src/main.cpp'

    # 清理并创建输出目录
    rm -rf $OUTPUT_DIR
    mkdir -p $OUTPUT_DIR

    FAILURE_LOG="$OUTPUT_DIR/failed_samples.log"
    rm -f $FAILURE_LOG

    # 2. 查找所有测试用例并运行
    for mtx_file in $(find $INPUTS_DIR -name "*.mtx"); do
        sample_name=$(basename $mtx_file .mtx)
        category_dir=$(dirname $mtx_file)
        sample_dir="$category_dir/$sample_name"
        
        echo "==================== Running test for $sample_name ===================="

        # 3. 解析矩阵维度
        dims=$(python3 scripts/parse_matrix.py $mtx_file)
        if [ $? -ne 0 ]; then
            echo "[ERROR]: Failed to parse matrix dimensions for $mtx_file"
            continue
        fi
        read -r m k n nnz window_num block_num <<< "$dims"
        echo "[INFO]: Matrix dimensions (M, K, N, NNZ): $m, $k, $n, $nnz"
        echo "[INFO]: Block info (WindowNum, BlockNum): $window_num, $block_num"

        # 4. 定义输入输出文件路径
        input_row_ptr="$sample_dir/row_ptr.bin"
        input_col="$sample_dir/col_idx.bin"
        input_values="$sample_dir/values.bin"
        input_b="$sample_dir/x2_gm.bin"
        output_c="$OUTPUT_DIR/${sample_name}_output_c.bin"

        # 5. 运行可执行文件并计时
        export LD_LIBRARY_PATH=$_ASCEND_INSTALL_PATH/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
        # echo "[INFO]: Execute op for $sample_name!"
        category=$(basename $category_dir)
        ./output/execute_spmm_op $m $k $n $window_num $block_num $input_row_ptr $input_col $input_values $input_b $output_c $category $sample_name
        if [ $? -ne 0 ]; then
            echo "[ERROR]: Acl executable run failed for sample $sample_name!"
            continue
        fi

        # 6. 比较真值文件
        golden_bin="$sample_dir/golden.bin"
        if [ -f "$golden_bin" ]; then
            # python3 scripts/verify_result.py $output_c $golden_bin > /dev/null 2>&1
            python3 scripts/verify_result.py $output_c $golden_bin > "$OUTPUT_DIR/${sample_name}_wrong_indices"
            if [ $? -ne 0 ]; then
                echo "[ERROR]: Verify result failed for sample $sample_name!"
                echo "[$sample_name] (M, K, N, NNZ): $m, $k, $n, $nnz" >> $FAILURE_LOG
            else
                echo "[INFO]: Verify result success for sample $sample_name!"
            fi
        else
            echo "[WARN]: golden.bin not found for sample $sample_name. Skipping verification."
        fi

        # 7. 删除输出文件以节省空间
        rm $output_c $input_row_indices $input_col_indices $input_values
        # echo "[INFO]: Removed output file and temp file"

        echo "==================== Finished test for $sample_name ===================="
        echo ""
    done
}

main
