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
    INPUTS_DIR="../inputs"
    # INPUTS_DIR="../inputs_all"
    OUTPUT_DIR="../output_all"
    MODE="reorder_condense"
    # INPUTS_DIR="/root/autodl-tmp/MatmulInvocationNeo_v1/inputs"
    # OUTPUT_DIR="../output"
    # 时间测试记录在 '../output' 目录下，详情见 './src/main.cpp'

    # 清理并创建输出目录
    rm -rf $OUTPUT_DIR
    mkdir -p $OUTPUT_DIR

    FAILURE_LOG="$OUTPUT_DIR/failed_samples.log"
    rm -f $FAILURE_LOG
    
    cate=""
    cate_refile=""

  while read -r incatedir; do
      while read -r sample_dir; do
    # 输出当前遍历到的文件夹路径（可替换为你的业务逻辑）
    IFS='/' read -ra dirlist <<< "$sample_dir"
    cate=${dirlist[-2]}
    catedir="${OUTPUT_DIR}/${cate}"
    sample_name=${dirlist[-1]%_re_colcondense}
    echo "==================== Running test for $sample_name ===================="
    #out_sample_dir="${catedir}/${dirlist[-1]}"

    mkdir -p "$catedir"

    # 4. 定义输入输出文件路径
    input_row_ptr="$sample_dir/rw_ptr.bin"
    input_col="$sample_dir/TC_col_ref.bin"
    input_values="$sample_dir/values.bin"
    input_ref="$sample_dir/reorder_ref.bin"
    input_b="$sample_dir/x2_gm.bin"
    input_info="$sample_dir/simple_info.txt"
    output_c="$catedir/${sample_name}_output_c.bin"
    cate_refile="$OUTPUT_DIR/${cate}_${MODE}.txt"

    echo "catefile ${cate_refile}"

    info_content=$(head -n 1 "$input_info")
    LFS=' ' read -r -a info_list <<< "${info_content}"
    echo "info is ${info_content[@]}"
    
    m=${info_list[0]}
    k=${info_list[1]}
    n=${info_list[2]}
    nnz=${info_list[3]}
    window_num=${info_list[4]}
    block_num=${info_list[5]}
    mean_nnz=${info_list[6]}
    con_thres=${info_list[7]}
    if [ -z "$con_thres" ]; then
      con_thres="1"
    fi
    echo "[INFO]: Matrix dimensions (M, K, N, NNZ): $m, $k, $n, $nnz"
    echo "[INFO]: Block info (WindowNum, BlockNum, Mean_nnz,Con_thres): $window_num, $block_num, $mean_nnz $con_thres"

    # 5. 运行可执行文件并计时
    export LD_LIBRARY_PATH=$_ASCEND_INSTALL_PATH/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
    echo "[INFO]: Execute op for $sample_name!"
    ./output/execute_spmm_op $m $k $n $window_num $block_num $input_row_ptr $input_col $input_values $input_b $output_c $cate $sample_name $MODE $input_ref $con_thres
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Acl executable run failed for sample $sample_name!"
        continue
    fi

    # 6. 比较真值文件
    golden_bin="$sample_dir/golden.bin"
    if [ -f "$golden_bin" ]; then
        # python3 scripts/verify_result.py $output_c $golden_bin > /dev/null 2>&1
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

    #7. "$OUTPUT_DIR/${sample_name}_wrong_indices"除输出文件以节省空间
    rm -r $output_c "$OUTPUT_DIR/${sample_name}_wrong_indices"
    echo "[INFO]: Removed output file and temp file"

    echo "==================== Finished test for $sample_name ===================="
    echo ""

  done   < <(find "$incatedir" -maxdepth 1 -type d -name "*_re_colcondense")
    echo "catefile ${cate_refile}"
    # 8.执行时间分析脚本
    #copy result to script dir for time analysis
    cp "$cate_refile" ../scripts/
    (
        cd ../scripts/
        python3 ./calculate_average_time.py
    )
done  < <(find "$INPUTS_DIR"  -maxdepth 1 -type d -name "*") 

}

main
