#include "kernel_operator.h"


template<typename aType, typename bType, typename cType>
class BcsrSpmmKernel {
// output C Tile size [16, 16]
uint32_t CUBE_BLOCK_M = 16;
uint32_t CUBE_BLOCK_K = 32 / sizeof(aType);
uint32_t CUBE_BLOCK_SIZE = CUBE_BLOCK_M * CUBE_BLOCK_K;

public:
    __aicore__ inline BcsrSpmmKernel() {}
    __aicore__ inline void Init(
        GM_ADDR a_shape,
        GM_ADDR row_ptr, GM_ADDR col, GM_ADDR val,
        GM_ADDR b, GM_ADDR c, GM_ADDR workspace,
        int32_t M, int32_t N, int32_t K,
        uint32_t formerNum, uint32_t formerLength,
        uint32_t tailNum, uint32_t tailLength,
        uint32_t mmadNum, uint32_t mmadN,   
        uint32_t lastMmadN, uint32_t lastMmadCubeBlockNum,
        uint32_t lastKLength
    ) {
        // set cube only
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

        this->M = M;
        this->K = K;
        this->N = N;
        this->mmadNum = mmadNum;
        this->mmadCubeBlockNum = mmadN / CUBE_BLOCK_M;  
        this->lastMmadN = lastMmadN;
        this->lastMmadCubeBlockNum = lastMmadCubeBlockNum;
        this->mmadN = mmadN;
        this->lastKLength = lastKLength;
        AscendC::printf("BcsrSpmmKernel Init: BlockIdx=%d, M=%d, K=%d, N=%d, mmadNum=%d, mmadN=%d\n", 
            AscendC::GetBlockIdx(), M, K, N, mmadNum, mmadN);
        if (AscendC::GetBlockIdx() < formerNum) {
            this->rowWindowNum = formerLength;
            rowPtrGm.SetGlobalBuffer((__gm__ int32_t *)row_ptr + formerLength * AscendC::GetBlockIdx(), formerLength + 1);
            cGm.SetGlobalBuffer((__gm__ cType *)c + formerLength * AscendC::GetBlockIdx() * CUBE_BLOCK_M * N, 
                formerLength * CUBE_BLOCK_M * N);
        } else if (AscendC::GetBlockIdx() < formerNum + tailNum) {
            this->rowWindowNum = tailLength;
            rowPtrGm.SetGlobalBuffer((__gm__ int32_t *)row_ptr + formerLength * formerNum +
                tailLength * (AscendC::GetBlockIdx() - formerNum), tailLength + 1
            );
            cGm.SetGlobalBuffer((__gm__ cType *)c + (formerLength * formerNum +
                tailLength * (AscendC::GetBlockIdx() - formerNum)) * CUBE_BLOCK_M * N,
                tailLength * CUBE_BLOCK_M * N
            );
        }
        colGm.SetGlobalBuffer((__gm__ int32_t *)col + rowPtrGm.GetValue(0), 
            rowPtrGm.GetValue(this->rowWindowNum) - rowPtrGm.GetValue(0)
        );
        valGm.SetGlobalBuffer((__gm__ aType *)val + CUBE_BLOCK_SIZE * rowPtrGm.GetValue(0),
            CUBE_BLOCK_SIZE * (rowPtrGm.GetValue(this->rowWindowNum) - rowPtrGm.GetValue(0))
        );
        bGm.SetGlobalBuffer((__gm__ bType *)b, (uint64_t)K * N);

        pipe.InitBuffer(inQueueA1, 1, CUBE_BLOCK_SIZE * sizeof(aType)); // 512B
        pipe.InitBuffer(inQueueA2, 1, CUBE_BLOCK_SIZE * sizeof(aType)); // 512B
        pipe.InitBuffer(inQueueB1, 1, CUBE_BLOCK_K * this->mmadN * sizeof(bType));
        pipe.InitBuffer(inQueueB2, 1, CUBE_BLOCK_K * this->mmadN * sizeof(bType));
        pipe.InitBuffer(outQueueCO1, 1, CUBE_BLOCK_M * this->mmadN  * sizeof(cType));
    }

    __aicore__ inline void Process()
    {
        for (int32_t row = 0; row < rowWindowNum; row++) {
            AscendC::printf("Blockidx=%d, Processing row window %d/%d\n", AscendC::GetBlockIdx(), row, rowWindowNum);
            // 行窗口中的每块
            for (int32_t i = 0; i < rowPtrGm.GetValue(row + 1) - rowPtrGm.GetValue(row); i++) {
                int32_t col = colGm.GetValue(i);
                AscendC::printf("  Processing block %d/%d, col block idx=%d\n", i, 
                    rowPtrGm.GetValue(row + 1) - rowPtrGm.GetValue(row), col);
                // B窗口行中的每个 mmad 块
                for (int32_t j = 0; j < mmadNum; j++) {
                    // 因为是流水线式的，所以需要每次搬运 A 即使源地址一样
                    CopyInA(row, i);
                    CopyInB(j, col);
                    SplitA();
                    SplitB(j);
                    Compute(j);
                    CopyOut(row, j);
                }
            }
        }
    }

private:
    // // 每次 A 只读一个块，所以 ND 即 ZZ
    // // 可以直接用 LoadData 搬运 512B, GM->A2
    // __aicore__ inline void CopyInA(int32_t row, int32_t i) {
    //     AscendC::LocalTensor<aType> a2Local = inQueueA2.AllocTensor<aType>();
    //     auto aGm = this->valGm[(rowPtrGm.GetValue(row) - rowPtrGm.GetValue(0) + i) * CUBE_BLOCK_SIZE];
    //     AscendC::LoadData2DParams params = {
    //         // startIndex, repeatTimes, srcStride, 0, dstGap, ifTranspose, 0
    //         static_cast<uint16_t>(rowPtrGm.GetValue(row) - rowPtrGm.GetValue(0) + i), 1, 0, 0, 0, false, 0
    //     };
    //     AscendC::LoadData(a2Local, aGm, params);
    //     inQueueA2.EnQue<aType>(a2Local);
    // }

    // 但是这里保留 Gm->A1->A2 的形式，方便后续扩展
    __aicore__ inline void CopyInA(int32_t row, int32_t i) {
        AscendC::LocalTensor<aType> a1Local = inQueueA1.AllocTensor<aType>();
        auto aGm = this->valGm[(rowPtrGm.GetValue(row) - rowPtrGm.GetValue(0) + i) * CUBE_BLOCK_SIZE];

        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = CUBE_BLOCK_M;
        params.dValue = CUBE_BLOCK_K;
        params.srcNdMatrixStride = 0;
        params.srcDValue = CUBE_BLOCK_K;
        params.dstNzC0Stride = CUBE_BLOCK_M;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 0;

        AscendC::DataCopy(a1Local, aGm, params);
        // if (row == 0 && i == 0) {
            uint32_t array[] = {static_cast<uint32_t>(16), static_cast<uint32_t>(16)};
            AscendC::ShapeInfo shapeInfo(2, array); 
        // //     AscendC::DumpTensor(aGm, 0, 16*16, shapeInfo);
            AscendC::DumpTensor(a1Local, 0, 16*16, shapeInfo);
        // }
        inQueueA1.EnQue<aType>(a1Local);
    }

    // DataCopy API for each line of B
    // 如果 leading N 太大用不了 ND2NZ 随路转化
    __aicore__ inline void CopyInB(int32_t j, int32_t col) {
        // col是A的列，对B来说是行
        // j 是B的block的列
        AscendC::LocalTensor<bType> b1Local = inQueueB1.AllocTensor<bType>();
        
        auto offset = col * N + j * this->mmadN;
        
        // AscendC::DataCopyParams params;
        // 有一小部分 padding 的数据是不需要的，Mmad 时会忽略
        // DataCopy的单位是32B，half类型就是16个元素
        // 逐行搬运所以count是1，len应当是32*2B/32B，这样还需要把B做32字节对齐
        // params.blockCount = 1;
        // params.blockLen = ((j == mmadNum - 1) ? lastMmadN : this->mmadN) * sizeof(bType) / 32;
        // params.srcStride = 0;
        // params.dstStride = 0;
        // AscendC::printf("%d %d %d %d\n", params.blockCount, params.blockLen, params.srcStride, params.dstStride);
        // for (int32_t i = 0; i < CUBE_BLOCK_K; i++) {
        //     // Copy one line at a time
        //     AscendC::DataCopy(b1Local[i * mmadN], this->bGm[offset + i * N], params);
        // }

        // 需要ND2NZ
        // AscendC::Nd2NzParams params;
        // params.ndNum = 1;
        // params.nValue = CUBE_BLOCK_K;
        // params.dValue = this->mmadN;
        // params.srcNdMatrixStride = 0;
        // params.srcDValue = N;
        // params.dstNzC0Stride = 1;
        // params.dstNzNStride = 1;
        // params.dstNzMatrixStride = 1;

        // AscendC::DataCopy(b1Local, this->bGm[offset], params);

        // 手动ND2NZ
        AscendC::DataCopyParams params;
        params.blockCount = 1;
        params.blockLen = 16 * sizeof(bType) / 32;
        params.srcStride = 0;
        params.dstStride = 0;
        // 16*32 拆成左右两个16*16
        for (int32_t i = 0; i < CUBE_BLOCK_K; i++) {
            AscendC::DataCopy(b1Local[i * 16], this->bGm[offset + i * N], params);
            AscendC::DataCopy(b1Local[(i + CUBE_BLOCK_K) * 16], this->bGm[offset + i * N + 16], params);
        }

        AscendC::printf("Debug B Block: row %d, block col %d\n", col, j);
        uint32_t array[] = {static_cast<uint32_t>(16), static_cast<uint32_t>(32)};
        AscendC::ShapeInfo shapeInfo(2, array); 
        // AscendC::DumpTensor(this->bGm[offset], 0, 16*32, shapeInfo);
        AscendC::DumpTensor(b1Local, 1, 16*32, shapeInfo);
        inQueueB1.EnQue<bType>(b1Local);
    }

    __aicore__ inline void SplitA() {
        AscendC::LocalTensor<aType> a1Local = inQueueA1.DeQue<aType>();
        AscendC::LocalTensor<aType> a2Local = inQueueA2.AllocTensor<aType>();

        AscendC::LoadData2DParams params;
        // params.repeatTimes = CUBE_BLOCK_SIZE * sizeof(aType) / 512;
        params.repeatTimes = 1;
        params.srcStride = 1;
        params.ifTranspose = false;
        AscendC::LoadData(a2Local, a1Local, params);
        // AscendC::printf("Debug SplitA:\n");

        // uint32_t array[] = {static_cast<uint32_t>(16), static_cast<uint32_t>(16)};
        // AscendC::ShapeInfo shapeInfo(2, array); 
        // AscendC::DumpTensor(a2Local, 0, 16*16, shapeInfo);

        inQueueA2.EnQue<aType>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }

    // NZ2ZN, LoadDataWithTranspose API
    // sizeof(bType) <= 2 时可以用
    __aicore__ inline void SplitB(int32_t progress) {
        AscendC::LocalTensor<bType> b1Local = inQueueB1.DeQue<bType>();
        AscendC::LocalTensor<bType> b2Local = inQueueB2.AllocTensor<bType>();

        AscendC::LoadData2dTransposeParams params;
        params.startIndex = 0;
        // params.repeatTimes = (progress == mmadNum - 1) ? lastMmadCubeBlockNum : mmadCubeBlockNum;
        params.repeatTimes = 2;
        params.srcStride = 1;
        // params.dstGap = sizeof(bType) == 2 ? 0 : 1;
        params.dstGap = 0;
        params.dstFracGap = 0;
        AscendC::LoadDataWithTranspose(b2Local, b1Local, params);
        // AscendC::printf("Debug SplitB: progress=%d\n", progress);
        // uint32_t array[] = {static_cast<uint32_t>(16), static_cast<uint32_t>(32)};
        // AscendC::ShapeInfo shapeInfo(2, array); 
        // // AscendC::DumpTensor(this->bGm[offset], 0, 16*32, shapeInfo);
        // AscendC::DumpTensor(b2Local, 1, 16*32, shapeInfo);

        inQueueB1.FreeTensor(b1Local);
        inQueueB2.EnQue<bType>(b2Local);
    }

    __aicore__ inline void Compute(int32_t progress) {
        AscendC::LocalTensor<aType> a2Local = inQueueA2.DeQue<aType>();
        AscendC::LocalTensor<bType> b2Local = inQueueB2.DeQue<bType>();
        AscendC::LocalTensor<cType> c1Local = outQueueCO1.AllocTensor<cType>();

        AscendC::MmadParams params;
        params.m = CUBE_BLOCK_M;
        // TODO: K 不一定跟 CUBE_BLOCK_K 对齐
        // col == K / CUBE_BLOCK_K * CUBE_BLOCK_K, 尾部需要特殊处理
        // 可能可以通过给 bGm 更大的空间，padding 0 来解决
        params.k = CUBE_BLOCK_K;
        params.n = (progress == mmadNum - 1) ? lastMmadN : this->mmadN;

        // if (progress == 0) {
        // uint32_t array[] = {static_cast<uint32_t>(16), static_cast<uint32_t>(32)};
        // AscendC::ShapeInfo shapeInfo(2, array); 
        // AscendC::DumpTensor(a2Local, 0, 16*32, shapeInfo);
        // AscendC::DumpTensor(b2Local, 1, 16*32, shapeInfo);
        // // AscendC::DumpTensor(c1Local, 2, 16*32, shapeInfo);
        // }

        AscendC::Mmad(c1Local, a2Local, b2Local, params);

        //debug output
        AscendC::printf("Debug Compute: progress=%d\n", progress);
        uint32_t array[] = {static_cast<uint32_t>(16), static_cast<uint32_t>(32)};
        AscendC::ShapeInfo shapeInfo(2, array); 
        // AscendC::DumpTensor(a2Local, 0, 16*32, shapeInfo);
        // AscendC::DumpTensor(b2Local, 1, 16*32, shapeInfo);
        AscendC::DumpTensor(c1Local, 2, 16*32, shapeInfo);
        
        outQueueCO1.EnQue<cType>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }

    // Fixpipe API
    __aicore__ inline void CopyOut(int32_t row, int32_t progress) {
        auto cGm = this->cGm[row * CUBE_BLOCK_M * N + progress * mmadCubeBlockNum * CUBE_BLOCK_M];
        AscendC::LocalTensor<cType> c1Local = outQueueCO1.DeQue<cType>();

        AscendC::FixpipeParamsV220 params;
        params.ndNum = 1;
        params.mSize = CUBE_BLOCK_M;
        params.nSize = (progress == mmadNum - 1) ? lastMmadN : this->mmadN;
        params.srcStride = CUBE_BLOCK_M;
        params.dstStride = N;
        params.srcNdStride = 0;
        params.dstNdStride = 0;

        AscendC::SetAtomicAdd<cType>();
        AscendC::Fixpipe(cGm, c1Local, params);
        AscendC::SetAtomicNone();
        // AscendC::printf("Debug C Block: row %d, block col %d\n", row, progress);
        uint32_t array[] = {static_cast<uint32_t>(16), static_cast<uint32_t>(32)};
        AscendC::ShapeInfo shapeInfo(2, array); 
        // AscendC::DumpTensor(this->cGm, 3, 32*32, shapeInfo);
        // AscendC::DumpTensor(c1Local, 1, 16*32, shapeInfo);
        outQueueCO1.FreeTensor(c1Local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;

    AscendC::GlobalTensor<int32_t> rowPtrGm;
    AscendC::GlobalTensor<int32_t> colGm;
    AscendC::GlobalTensor<aType> valGm;

    AscendC::GlobalTensor<bType> bGm;
    AscendC::GlobalTensor<cType> cGm;

    int32_t M;
    int32_t K;
    int32_t N;
    uint32_t rowWindowNum;
    uint32_t mmadNum;
    uint32_t mmadCubeBlockNum;
    uint32_t lastMmadN;
    uint32_t lastMmadCubeBlockNum;
    uint32_t mmadN;
    uint32_t lastKLength;
};

extern "C" __global__ __aicore__ void bcsr_spmm_custom(
    GM_ADDR a_shape, GM_ADDR row_ptr, GM_ADDR col, GM_ADDR val,
    GM_ADDR b, GM_ADDR c,
    GM_ADDR workspace, GM_ADDR tiling
) {
    GET_TILING_DATA(tiling_data, tiling);

    BcsrSpmmKernel<half, half, float> op;
    op.Init(a_shape, row_ptr, col, val, b, c, workspace,
        tiling_data.M, tiling_data.N, tiling_data.K,
        tiling_data.formerNum, tiling_data.formerLength,
        tiling_data.tailNum, tiling_data.tailLength,
        tiling_data.mmadNum, tiling_data.mmadN,
        tiling_data.lastMmadN, tiling_data.lastMmadCubeBlockNum, 
        tiling_data.lastKLength
    );
    op.Process();
}