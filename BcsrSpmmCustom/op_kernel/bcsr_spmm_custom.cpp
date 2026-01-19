#include "kernel_operator.h"


template<typename aType, typename bType, typename cType,typename idxType>
class BcsrSpmmKernel {
// output C Tile size [16, 16]
uint32_t CUBE_BLOCK_M = 8;
uint32_t CUBE_BLOCK_K = 32 / sizeof(aType);
uint32_t CUBE_BLOCK_N = 8;
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
        uint32_t lastKLength
    ) {
        // set cube only
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

        this->M = M;
        this->K = K;
        this->N = N;
        this->lastKLength = lastKLength;
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
        // colGm.SetGlobalBuffer((__gm__ int32_t *)col + rowPtrGm.GetValue(0), 
        //     rowPtrGm.GetValue(this->rowWindowNum) - rowPtrGm.GetValue(0)
        // );
        colGm.SetGlobalBuffer((__gm__ int32_t *)col + rowPtrGm.GetValue(0)*CUBE_BLOCK_K, 
            (rowPtrGm.GetValue(this->rowWindowNum) - rowPtrGm.GetValue(0))*CUBE_BLOCK_K
        );
        valGm.SetGlobalBuffer((__gm__ aType *)val + CUBE_BLOCK_SIZE * rowPtrGm.GetValue(0),
            CUBE_BLOCK_SIZE * (rowPtrGm.GetValue(this->rowWindowNum) - rowPtrGm.GetValue(0))
        );
        //每个核都需要获得完整的B矩阵
        bGm.SetGlobalBuffer((__gm__ bType *)b, (uint64_t)K * N);
            
        pipe.InitBuffer(inQueueA1, 1, CUBE_BLOCK_SIZE * sizeof(aType)); // 512B
        pipe.InitBuffer(inQueueA2, 1, CUBE_BLOCK_SIZE * sizeof(aType)); // 512B
        pipe.InitBuffer(inQueueB1, 1, CUBE_BLOCK_K * this->N * sizeof(bType));
        //pipe.InitBuffer(inQueueB2, 1, CUBE_BLOCK_K * this->N * sizeof(bType));
        pipe.InitBuffer(inQueueB2, 1, CUBE_BLOCK_K *CUBE_BLOCK_N* sizeof(bType));
        //pipe.InitBuffer(outQueueCO1, 1, CUBE_BLOCK_M * this->N  * sizeof(cType));
        pipe.InitBuffer(outQueueCO1, 1, CUBE_BLOCK_M * CUBE_BLOCK_N * sizeof(cType));
        pipe.InitBuffer(rowBQueue,1,CUBE_BLOCK_K*sizeof(idxType));
    }

    // __aicore__ inline void Process()
    // {
    //     //这里的row就是 对应的一行，不是一个windows
    //     for (int32_t row = 0; row < rowWindowNum; row++) {
    //         // AscendC::printf("Blockidx=%d, Processing row window %d/%d\n", AscendC::GetBlockIdx(), row, rowWindowNum);
    //         // 行窗口中的每块 
    //         for (int32_t i = 0; i < rowPtrGm.GetValue(row + 1) - rowPtrGm.GetValue(row); i++) {
    //             int32_t col = colGm.GetValue(
    //                 rowPtrGm.GetValue(row) - rowPtrGm.GetValue(0) + i
    //             );
    //             CopyInA(row, i);
    //             CopyInB(col);
    //             SplitA();
    //             SplitB();
    //             Compute();
    //             CopyOut(row);
    //         }
    //     }
    // }

    __aicore__ inline void Process()
    {
        //这里的row就是 对应的一行，不是一个windows
        for (int32_t row = 0; row < rowWindowNum; row++) {
             AscendC::printf("Blockidx=%d, Processing row window %d/%d\n", AscendC::GetBlockIdx(), row, rowWindowNum);
            // 行窗口中的每块
            
            for (int32_t i = 0; i < rowPtrGm.GetValue(row + 1) - rowPtrGm.GetValue(row); i++) {
                // int32_t col = colGm.GetValue(
                //     rowPtrGm.GetValue(row) - rowPtrGm.GetValue(0) + i
                // );
                //将当前TC块的列对应B的行写入队列
                AscendC::LocalTensor<idxType> sab=rowBQueue.AllocTensor<idxType>();
                AscendC::DataCopyParams sabparam;
                sabparam.blockCount=1;
                sabparam.blockLen=CUBE_BLOCK_K*sizeof(idxType)/32;
                sabparam.srcStride=0;
                sabparam.dstStride=0;
                AscendC::DataCopy(sab,colGm[(rowPtrGm(row)-rowPtrGm(0)+i)*CUBE_BLOCK_K],sabparam);
                AscendC::DumpTensor(sab,1,CUBE_BLOCK_K);
                AscendC::DumpTensor(colGm[(rowPtrGm(row)-rowPtrGm(0)+i)*CUBE_BLOCK_K],0,CUBE_BLOCK_K);
                rowBQueue.EnQue<idxType>(sab);
                CopyInA(row, i);
                AscendC::printf("Blockid:%d,CopyInA is over\n",AscendC::GetBlockIdx());
                CopyInB();
                AscendC::printf("Blockid:%d,CopyInB is over\n",AscendC::GetBlockIdx());
                SplitA();
                AscendC::printf("Blockid:%d,splitA is over\n",AscendC::GetBlockIdx());
                for(int j=0;j<N/CUBE_BLOCK_N;++j){
                  AscendC::printf("Blockid:%d,col loop is %d :",AscendC::GetBlockIdx(),j);
                  SplitB(j);
                  AscendC::printf("splitB is over, ");
                  Compute();
                  AscendC::printf("compute is over, ");
                  CopyOut(row,j);
                  AscendC::printf("copyout is over\n");
                }
                   
            }
        }
    }

private:

    // 但是这里保留 Gm->A1->A2 的形式，方便后续扩展
    __aicore__ inline void CopyInA(int32_t row, int32_t i) {
        AscendC::LocalTensor<aType> a1Local = inQueueA1.AllocTensor<aType>();
        //选择 对应的block
        auto aGm = this->valGm[(rowPtrGm.GetValue(row) - rowPtrGm.GetValue(0) + i) * CUBE_BLOCK_SIZE];
        AscendC::printf("\n");
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = CUBE_BLOCK_M;
        params.dValue = CUBE_BLOCK_K;
        params.srcNdMatrixStride = 0;
        params.srcDValue = CUBE_BLOCK_K;
        // params.dstNzC0Stride = 0; //这个值暂时没用
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 0;

        AscendC::DataCopy(a1Local, aGm, params);
        
        inQueueA1.EnQue<aType>(a1Local);
    }

    // // DataCopy API for each line of B
    // // 如果 leading N 太大用不了 ND2NZ 随路转化,N大小设置为128时可以用 ND2NZ
    // __aicore__ inline void CopyInB(int32_t col) {
    //     // col是A的列，对B来说是行
    //     // j 是B的block的列
    //     AscendC::LocalTensor<bType> b1Local = inQueueB1.AllocTensor<bType>();
    //     auto offset = col * N ;// // 需要ND2NZ
    //     AscendC::Nd2NzParams params;
    //     params.ndNum = 1;
    //     params.nValue = CUBE_BLOCK_K;
    //     params.dValue = N;
    //     params.srcNdMatrixStride = 0;
    //     params.srcDValue = N;
    //     params.dstNzC0Stride = CUBE_BLOCK_K;
    //     params.dstNzNStride = 1;
    //     params.dstNzMatrixStride = 0;

    //     AscendC::DataCopy(b1Local, this->bGm[offset], params);

    //     inQueueB1.EnQue<bType>(b1Local);
    // }

    //列浓缩后的CopyInB
    __aicore__ inline void CopyInB() {

           
        AscendC::LocalTensor<bType> b1local=inQueueB1.AllocTensor<bType>();
        AscendC::LocalTensor<idxType> idxBlocal=rowBQueue.DeQue<idxType>();
        //此处DumpTensor不为零
        AscendC::DumpTensor(idxBlocal,0,CUBE_BLOCK_K);
        AscendC::DataCopyParams b1param;
        b1param.blockCount=N/CUBE_BLOCK_N;
        b1param.blockLen=CUBE_BLOCK_N*sizeof(bType)/32;
        b1param.srcStride=0;
        //copy同时进行ND->NZ转换
        b1param.dstStride=CUBE_BLOCK_N*(CUBE_BLOCK_K-1)*sizeof(bType)/32;
        for(int i=0;i<CUBE_BLOCK_K;++i){
           //直接获取值就为零
           AscendC::printf("第%d次对应B的第%d行\n",i,idxBlocal.GetValue(i));
           DataCopy(b1local[i*CUBE_BLOCK_N],bGm[idxBlocal.GetValue(i)*N],b1param);
          
        }
        AscendC::DumpTensor(b1local,0,N*CUBE_BLOCK_K);
        inQueueB1.EnQue<bType>(b1local);

        rowBQueue.FreeTensor(idxBlocal);
    }

    __aicore__ inline void SplitA() {
        AscendC::LocalTensor<aType> a1Local = inQueueA1.DeQue<aType>();
        AscendC::LocalTensor<aType> a2Local = inQueueA2.AllocTensor<aType>();

        AscendC::LoadData2DParams params;
        // params.repeatTimes = CUBE_BLOCK_SIZE * sizeof(aType) / 512;
        params.repeatTimes = 1;
        params.srcStride = 0;
        params.dstGap = 0;
        params.ifTranspose = false;
        AscendC::LoadData(a2Local, a1Local, params);

        inQueueA2.EnQue<aType>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }

    // // NZ2ZN, LoadDataWithTranspose API
    // // sizeof(bType) <= 2 时可以用
    // __aicore__ inline void SplitB() {
    //     AscendC::LocalTensor<bType> b1Local = inQueueB1.DeQue<bType>();
    //     AscendC::LocalTensor<bType> b2Local = inQueueB2.AllocTensor<bType>();

    //     AscendC::LoadData2DParams loadDataparams;
    //     loadDataparams.repeatTimes = N/16;
    //     loadDataparams.srcStride = 1;
    //     loadDataparams.dstGap = 0;
    //     loadDataparams.ifTranspose = true;
    //     AscendC::LoadData(b2Local, b1Local, loadDataparams);

    //     inQueueB2.EnQue<bType>(b2Local);
    //     inQueueB1.FreeTensor(b1Local);
    // }

    //列浓缩之后的splitB
    __aicore__ inline void SplitB(const int32_t bcol) {
        AscendC::LocalTensor<bType> b1Local = inQueueB1.DeQue<bType>();
        AscendC::LocalTensor<bType> b2Local = inQueueB2.AllocTensor<bType>();

        AscendC::LoadData2DParams loadDataparams;
        loadDataparams.repeatTimes = 1;
        loadDataparams.srcStride = 1;
        loadDataparams.dstGap = 0;
        loadDataparams.ifTranspose = true;
        AscendC::LoadData(b2Local, b1Local[bcol*CUBE_BLOCK_K*CUBE_BLOCK_N], loadDataparams);
        // if(AscendC::GetBlockIdx()==0 && bcol==12){
        // AscendC::printf("12:\n");
        //    for(int i=0;i<CUBE_BLOCK_K;++i){
        //     for(int j=0;j<CUBE_BLOCK_N;++j){
        //         AscendC::printf("\n%f ",static_cast<__gm__ float>(b2Local(i*CUBE_BLOCK_N+j)));
        //     }
        //     AscendC::printf("\n");
        //    }
        // }
        inQueueB2.EnQue<bType>(b2Local);
        if(bcol==(N-CUBE_BLOCK_N)/CUBE_BLOCK_N)
           inQueueB1.FreeTensor(b1Local);
    }

    // __aicore__ inline void Compute() {
    //     AscendC::LocalTensor<aType> a2Local = inQueueA2.DeQue<aType>();
    //     AscendC::LocalTensor<bType> b2Local = inQueueB2.DeQue<bType>();
    //     AscendC::LocalTensor<cType> c1Local = outQueueCO1.AllocTensor<cType>();

    //     AscendC::MmadParams params;
    //     params.m = CUBE_BLOCK_M;
    //     params.k = CUBE_BLOCK_K;
    //     params.n =N;

    //     AscendC::Mmad(c1Local, a2Local, b2Local, params);

    //     outQueueCO1.EnQue<cType>(c1Local);
    //     inQueueA2.FreeTensor(a2Local);
    //     inQueueB2.FreeTensor(b2Local);
    // }
      //列浓缩后的计算
        __aicore__ inline void Compute() {
        AscendC::LocalTensor<aType> a2Local = inQueueA2.DeQue<aType>();
        AscendC::LocalTensor<bType> b2Local = inQueueB2.DeQue<bType>();
        AscendC::LocalTensor<cType> c1Local = outQueueCO1.AllocTensor<cType>();
        AscendC::printf(" 1 ");
        AscendC::MmadParams params;
        params.m = CUBE_BLOCK_M;
        params.k = CUBE_BLOCK_K;
        params.n = CUBE_BLOCK_N;
        AscendC::printf(" 2 ");
        AscendC::Mmad(c1Local, a2Local, b2Local, params);
        AscendC::printf(" 3 ");
        outQueueCO1.EnQue<cType>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
        AscendC::printf(" 4 ");
    }
    // // Fixpipe API
    // __aicore__ inline void CopyOut(int32_t row) {
    //     auto cGm = this->cGm[row * CUBE_BLOCK_M * N ];
    //     AscendC::LocalTensor<cType> c1Local = outQueueCO1.DeQue<cType>();

    //     AscendC::FixpipeParamsV220 params;
    //     params.ndNum = 1;
    //     params.mSize = CUBE_BLOCK_M;
    //     params.nSize = N;
    //     params.srcStride = CUBE_BLOCK_M;
    //     params.dstStride = N;
    //     params.srcNdStride = 0;
    //     params.dstNdStride = 0;

    //     AscendC::SetAtomicAdd<cType>();
    //     AscendC::Fixpipe(cGm, c1Local, params);
    //     AscendC::SetAtomicNone();
    //     // AscendC::printf("Debug C Block: row %d, block col %d\n", row, progress);
    //     // uint32_t array[] = {static_cast<uint32_t>(16), static_cast<uint32_t>(32)};
    //     // AscendC::ShapeInfo shapeInfo(2, array); 
    //     // AscendC::DumpTensor(this->cGm, 3, 32*32, shapeInfo);
    //     // AscendC::DumpTensor(c1Local, 1, 16*32, shapeInfo);
    //     outQueueCO1.FreeTensor(c1Local);
    // }

    // Fixpipe API
    __aicore__ inline void CopyOut(int32_t row,int32_t bcol) {
        auto cGm = this->cGm[row * CUBE_BLOCK_M * N + bcol*CUBE_BLOCK_N];
        AscendC::LocalTensor<cType> c1Local = outQueueCO1.DeQue<cType>();

        AscendC::FixpipeParamsV220 params;
        params.ndNum = 1;
        params.mSize = CUBE_BLOCK_M;
        params.nSize = CUBE_BLOCK_N;
        params.srcStride = CUBE_BLOCK_M;
        params.dstStride = N;
        params.srcNdStride = 0;
        params.dstNdStride = 0;

        AscendC::SetAtomicAdd<cType>();
        AscendC::Fixpipe(cGm, c1Local, params);
        AscendC::SetAtomicNone();

        // AscendC::printf("Debug C Block: row %d, block col %d\n", row, progress);
        // uint32_t array[] = {static_cast<uint32_t>(16), static_cast<uint32_t>(32)};
        // AscendC::ShapeInfo shapeInfo(2, array); 
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
    //使用VECIN不知道为什么也不行，A1可以
    AscendC::TQue<AscendC::TPosition::A1, 1> rowBQueue;

    AscendC::GlobalTensor<int32_t> rowPtrGm;
    AscendC::GlobalTensor<int32_t> colGm;
    AscendC::GlobalTensor<aType> valGm;

    AscendC::GlobalTensor<bType> bGm;
    AscendC::GlobalTensor<cType> cGm;

    int32_t M;
    int32_t K;
    int32_t N;
    uint32_t rowWindowNum;
    uint32_t mmadCubeBlockNum;
    uint32_t lastKLength;
    
};

extern "C" __global__ __aicore__ void bcsr_spmm_custom(
    GM_ADDR a_shape, GM_ADDR row_ptr, GM_ADDR col, GM_ADDR val,
    GM_ADDR b, GM_ADDR c,
    GM_ADDR workspace, GM_ADDR tiling
) {
    GET_TILING_DATA(tiling_data, tiling);

    BcsrSpmmKernel<half, half, float,int32_t> op;
    op.Init(a_shape, row_ptr, col, val, b, c, workspace,
        tiling_data.M, tiling_data.N, tiling_data.K,
        tiling_data.formerNum, tiling_data.formerLength,
        tiling_data.tailNum, tiling_data.tailLength,
        tiling_data.lastKLength
    );
    op.Process();
}