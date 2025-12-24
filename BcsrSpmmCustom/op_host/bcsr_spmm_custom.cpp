
#include "bcsr_spmm_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

constexpr uint32_t MAX_MMAD_N = 32;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    BcsrSpmmCustomTilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    // a_shape, row_ptr, col, val, b
    auto shape_a_addr = context->GetInputTensor(0)->GetData<int64_t>();
    auto shape_b = context->GetInputTensor(4)->GetOriginShape();
    // int32_t M = shape_a_addr[0];
    // int32_t K = shape_a_addr[1];
    // int32_t N = shape_b.GetDim(1);

    // 备用实现
    // auto shape_b = context->GetInputTensor(4)->GetOriginShape();
    auto shape_c = context->GetOutputShape(0)->GetOriginShape();
    int32_t M = shape_c.GetDim(0);
    int32_t K = shape_b.GetDim(0);
    int32_t N = shape_b.GetDim(1);

    tiling.set_M(M);
    tiling.set_N(N);
    tiling.set_K(K);

    // totalLength 行窗口数
    uint32_t totalLength = context->GetInputShape(1)->GetOriginShape().GetShapeSize() - 1;
    uint32_t blockDim = ascendcPlatform.GetCoreNumAic();    // Cube core 数量
    blockDim = blockDim > totalLength ? totalLength : blockDim;
    context->SetBlockDim(blockDim);
    // context->SetBlockDim(1);
    tiling.set_totalLength(totalLength);

    uint32_t formerNum = totalLength % blockDim;
    if (formerNum == 0) {
        formerNum = blockDim;
    }
    uint32_t formerLength = (totalLength + blockDim - 1) / blockDim;
    uint32_t tailNum = blockDim - formerNum;
    uint32_t tailLength = totalLength / blockDim;
    tiling.set_formerNum(formerNum);
    tiling.set_formerLength(formerLength);
    tiling.set_tailNum(tailNum);
    tiling.set_tailLength(tailLength);

    printf("BcsrSpmmCustom Tiling: M=%d, K=%d, N=%d, totalLength=%d, blockDim=%d, formerNum=%d, formerLength=%d, tailNum=%d, tailLength=%d\n",
        M, K, N, totalLength, blockDim, formerNum, formerLength, tailNum, tailLength
    );

    uint32_t alignNum = 32 / sizeof(uint16_t);
    // mmad相关参数计算
    uint32_t mmadN = MAX_MMAD_N;
    uint32_t mmadNum = (N + mmadN - 1) / mmadN;
    uint32_t lastMmadN = N - (mmadNum - 1) * mmadN;
    uint32_t lastMmadCubeBlockNum = (lastMmadN + alignNum - 1) / alignNum;
    tiling.set_mmadNum(mmadNum);
    tiling.set_mmadN(mmadN);
    tiling.set_lastMmadN(lastMmadN);
    tiling.set_lastMmadCubeBlockNum(lastMmadCubeBlockNum);

    // 处理K不对齐
    uint32_t lastKLength = K % alignNum;
    if (lastKLength == 0) {
        lastKLength = alignNum;
    }
    tiling.set_lastKLength(lastKLength);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    // a_shape, row_ptr, col, val, b
    auto a_shape_addr = context->GetInputTensor(0)->GetData<int64_t>();
    auto b_shape = context->GetInputShape(4);
    auto c_shape = context->GetOutputShape(0);
    if (b_shape == nullptr || a_shape_addr == nullptr || c_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int M = a_shape_addr[0];
    int N = b_shape->GetDim(1);
    c_shape->SetDimNum(2);
    c_shape->SetDim(0, M);
    c_shape->SetDim(1, N);

    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context->SetOutputDataType(0, ge::DataType::DT_FLOAT) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class BcsrSpmmCustom : public OpDef {
public:
    explicit BcsrSpmmCustom(const char* name) : OpDef(name)
    {
        this->Input("a_shape")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .ValueDepend(REQUIRED); // 声明 a_shape 输入为数据依赖输入
        this->Input("row_ptr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        this->Input("col")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        this->Input("val")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(BcsrSpmmCustom);
}
