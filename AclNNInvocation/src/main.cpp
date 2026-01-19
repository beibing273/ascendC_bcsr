/**
 * @file main.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>

#include "acl/acl.h"
#include "common.h"
#include "op_runner.h"
#include "re_handler.h"
#include "timer.h"

bool g_isDevice = false;
int deviceId = 0;

const int64_t TILE_M = 16;
const int64_t TILE_K = 16;

OperatorDesc CreateOpDesc(int64_t m, int64_t k, int64_t n, int64_t windowNum, int64_t blockNum)
{
    // define operator
    std::vector<int64_t> shapeRowPtr{windowNum + 1};
    //std::vector<int64_t> shapeCol{blockNum};
    std::vector<int64_t> shapeCol{blockNum*TILE_K};
    std::vector<int64_t> shapeValues{blockNum * TILE_M * TILE_K};
    std::vector<int64_t> shapeAShape{2};
    std::vector<int64_t> shapeB{k, n};
    std::vector<int64_t> shapeC{m, n};

    aclDataType dataTypeIndices = ACL_INT32;
    aclDataType dataTypeValues = ACL_FLOAT16;
    aclDataType dataTypeAShape = ACL_INT64;
    aclDataType dataTypeB = ACL_FLOAT16;
    aclDataType dataTypeC = ACL_FLOAT;

    aclFormat format = ACL_FORMAT_ND;

    OperatorDesc opDesc;
    opDesc.SetInputArrayNum(1);
    opDesc.AddInputTensorDesc(dataTypeAShape, shapeAShape.size(), shapeAShape.data(), format);
    opDesc.AddInputTensorDesc(dataTypeIndices, shapeRowPtr.size(), shapeRowPtr.data(), format);
    opDesc.AddInputTensorDesc(dataTypeIndices, shapeCol.size(), shapeCol.data(), format);
    opDesc.AddInputTensorDesc(dataTypeValues, shapeValues.size(), shapeValues.data(), format);
    opDesc.AddInputTensorDesc(dataTypeB, shapeB.size(), shapeB.data(), format);
    opDesc.AddOutputTensorDesc(dataTypeC, shapeC.size(), shapeC.data(), format);

    return opDesc;
}

bool SetInputData(OpRunner &runner, int64_t m, int64_t k, const std::string& rowPtrPath, const std::string& colPath, const std::string& valuesPath, const std::string& bPath)
{
    // set a_shape
    auto aShapePtr = runner.GetInputBuffer<int64_t>(0); // int64_t *
    aShapePtr[0] = m;
    aShapePtr[1] = k;

    size_t fileSize = 0;
    ReadFile(rowPtrPath.c_str(), fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1));
    ReadFile(colPath.c_str(), fileSize, runner.GetInputBuffer<void>(2), runner.GetInputSize(2));
    ReadFile(valuesPath.c_str(), fileSize, runner.GetInputBuffer<void>(3), runner.GetInputSize(3));
    ReadFile(bPath.c_str(), fileSize, runner.GetInputBuffer<void>(4), runner.GetInputSize(4));
    // INFO_LOG("Set input success");
    return true;
}
bool ReadReorderRef(const std::string& refPath,int64_t m,int64_t* reorder_ref){
    size_t filesize=0;
    int32_t* temp_buffer=new int32_t[m];
    ReadFile(refPath.c_str(),filesize,temp_buffer,m*sizeof(int32_t));
    try{
       for(int i=0;i<m;++i){
        reorder_ref[i]=static_cast<int64_t>(temp_buffer[i]);
    } 
    }
    catch(out_of_range oe){
        ERROR_LOG("reorder_ref size is error!\n");
        delete[] temp_buffer;
        return false;
    }
    catch(exception e){
        ERROR_LOG("reorder_ref udefined error\n");
        delete[] temp_buffer;
        return false;
    }

    delete[] temp_buffer;
    return true;
}
bool ProcessOutputData(OpRunner &runner, const std::string& outputCPath)
{
    WriteFile(outputCPath.c_str(), runner.GetOutputBuffer<void>(0), runner.GetOutputSize(0));
    // INFO_LOG("Write output success");
    return true;
}
bool ProcessOutputDataReorder(void* output,size_t output_size, const std::string& outputCPath)
{
    WriteFile(outputCPath.c_str(), output,output_size);
    // INFO_LOG("Write output success");
    return true;
}
void DestroyResource()
{
    bool flag = false;
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    // INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destroy resource failed");
    } else {
        // INFO_LOG("Destroy resource success");
    }
}

bool InitResource()
{
    std::string output = "../output";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            // INFO_LOG("Make output directory successfully");
        } else {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    if (aclInit(nullptr) != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }
    // INFO_LOG("Set device[%d] success", deviceId);

    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestroyResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    // INFO_LOG("Get RunMode[%d] success", runMode);

    return true;
}

bool RunOp(int64_t m, int64_t k, int64_t n, int64_t windowNum, int64_t blockNum, const std::string& rowPtr, const std::string& col, const std::string& values, const std::string reorder_ref, const std::string& b, const std::string& c,const std::string& smode)
{
    // create op desc
    OperatorDesc opDesc = CreateOpDesc(m, k, n, windowNum, blockNum);

    // create Runner
    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    // Load inputs
    if (!SetInputData(opRunner, m, k, rowPtr, col, values, b)) {
        ERROR_LOG("Set input data failed");
        return false;
    }

    // Run op
    Timer::Start("opRunner.RunOp");
    bool result = opRunner.RunOp();
    Timer::Stop("opRunner.RunOp");

    if (!result) {
        ERROR_LOG("Run op failed");
        return false;
    }
    // if(smode=="reorder"){
    //    ReHandler<float> rh(opRunner.GetOutputBuffer<float>(0), opRunner.GetOutputSize(0)/sizeof(float),m,n);
    //    int64_t* ref_data=new int64_t[m];
    //    ReadReorderRef(reorder_ref,m,ref_data);
    //    // process output data
    //    if (!ProcessOutputDataReorder(rh.reorder_and_get(ref_data,m),opRunner.GetOutputSize(0),c)) {
    //     ERROR_LOG("Process output data failed");
    //     return false;
    //    } 
    // }
    // else{
         if (!ProcessOutputData(opRunner, c)) {
           ERROR_LOG("Process output data failed");
           return false;
        }
    //}
    INFO_LOG("Run op success");
    return true;
}
int main(int argc, char **argv)
{
     if (argc < 14) {
        // std::cerr << "Usage: " << argv[0] << " <M> <K> <N> <NNZ> <row_ptr.bin> <col.bin> <values.bin> <b.bin> <c.bin> <category> <sample_name>" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <M> <K> <N> <WINDOW_NUM> <BLOCK_NUM> <row_ptr.bin> <col.bin> <values.bin> <b.bin> <c.bin> <category> <sample_name>" << std::endl;
        return FAILED;
    }

    int64_t m = std::stoll(argv[1]);
    int64_t k = std::stoll(argv[2]);
    int64_t n = std::stoll(argv[3]);
    int64_t windowNum = std::stoll(argv[4]);
    int64_t blockNum = std::stoll(argv[5]);
    std::string rowPtr = argv[6];
    std::string col = argv[7];
    std::string values = argv[8];
    
    std::string b = argv[9];
    std::string c = argv[10];
    std::string category = argv[11];
    std::string sampleName = argv[12];
    std::string smode=argv[13];
    std::string reorder_ref ="";
    if(smode=="reorder"){
        reorder_ref=argv[14];
        if (argc != 15) {
        // std::cerr << "Usage: " << argv[0] << " <M> <K> <N> <NNZ> <row_ptr.bin> <col.bin> <values.bin> <b.bin> <c.bin> <category> <sample_name>" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <M> <K> <N> <WINDOW_NUM> <BLOCK_NUM> <row_ptr.bin> <col.bin> <values.bin> <b.bin> <c.bin> <category> <sample_name> <mode> <reorder_ref>" << std::endl;
          return FAILED;
          }
    }else{
        if (argc != 14) {
        // std::cerr << "Usage: " << argv[0] << " <M> <K> <N> <NNZ> <row_ptr.bin> <col.bin> <values.bin> <b.bin> <c.bin> <category> <sample_name>" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <M> <K> <N> <WINDOW_NUM> <BLOCK_NUM> <row_ptr.bin> <col.bin> <values.bin> <b.bin> <c.bin> <category> <sample_name> <mode>" << std::endl;
        return FAILED;
        }
    }
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    // INFO_LOG("Init resource success");

    if (!RunOp(m, k, n, windowNum, blockNum, rowPtr, col, values,reorder_ref, b, c,smode)) {
        DestroyResource();
        return FAILED;
    }

    DestroyResource();

    Timer::CalculateAndRecordAll();
    Log::Write(category, sampleName, Timer::GetTimings(),smode);
    Timer::Clear();

    return SUCCESS;
}
