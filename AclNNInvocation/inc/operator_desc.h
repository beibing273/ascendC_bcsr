/**
 * @file operator_desc.h
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef OPERATOR_DESC_H
#define OPERATOR_DESC_H

#include <string>
#include <vector>

#include "acl/acl.h"

/**
 * Op description
 */
struct OperatorDesc {
    /**
     * Constructor
     */
    explicit OperatorDesc();

    /**
     * Destructor
     */
    virtual ~OperatorDesc();

    /**
     * Add an input tensor description
     * @param [in] dataType: data type
     * @param [in] numDims: number of dims
     * @param [in] dims: dims
     * @param [in] format: format
     * @return OperatorDesc
     */
    OperatorDesc &AddInputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format);
    OperatorDesc& AddAttr(void* attr);
    /**
     * Add an output tensor description
     * @param [in] dataType: data type
     * @param [in] numDims: number of dims
     * @param [in] dims: dims
     * @param [in] format: format
     * @return OperatorDesc
     */
    OperatorDesc &AddOutputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format);

    void SetInputArrayNum(size_t num)
    {
        numInputArray = num;
    }
    void SetAttrNum(size_t num){
        numAttr=num;
    }
    std::string opType;
    std::vector<aclTensorDesc *> inputDesc;
    std::vector<aclTensorDesc *> outputDesc;
    std::vector<void *> attrDesc;
    // no acl array descriptions
    size_t numInputArray = 0;
    size_t numAttr=0;
};

#endif // OPERATOR_DESC_H
