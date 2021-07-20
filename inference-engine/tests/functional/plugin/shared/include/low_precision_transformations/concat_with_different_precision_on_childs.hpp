// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {
class ConcatWithDifferentChildsTransformationParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData1;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData2;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    std::string, // target device: CPU, GPU
    ConcatWithDifferentChildsTransformationParam,
    ngraph::pass::low_precision::LayerTransformation::Params, // transformation parameters
    // multichannel
    bool> ConcatWithDifferentChildsTransformationParams;

class ConcatWithDifferentChildsTransformation :
    public testing::WithParamInterface<ConcatWithDifferentChildsTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConcatWithDifferentChildsTransformationParams> obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;

private:
    void validate();
};

}  // namespace LayerTestsDefinitions
