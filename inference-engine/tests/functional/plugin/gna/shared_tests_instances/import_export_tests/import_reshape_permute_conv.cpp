// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "import_export_tests/import_reshape_permute_conv.hpp"

#include <fstream>
#include <stdio.h>

using namespace LayerTestsDefinitions;

namespace {

class ImportReshapePermuteConvGNA : public ImportReshapePermuteConv {
private:
    void exportImportNetwork() override {
        executableNetwork.Export(fileName);
        std::fstream inputStream(fileName, std::ios_base::in | std::ios_base::binary);
        if (inputStream.fail()) {
            FAIL() << "Cannot open file to import model: " << fileName;
        }
        executableNetwork = core->ImportNetwork(inputStream, targetDevice, configuration);
    }
protected:
    void TearDown() override {
        if (remove(fileName.c_str()) != 0) {
            FAIL() << "Error: could not delete file " << fileName;
        }
    }

private:
    std::string fileName = "exported_model.blob";
};

TEST_P(ImportReshapePermuteConvGNA, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> exportConfigs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "327.67"}
    }
};

const std::vector<std::map<std::string, std::string>> importConfigs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "32767"}
    },
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "327.67"}
    },
};

INSTANTIATE_TEST_CASE_P(smoke_ImportNetworkCase, ImportReshapePermuteConvGNA,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(exportConfigs),
                            ::testing::ValuesIn(importConfigs)),
                        ImportReshapePermuteConvGNA::getTestCaseName);

} // namespace
