// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <legacy/net_pass.h>
#include "common_test_utils/common_utils.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;

class LocaleTests : public ::testing::Test {
    std::string originalLocale;
    std::string _model = R"V0G0N(
<net name="Power_Only" version="3" precision="FP32" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer name="power" type="Power" precision="FP32" id="1">
            <data scale="0.75"
                  shift="0.35"
                  power="0.5"/>

            <input>
                <port id="1">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer name="sum" type="Eltwise" precision="FP32" id="2">
            <data coeff="0.77,0.33"/>
            <input>
                <port id="1">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
                <port id="2">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="1"/>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
    </edges>
    <pre-process reference-layer-name="data">
        <channel id="0">
            <mean value="104.006"/>
            <scale value="0.1"/>
        </channel>
        <channel id="1">
            <mean value="116.668"/>
            <scale value="0.2"/>
        </channel>
        <channel id="2">
            <mean value="122.678"/>
            <scale value="0.3"/>
        </channel>
    </pre-process>

</net>
)V0G0N";

    std::string _model_LSTM = R"V0G0N(
 <net batch="1" name="model" version="2">
    <layers>
        <layer id="0" name="Input" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>30</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Split" precision="FP32" type="Split">
            <data axis="1" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>30</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="LSTMCell" precision="FP32" type="LSTMCell">
            <data hidden_size="10" clip="0.2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
            <blobs>
                <weights offset="0" size="3200"/>
                <biases offset="3200" size="160"/>
            </blobs>
        </layer>
        <layer name="Eltwise" type="Eltwise" id="3" precision="FP32">
            <data operation="sum" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
            <edge from-layer="1" from-port="2" to-layer="2" to-port="1"/>
            <edge from-layer="1" from-port="3" to-layer="2" to-port="2"/>
            <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
            <edge from-layer="2" from-port="4" to-layer="3" to-port="1"/>
        </edges>
    </net>
)V0G0N";

protected:
    void SetUp() override {
        originalLocale  = setlocale(LC_ALL, nullptr);
    }
    void TearDown() override  {
        setlocale(LC_ALL, originalLocale.c_str());
    }

    void testBody(bool isLSTM = false) const {
        InferenceEngine::Core core;

        // This model contains layers with float attributes.
        // Conversion from string may be affected by locale.
        std::string model = isLSTM ? _model_LSTM : _model;
        auto blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {3360}, Layout::C));
        blob->allocate();
        auto net = core.ReadNetwork(model, blob);

        IE_SUPPRESS_DEPRECATED_START
        if (!isLSTM) {
            auto power_layer = dynamic_pointer_cast<PowerLayer>(CommonTestUtils::getLayerByName(net, "power"));
            ASSERT_EQ(power_layer->scale, 0.75f);
            ASSERT_EQ(power_layer->offset, 0.35f);
            ASSERT_EQ(power_layer->power, 0.5f);

            auto sum_layer = dynamic_pointer_cast<EltwiseLayer>(CommonTestUtils::getLayerByName(net, "sum"));
            std::vector<float> ref_coeff{0.77f, 0.33f};
            ASSERT_EQ(sum_layer->coeff, ref_coeff);

            auto info = net.getInputsInfo();
            auto preproc = info.begin()->second->getPreProcess();
            ASSERT_EQ(preproc[0]->stdScale, 0.1f);
            ASSERT_EQ(preproc[0]->meanValue, 104.006f);
        } else {
            InferenceEngine::NetPass::UnrollRNN_if(net, [] (const RNNCellBase& rnn) -> bool { return true; });
            auto lstmcell_layer = dynamic_pointer_cast<ClampLayer>(CommonTestUtils::getLayerByName(net, "LSTMCell:split_clip"));

            float ref_coeff = 0.2f;
            ASSERT_EQ(lstmcell_layer->min_value, -ref_coeff);
            ASSERT_EQ(lstmcell_layer->max_value,  ref_coeff);

            ASSERT_EQ(lstmcell_layer->GetParamAsFloat("min"), -ref_coeff);
            ASSERT_EQ(lstmcell_layer->GetParamAsFloat("max"),  ref_coeff);
        }
        IE_SUPPRESS_DEPRECATED_END
    }
};

TEST_F(LocaleTests, WithRULocale) {
    setlocale(LC_ALL, "ru_RU.UTF-8");
    testBody();
}

TEST_F(LocaleTests, WithUSLocale) {
    setlocale(LC_ALL, "en_US.UTF-8");
    testBody();
}

TEST_F(LocaleTests, WithRULocaleOnLSTM) {
    setlocale(LC_ALL, "ru_RU.UTF-8");
    testBody(true);
}

TEST_F(LocaleTests, WithUSLocaleOnLSTM) {
    setlocale(LC_ALL, "en_US.UTF-8");
    testBody(true);
}

TEST_F(LocaleTests, DISABLED_WithRULocaleCPP) {
    auto prev = std::locale();
    std::locale::global(std::locale("ru_RU.UTF-8"));
    testBody();
    std::locale::global(prev);
}

TEST_F(LocaleTests, DISABLED_WithUSLocaleCPP) {
    auto prev = std::locale();
    std::locale::global(std::locale("en_US.UTF-8"));
    testBody();
    std::locale::global(prev);
}
