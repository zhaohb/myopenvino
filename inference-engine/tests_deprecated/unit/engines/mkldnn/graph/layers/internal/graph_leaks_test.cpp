// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_graph.hpp"

#include <mkldnn_plugin.h>
#include "mkldnn_exec_network.h"
#include <ie_core.hpp>
#include <mkldnn_extension_utils.h>
#include <config.h>

using namespace std;
using namespace mkldnn;

class MKLDNNTestExecNetwork: public MKLDNNPlugin::MKLDNNExecNetwork {
public:
    MKLDNNPlugin::MKLDNNGraph& getGraph() {
        return *(_graphs.begin()->get());
    }
};

struct TestExecutableNetworkBase : public InferenceEngine::ExecutableNetworkBase {
    using InferenceEngine::ExecutableNetworkBase::_impl;
    ~TestExecutableNetworkBase() override = default;
};

static MKLDNNPlugin::MKLDNNGraph& getGraph(InferenceEngine::IExecutableNetwork::Ptr execNetwork) {
    return reinterpret_cast<MKLDNNTestExecNetwork*>(
        reinterpret_cast<TestExecutableNetworkBase*>(
            execNetwork.get())->_impl.get())->getGraph();
}

class MKLDNNGraphLeaksTests: public ::testing::Test {
protected:
    void addOutputToEachNode(InferenceEngine::CNNNetwork& network, std::vector<std::string>& new_outputs,
                             InferenceEngine::CNNLayerPtr cnnLayer) {
        auto outputs = network.getOutputsInfo();
        if (outputs.find(cnnLayer->name) != outputs.end())
            return;

        network.addOutput(cnnLayer->name);
        new_outputs.push_back(cnnLayer->name);

        for (const auto &layer : cnnLayer->outData) {
            for (const auto &data : getInputTo(layer)) {
                addOutputToEachNode(network, new_outputs, data.second);
            }
        }
    }

    void fill_data(float *data, size_t size, size_t duty_ratio = 10) {
        for (size_t i = 0; i < size; i++) {
            if ( ( i / duty_ratio)%2 == 1) {
                data[i] = 0.0;
            } else {
                data[i] = (float) sin((float)i);
            }
        }
    }
};

TEST_F(MKLDNNGraphLeaksTests, MKLDNN_not_release_outputs_fp32) {
    try {
        std::string model = "<net name=\"LeNet\" version=\"2\" batch=\"1\">\n"
                "    <layers>\n"
                "        <layer name=\"data\" type=\"Input\" precision=\"FP32\" id=\"0\">\n"
                "            <output>\n"
                "                <port id=\"0\">\n"
                "                    <dim>1</dim>\n"
                "                    <dim>1</dim>\n"
                "                    <dim>28</dim>\n"
                "                    <dim>28</dim>\n"
                "                </port>\n"
                "            </output>\n"
                "        </layer>\n"
                "        <layer name=\"conv1\" type=\"Convolution\" precision=\"FP32\" id=\"1\">\n"
                "            <convolution_data stride-x=\"1\" stride-y=\"1\" pad-x=\"0\" pad-y=\"0\" kernel-x=\"5\" kernel-y=\"5\" output=\"20\" group=\"1\"/>\n"
                "            <input>\n"
                "                <port id=\"1\">\n"
                "                    <dim>1</dim>\n"
                "                    <dim>1</dim>\n"
                "                    <dim>28</dim>\n"
                "                    <dim>28</dim>\n"
                "                </port>\n"
                "            </input>\n"
                "            <output>\n"
                "                <port id=\"2\">\n"
                "                    <dim>1</dim>\n"
                "                    <dim>20</dim>\n"
                "                    <dim>24</dim>\n"
                "                    <dim>24</dim>\n"
                "                </port>\n"
                "            </output>\n"
                "            <weights offset=\"0\" size=\"2000\"/>\n"
                "            <biases offset=\"2000\" size=\"80\"/>\n"
                "        </layer>\n"
                "        <layer name=\"pool1\" type=\"Pooling\" precision=\"FP32\" id=\"2\">\n"
                "            <pooling_data kernel-x=\"2\" kernel-y=\"2\" pad-x=\"0\" pad-y=\"0\" stride-x=\"2\" stride-y=\"2\" rounding-type=\"ceil\" pool-method=\"max\"/>\n"
                "            <input>\n"
                "                <port id=\"3\">\n"
                "                    <dim>1</dim>\n"
                "                    <dim>20</dim>\n"
                "                    <dim>24</dim>\n"
                "                    <dim>24</dim>\n"
                "                </port>\n"
                "            </input>\n"
                "            <output>\n"
                "                <port id=\"4\">\n"
                "                    <dim>1</dim>\n"
                "                    <dim>20</dim>\n"
                "                    <dim>12</dim>\n"
                "                    <dim>12</dim>\n"
                "                </port>\n"
                "            </output>\n"
                "        </layer>\n"
                "        <layer name=\"conv2\" type=\"Convolution\" precision=\"FP32\" id=\"3\">\n"
                "            <convolution_data stride-x=\"1\" stride-y=\"1\" pad-x=\"0\" pad-y=\"0\" kernel-x=\"5\" kernel-y=\"5\" output=\"50\" group=\"1\"/>\n"
                "            <input>\n"
                "                <port id=\"5\">\n"
                "                    <dim>1</dim>\n"
                "                    <dim>20</dim>\n"
                "                    <dim>12</dim>\n"
                "                    <dim>12</dim>\n"
                "                </port>\n"
                "            </input>\n"
                "            <output>\n"
                "                <port id=\"6\">\n"
                "                    <dim>1</dim>\n"
                "                    <dim>50</dim>\n"
                "                    <dim>8</dim>\n"
                "                    <dim>8</dim>\n"
                "                </port>\n"
                "            </output>\n"
                "            <weights offset=\"2080\" size=\"100000\"/>\n"
                "            <biases offset=\"102080\" size=\"200\"/>\n"
                "        </layer>\n"
                "        <layer name=\"pool2\" type=\"Pooling\" precision=\"FP32\" id=\"4\">\n"
                "            <pooling_data kernel-x=\"2\" kernel-y=\"2\" pad-x=\"0\" pad-y=\"0\" stride-x=\"2\" stride-y=\"2\" rounding-type=\"ceil\" pool-method=\"max\"/>\n"
                "            <input>\n"
                "                <port id=\"7\">\n"
                "                    <dim>1</dim>\n"
                "                    <dim>50</dim>\n"
                "                    <dim>8</dim>\n"
                "                    <dim>8</dim>\n"
                "                </port>\n"
                "            </input>\n"
                "            <output>\n"
                "                <port id=\"8\">\n"
                "                    <dim>1</dim>\n"
                "                    <dim>50</dim>\n"
                "                    <dim>4</dim>\n"
                "                    <dim>4</dim>\n"
                "                </port>\n"
                "            </output>\n"
                "        </layer>\n"
                "        <layer name=\"ip1\" type=\"FullyConnected\" precision=\"FP32\" id=\"5\">\n"
                "            <fc_data out-size=\"500\"/>\n"
                "            <input>\n"
                "                <port id=\"9\">\n"
                "                    <dim>1</dim>\n"
                "                    <dim>50</dim>\n"
                "                    <dim>4</dim>\n"
                "                    <dim>4</dim>\n"
                "                </port>\n"
                "            </input>\n"
                "            <output>\n"
                "                <port id=\"10\">\n"
                "                    <dim>1</dim>\n"
                "                    <dim>500</dim>\n"
                "                </port>\n"
                "            </output>\n"
                "            <weights offset=\"102280\" size=\"1600000\"/>\n"
                "            <biases offset=\"1702280\" size=\"2000\"/>\n"
                "        </layer>\n"
                "        <layer name=\"relu1\" type=\"ReLU\" precision=\"FP32\" id=\"6\">\n"
                "            <input>\n"
                "                <port id=\"11\">\n"
                "                    <dim>1</dim>\n"
                "                    <dim>500</dim>\n"
                "                </port>\n"
                "            </input>\n"
                "            <output>\n"
                "                <port id=\"12\">\n"
                "                    <dim>1</dim>\n"
                "                    <dim>500</dim>\n"
                "                </port>\n"
                "            </output>\n"
                "        </layer>\n"
                "        <layer name=\"ip2\" type=\"FullyConnected\" precision=\"FP32\" id=\"7\">\n"
                "            <fc_data out-size=\"10\"/>\n"
                "            <input>\n"
                "                <port id=\"13\">\n"
                "                    <dim>1</dim>\n"
                "                    <dim>500</dim>\n"
                "                </port>\n"
                "            </input>\n"
                "            <output>\n"
                "                <port id=\"14\">\n"
                "                    <dim>1</dim>\n"
                "                    <dim>10</dim>\n"
                "                </port>\n"
                "            </output>\n"
                "            <weights offset=\"1704280\" size=\"20000\"/>\n"
                "            <biases offset=\"1724280\" size=\"40\"/>\n"
                "        </layer>\n"
                "        <layer name=\"prob\" type=\"SoftMax\" precision=\"FP32\" id=\"8\">\n"
                "            <input>\n"
                "                <port id=\"15\">\n"
                "                    <dim>1</dim>\n"
                "                    <dim>10</dim>\n"
                "                </port>\n"
                "            </input>\n"
                "            <output>\n"
                "                <port id=\"16\">\n"
                "                    <dim>1</dim>\n"
                "                    <dim>10</dim>\n"
                "                </port>\n"
                "            </output>\n"
                "        </layer>\n"
                "    </layers>\n"
                "    <edges>\n"
                "        <edge from-layer=\"0\" from-port=\"0\" to-layer=\"1\" to-port=\"1\"/>\n"
                "        <edge from-layer=\"1\" from-port=\"2\" to-layer=\"2\" to-port=\"3\"/>\n"
                "        <edge from-layer=\"2\" from-port=\"4\" to-layer=\"3\" to-port=\"5\"/>\n"
                "        <edge from-layer=\"3\" from-port=\"6\" to-layer=\"4\" to-port=\"7\"/>\n"
                "        <edge from-layer=\"4\" from-port=\"8\" to-layer=\"5\" to-port=\"9\"/>\n"
                "        <edge from-layer=\"5\" from-port=\"10\" to-layer=\"6\" to-port=\"11\"/>\n"
                "        <edge from-layer=\"6\" from-port=\"12\" to-layer=\"7\" to-port=\"13\"/>\n"
                "        <edge from-layer=\"7\" from-port=\"14\" to-layer=\"8\" to-port=\"15\"/>\n"
                "    </edges>\n"
                "</net>";

        size_t weights_size = 1724320;

        InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {weights_size}, InferenceEngine::C });
        weights->allocate();
        fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
        InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

        InferenceEngine::Core core;
        InferenceEngine::CNNNetwork network;
        ASSERT_NO_THROW(network = core.ReadNetwork(model, weights_ptr));

        auto outputs = network.getOutputsInfo();
        std::vector<std::string> new_outputs;

        for (auto input : network.getInputsInfo()) {
            for (const auto &layer : getInputTo(input.second->getInputData())) {
                addOutputToEachNode(network, new_outputs, layer.second);
            }
        }

        ASSERT_NE(1, network.getOutputsInfo().size());

        std::shared_ptr<MKLDNNPlugin::Engine> score_engine(new MKLDNNPlugin::Engine());
        InferenceEngine::ExecutableNetwork exeNetwork1;
        ASSERT_NO_THROW(exeNetwork1 = score_engine->LoadNetwork(network, {}));

        size_t modified_outputs_size = getGraph(exeNetwork1).GetOutputNodes().size();

        InferenceEngine::CNNNetwork network2;
        ASSERT_NO_THROW(network2 = core.ReadNetwork(model, weights_ptr));
        ASSERT_EQ(1, network2.getOutputsInfo().size());

        InferenceEngine::ExecutableNetwork exeNetwork2;
        ASSERT_NO_THROW(exeNetwork2 = score_engine->LoadNetwork(network2, {}));

        size_t original_outputs_size = getGraph(exeNetwork2).GetOutputNodes().size();

        ASSERT_NE(modified_outputs_size, original_outputs_size);
        ASSERT_EQ(1, original_outputs_size);
    } catch (std::exception& e) {
        FAIL() << e.what();
    } catch (...) {
        FAIL();
    }
}
