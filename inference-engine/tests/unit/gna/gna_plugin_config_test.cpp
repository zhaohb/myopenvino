// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gna/gna_config.hpp>
#include "gna_plugin_config.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <map>

using namespace InferenceEngine;
using namespace GNAPluginNS;

const std::map<std::string, std::string>  supportedConfigKeysWithDefaults = {
    {GNA_CONFIG_KEY(SCALE_FACTOR), "1.000000"},
    {GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_0"), "1.000000"},
    {GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE), ""},
    {GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE_GENERATION), ""},
    {GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_SW_EXACT},
    {GNA_CONFIG_KEY(COMPACT_MODE), CONFIG_VALUE(NO)},
    {CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(NO)},
    {GNA_CONFIG_KEY(PRECISION), Precision(Precision::I16).name()},
    {GNA_CONFIG_KEY(PWL_UNIFORM_DESIGN), CONFIG_VALUE(NO)},
    {CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(NO)},
    {GNA_CONFIG_KEY(LIB_N_THREADS), "1"},
    {CONFIG_KEY(SINGLE_THREAD), CONFIG_VALUE(YES)}
};

class GNAPluginConfigTest : public ::testing::Test {
protected:
    Config config;
    void SetAndCompare(const std::string& key, const std::string& val) {
        config.UpdateFromMap({{key, val}});
        EXPECT_EQ(config.GetParameter(key), val);
    }
    void ExpectThrow(const std::string& key, const std::string& val) {
        EXPECT_THROW(config.UpdateFromMap({{key, val}}),
                     details::InferenceEngineException);
    }
    void SetAndCheckFlag(const std::string& key, bool& val, bool reverse = false) {
        const bool yes = reverse ? false : true;
        const bool no = !yes;
        SetAndCompare(key, CONFIG_VALUE(YES));
        EXPECT_EQ(val, yes);
        SetAndCompare(key, CONFIG_VALUE(NO));
        EXPECT_EQ(val, no);
        SetAndCompare(key, CONFIG_VALUE(YES));
        EXPECT_EQ(val, yes);
        ExpectThrow(key, "abc");
        ExpectThrow(key, "");
    }
};

TEST_F(GNAPluginConfigTest, GnaConfigDefaultConfigIsExpected) {
    ASSERT_EQ(config.keyConfigMap, supportedConfigKeysWithDefaults);
}

TEST_F(GNAPluginConfigTest, GnaConfigScaleFactorTest) {
    config.UpdateFromMap({{GNA_CONFIG_KEY(SCALE_FACTOR), std::string("34")}});
    EXPECT_EQ(config.GetParameter(GNA_CONFIG_KEY(SCALE_FACTOR)), std::string("34.000000"));
    EXPECT_EQ(config.GetParameter(GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_0")), std::string("34.000000"));
    EXPECT_EQ(config.inputScaleFactors.size(), 1);
    EXPECT_FLOAT_EQ(config.inputScaleFactors[0], 34.0);

    config.UpdateFromMap({{GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_3"), std::string("15.2")}});
    EXPECT_EQ(config.GetParameter(GNA_CONFIG_KEY(SCALE_FACTOR)), std::string("34.000000"));
    EXPECT_EQ(config.GetParameter(GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_0")), std::string("34.000000"));
    EXPECT_EQ(config.GetParameter(GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_1")), std::string("1.000000"));
    EXPECT_EQ(config.GetParameter(GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_2")), std::string("1.000000"));
    EXPECT_EQ(config.GetParameter(GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_3")), std::string("15.200000"));
    EXPECT_EQ(config.inputScaleFactors.size(), 4);
    EXPECT_FLOAT_EQ(config.inputScaleFactors[0], 34.0);
    EXPECT_FLOAT_EQ(config.inputScaleFactors[1], 1.0);
    EXPECT_FLOAT_EQ(config.inputScaleFactors[2], 1.0);
    EXPECT_FLOAT_EQ(config.inputScaleFactors[3], 15.2);

    config.UpdateFromMap({{GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_99"), std::string("8.43")}});
    EXPECT_EQ(config.GetParameter(GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_99")), std::string("8.430000"));
    EXPECT_EQ(config.inputScaleFactors.size(), 100);
    EXPECT_FLOAT_EQ(config.inputScaleFactors[99], 8.43);

    ExpectThrow(GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_100"), std::string("8.43"));
    ExpectThrow(GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("&1"), std::string("8.43"));
    ExpectThrow(GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_"), std::string("8.43"));
    ExpectThrow(GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("abs"), std::string("8.43"));
    ExpectThrow(GNA_CONFIG_KEY(SCALE_FACTOR), std::string("abc"));
    ExpectThrow(GNA_CONFIG_KEY(SCALE_FACTOR), std::string("0"));
}

TEST_F(GNAPluginConfigTest, GnaConfigFirmwareModelImageTest) {
    SetAndCompare(GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE), "abc");
    EXPECT_EQ(config.dumpXNNPath, "abc");
}

TEST_F(GNAPluginConfigTest, GnaConfigFirmwareModelImageGeneratorTest) {
    SetAndCompare(GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE_GENERATION), "def");
    EXPECT_EQ(config.dumpXNNGeneration, "def");
}

TEST_F(GNAPluginConfigTest, GnaConfigDeviceModeTest) {
    SetAndCompare(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_HW);
#if GNA_LIB_VER == 1
    EXPECT_EQ(config.gna_proc_type, static_cast<intel_gna_proc_t>(GNA_HARDWARE));
#else
    EXPECT_EQ(config.pluginGna2AccMode, Gna2AccelerationModeHardware);
    EXPECT_EQ(config.pluginGna2DeviceConsistent, Gna2DeviceVersionSoftwareEmulation);
#endif
    SetAndCompare(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_SW);
#if GNA_LIB_VER == 1
    EXPECT_EQ(config.gna_proc_type, static_cast<intel_gna_proc_t>(GNA_SOFTWARE));
#else
    EXPECT_EQ(config.pluginGna2AccMode, Gna2AccelerationModeSoftware);
    EXPECT_EQ(config.pluginGna2DeviceConsistent, Gna2DeviceVersionSoftwareEmulation);
#endif
    SetAndCompare(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_SW_EXACT);
#if GNA_LIB_VER == 1
    EXPECT_EQ(config.gna_proc_type, static_cast<intel_gna_proc_t>(GNA_SOFTWARE & GNA_HARDWARE));
#else
    EXPECT_EQ(config.pluginGna2AccMode, Gna2AccelerationModeSoftware);
    EXPECT_EQ(config.pluginGna2DeviceConsistent, Gna2DeviceVersion1_0);
#endif
    SetAndCompare(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_AUTO);
#if GNA_LIB_VER == 1
    EXPECT_EQ(config.gna_proc_type, static_cast<intel_gna_proc_t>(GNA_AUTO));
#else
    EXPECT_EQ(config.pluginGna2AccMode, Gna2AccelerationModeAuto);
    EXPECT_EQ(config.pluginGna2DeviceConsistent, Gna2DeviceVersionSoftwareEmulation);
#endif
    ExpectThrow(GNA_CONFIG_KEY(DEVICE_MODE), "");
    ExpectThrow(GNA_CONFIG_KEY(DEVICE_MODE), "abc");
}

TEST_F(GNAPluginConfigTest, GnaConfigCompactMode) {
    SetAndCheckFlag(GNA_CONFIG_KEY(COMPACT_MODE),
                    config.gnaFlags.compact_mode);
}

TEST_F(GNAPluginConfigTest, GnaConfigExclusiveAsyncRequestTest) {
    SetAndCheckFlag(CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS),
                    config.gnaFlags.exclusive_async_requests);
}

TEST_F(GNAPluginConfigTest, GnaConfigPrecisionTest) {
    SetAndCompare(GNA_CONFIG_KEY(PRECISION), Precision(Precision::I8).name());
    EXPECT_EQ(config.gnaPrecision, Precision::I8);
    SetAndCompare(GNA_CONFIG_KEY(PRECISION), Precision(Precision::I16).name());
    EXPECT_EQ(config.gnaPrecision, Precision::I16);
    ExpectThrow(GNA_CONFIG_KEY(PRECISION), Precision(Precision::FP32).name());
    ExpectThrow(GNA_CONFIG_KEY(PRECISION), "");
}

TEST_F(GNAPluginConfigTest, GnaConfigPwlUniformDesignTest) {
    SetAndCheckFlag(GNA_CONFIG_KEY(PWL_UNIFORM_DESIGN),
                    config.gnaFlags.uniformPwlDesign);
}

TEST_F(GNAPluginConfigTest, GnaConfigPerfCountTest) {
    SetAndCheckFlag(CONFIG_KEY(PERF_COUNT),
                    config.gnaFlags.performance_counting);
}

TEST_F(GNAPluginConfigTest, GnaConfigLibNThreadsTest) {
    SetAndCompare(GNA_CONFIG_KEY(LIB_N_THREADS), "2");
    EXPECT_EQ(config.gnaFlags.gna_lib_async_threads_num, 2);
    SetAndCompare(GNA_CONFIG_KEY(LIB_N_THREADS), "25");
    EXPECT_EQ(config.gnaFlags.gna_lib_async_threads_num, 25);
    ExpectThrow(GNA_CONFIG_KEY(LIB_N_THREADS), "");
    ExpectThrow(GNA_CONFIG_KEY(LIB_N_THREADS), "0");
    ExpectThrow(GNA_CONFIG_KEY(LIB_N_THREADS), "128");
    ExpectThrow(GNA_CONFIG_KEY(LIB_N_THREADS), "abc");
}

TEST_F(GNAPluginConfigTest, GnaConfigSingleThreadTest) {
    SetAndCheckFlag(CONFIG_KEY(SINGLE_THREAD),
                    config.gnaFlags.gna_openmp_multithreading,
                    true);
}
