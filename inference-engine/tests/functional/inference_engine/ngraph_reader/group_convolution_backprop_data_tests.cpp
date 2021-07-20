// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadGroupConvolutionBackpropDataNetwork) {
    std::string model = R"V0G0N(
<net name="GroupConvBackpropData" version="10">
	<layers>
		<layer id="0" name="484/placeholder_port_0" type="Parameter" version="opset1">
			<data shape="1,64,64,64" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="910913_const" type="Const" version="opset1">
			<data offset="0" size="4096" shape="64,1,1,4,4" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="485" type="GroupConvolutionBackpropData" version="opset1">
			<data strides="2,2" dilations="1,1" pads_begin="1,1" pads_end="1,1" output_padding="0,0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="485/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
	</edges>
</net>
)V0G0N";
    std::string modelV7 = R"V0G0N(
<net name="GroupConvBackpropData" version="7">
	<layers>
		<layer id="0" name="484/placeholder_port_0" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="485" type="Deconvolution" version="opset1">
			<data group="64" strides="2,2" dilations="1,1" kernel="4,4" pads_begin="1,1" pads_end="1,1" output="64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>128</dim>
					<dim>128</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="4096" precision="FP32"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV7, 4096);
}