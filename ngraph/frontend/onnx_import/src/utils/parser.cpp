//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <onnx/onnx_pb.h>

#include "ngraph/except.hpp"
#include "utils/parser.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        ONNX_NAMESPACE::ModelProto parse_from_file(const std::string& file_path)
        {
            std::ifstream file_stream{file_path, std::ios::in | std::ios::binary};

            if (!file_stream.is_open())
            {
                throw ngraph_error("Could not open the file: " + file_path);
            };

            return parse_from_istream(file_stream);
        }

        ONNX_NAMESPACE::ModelProto parse_from_istream(std::istream& model_stream)
        {
            if (!model_stream.good())
            {
                model_stream.clear();
                model_stream.seekg(0);
                if (!model_stream.good())
                {
                    throw ngraph_error("Provided input stream has incorrect state.");
                }
            }

            ONNX_NAMESPACE::ModelProto model_proto;
            if (!model_proto.ParseFromIstream(&model_stream))
            {
#ifdef NGRAPH_USE_PROTOBUF_LITE
                throw ngraph_error(
                    "Error during import of ONNX model provided as input stream "
                    " with binary protobuf message.");
#else
                // Rewind to the beginning and clear stream state.
                model_stream.clear();
                model_stream.seekg(0);
                google::protobuf::io::IstreamInputStream iistream(&model_stream);
                // Try parsing input as a prototxt message
                if (!google::protobuf::TextFormat::Parse(&iistream, &model_proto))
                {
                    throw ngraph_error(
                        "Error during import of ONNX model provided as input stream with prototxt "
                        "protobuf message.");
                }
#endif
            }

            return model_proto;
        }
    } // namespace onnx_import
} // namespace ngraph
