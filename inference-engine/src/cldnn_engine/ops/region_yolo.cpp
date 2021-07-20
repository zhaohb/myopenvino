// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/region_yolo.hpp"

#include "api/region_yolo.hpp"

namespace CLDNNPlugin {

void CreateRegionYoloOp(Program& p, const std::shared_ptr<ngraph::op::v0::RegionYolo>& op) {
    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    uint32_t coords = op->get_num_coords();
    uint32_t classes = op->get_num_classes();
    uint32_t num = op->get_num_regions();
    bool do_softmax = op->get_do_softmax();
    uint32_t mask_size = op->get_mask().size();

    auto regionPrim = cldnn::region_yolo(layerName,
                                         inputPrimitives[0],
                                         coords,
                                         classes,
                                         num,
                                         mask_size,
                                         do_softmax);

    p.AddPrimitive(regionPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, RegionYolo);

}  // namespace CLDNNPlugin
