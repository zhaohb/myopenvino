// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_roi_pooling_node.h"

#include <legacy/ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <math.h>
#include <mkldnn_extension_utils.h>
#include <cpu/x64/jit_generator.hpp>
#include "ie_parallel.hpp"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu::x64;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_roi_pooling_call_args, field)

template <cpu_isa_t isa>
struct jit_uni_roi_pooling_kernel_f32 : public jit_uni_roi_pooling_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_roi_pooling_kernel_f32)

    explicit jit_uni_roi_pooling_kernel_f32(jit_roi_pooling_params jcp) : jit_uni_roi_pooling_kernel(jcp), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    };

    void generate() override {
        this->preamble();

        Label exit_label;
        Label tail_label;

        mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
        mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);

        mov(reg_bin_area, ptr[this->param1 + GET_OFF(bin_area)]);
        mov(reg_c_blocks, ptr[this->param1 + GET_OFF(c_blocks)]);

        if (jpp_.alg == ROIPoolingOpType::Max) {
            mov(reg_kh, ptr[this->param1 + GET_OFF(kh)]);
            mov(reg_kw, ptr[this->param1 + GET_OFF(kw)]);
        } else {
            mov(reg_yf, ptr[this->param1 + GET_OFF(yf)]);
            mov(reg_xf, ptr[this->param1 + GET_OFF(xf)]);
            mov(reg_yoff, ptr[this->param1 + GET_OFF(yoff)]);
            mov(reg_xoff, ptr[this->param1 + GET_OFF(xoff)]);
        }

        int nb_c_tail = jpp_.nb_c % jpp_.nb_c_blocking;
        cmp(reg_c_blocks, jpp_.nb_c_blocking);
        jne(nb_c_tail ? tail_label : exit_label, T_NEAR);

        loop_body(jpp_.nb_c_blocking);
        jmp(exit_label, T_NEAR);

        if (nb_c_tail) {
            L(tail_label);
            loop_body(nb_c_tail);
        }

        L(exit_label);

        this->postamble();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    const int vlen = cpu_isa_traits<isa>::vlen;

    Vmm vmm_mask = Vmm(0);
    Vmm vmm_zero = Vmm(0);

    Xmm xmm_yf = Xmm(0);
    Vmm vmm_yf = Vmm(0);
    Xmm xmm_xf = Xmm(1);
    Vmm vmm_xf = Vmm(1);

    Vmm get_acc_reg(int idx) { return Vmm(2*idx + 1); }
    Vmm get_src_reg(int idx) { return Vmm(2*idx + 2); }

    Opmask k_store_mask = Opmask(7);

    const unsigned char _cmp_lt_os = 1;

    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_input     = r8;
    reg64_t aux_reg_input = rax;
    reg64_t aux_reg_input1 = rdx;
    reg64_t reg_output    = r9;
    reg64_t reg_kh    = r10;
    reg64_t reg_kw    = r11;

    reg64_t h_iter = r14;
    reg64_t w_iter = r15;

    reg64_t reg_c_blocks = rbx;
    reg64_t reg_bin_area = rdx;

    reg64_t reg_yf = reg_kh;
    reg64_t reg_xf = reg_kw;

    reg64_t reg_yoff = h_iter;
    reg64_t reg_xoff = r12;

    void roi_pool_max(int c_blocks) {
        Label h_loop_label;
        Label w_loop_label;

        mov(aux_reg_input, reg_input);

        for (int i = 0; i < c_blocks; i++) {
            Vmm vmm_max = get_acc_reg(i);
            uni_vmovups(vmm_max, ptr[reg_input + i * jpp_.ih * jpp_.iw * jpp_.c_block * sizeof(float)]);
        }

        xor_(h_iter, h_iter);
        L(h_loop_label); {
            xor_(w_iter, w_iter);
            mov(aux_reg_input1, aux_reg_input);
            L(w_loop_label); {
                for (int i = 0; i < c_blocks; i++) {
                    Vmm vmm_max = get_acc_reg(i);
                    Vmm vmm_src = get_src_reg(i);

                    uni_vmovups(vmm_src, ptr[aux_reg_input1 + i * jpp_.ih * jpp_.iw * jpp_.c_block * sizeof(float)]);
                    if (isa == cpu::x64::sse41) {
                        movups(vmm_mask, vmm_max);
                        cmpps(vmm_mask, vmm_src, _cmp_lt_os);
                        blendvps(vmm_max, vmm_src);
                    } else if (isa == cpu::x64::avx2) {
                        vcmpps(vmm_mask, vmm_max, vmm_src, _cmp_lt_os);
                        vblendvps(vmm_max, vmm_max, vmm_src, vmm_mask);
                    } else if (isa == cpu::x64::avx512_common) {
                        vcmpps(k_store_mask,  vmm_max,  vmm_src, _cmp_lt_os);
                        vblendmps(vmm_max| k_store_mask, vmm_max, vmm_src);
                    }
                }

                add(aux_reg_input1, jpp_.c_block * sizeof(float));

                inc(w_iter);
                cmp(w_iter, reg_kw);
                jl(w_loop_label, T_NEAR);
            }

            add(aux_reg_input, jpp_.iw * jpp_.c_block * sizeof(float));

            inc(h_iter);
            cmp(h_iter, reg_kh);
            jl(h_loop_label, T_NEAR);
        }

        for (int i = 0; i < c_blocks; i++) {
            Vmm vmm_dst = get_acc_reg(i);
            uni_vmovups(ptr[reg_output + i * jpp_.oh * jpp_.ow * jpp_.c_block * sizeof(float)], vmm_dst);
        }
    }

    void roi_pool_bilinear(int c_blocks) {
        movq(xmm_yf, reg_yf);
        uni_vbroadcastss(vmm_yf, xmm_yf);
        movq(xmm_xf, reg_xf);
        uni_vbroadcastss(vmm_xf, xmm_xf);

        Vmm vmm_src00 = get_src_reg(0);
        Vmm vmm_src01 = get_src_reg(1);
        Vmm vmm_src10 = get_src_reg(2);
        Vmm vmm_src11 = get_src_reg(3);

        for (int i = 0; i < c_blocks; i++) {
            int src_c_off = i * jpp_.ih * jpp_.iw * jpp_.c_block * sizeof(float);

            mov(aux_reg_input, reg_input);
            uni_vmovups(vmm_src00, ptr[aux_reg_input + src_c_off]);
            add(aux_reg_input, reg_xoff);
            uni_vmovups(vmm_src01, ptr[aux_reg_input + src_c_off]);

            add(aux_reg_input, reg_yoff);
            uni_vmovups(vmm_src11, ptr[aux_reg_input + src_c_off]);
            sub(aux_reg_input, reg_xoff);
            uni_vmovups(vmm_src10, ptr[aux_reg_input + src_c_off]);

            uni_vsubps(vmm_src01, vmm_src01, vmm_src00);
            uni_vfmadd213ps(vmm_src01, vmm_xf, vmm_src00);

            uni_vsubps(vmm_src11, vmm_src11, vmm_src10);
            uni_vfmadd213ps(vmm_src11, vmm_xf, vmm_src10);

            uni_vsubps(vmm_src11, vmm_src11, vmm_src01);
            uni_vfmadd213ps(vmm_src11, vmm_yf, vmm_src01);

            int dst_c_off = i * jpp_.oh * jpp_.ow * jpp_.c_block * sizeof(float);
            uni_vmovups(ptr[reg_output + dst_c_off], vmm_src11);
        }
    }

    void empty_roi(int c_blocks) {
        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
        for (int i = 0; i < c_blocks; i++) {
            uni_vmovups(ptr[reg_output + i * jpp_.oh * jpp_.ow * jpp_.c_block * sizeof(float)], vmm_zero);
        }
    }

    void loop_body(int c_blocks) {
        Label empty_roi_label;
        Label exit_label;

        cmp(reg_bin_area, 0);
        je(empty_roi_label, T_NEAR);

        if (jpp_.alg == ROIPoolingOpType::Max)
            roi_pool_max(c_blocks);
        else
            roi_pool_bilinear(c_blocks);

        if (isa == cpu::x64::sse41) {
            add(reg_input, 4 * sizeof(float));
            add(reg_output, 4 * sizeof(float));

            if (jpp_.alg == ROIPoolingOpType::Max)
                roi_pool_max(c_blocks);
            else
                roi_pool_bilinear(c_blocks);
        }
        jmp(exit_label, T_NEAR);

        L(empty_roi_label);
        empty_roi(c_blocks);
        if (isa == cpu::x64::sse41) {
            add(reg_output, 4 * sizeof(float));
            empty_roi(c_blocks);
        }

        L(exit_label);
    }
};

MKLDNNROIPoolingNode::MKLDNNROIPoolingNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNROIPoolingNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    GenericLayer* genericLayer = getCnnLayer().get();
    if (genericLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert ROIPooling layer.";

    std::string errorPrefix = "ROIPooling layer with name '" + getName() + "' ";

    if (getParentEdges().size() != 2)
        THROW_IE_EXCEPTION << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << errorPrefix << "has incorrect number of output edges: " << getChildEdges().size();

    if (getParentEdgeAt(0)->getDims().ndims() != 4) {
        THROW_IE_EXCEPTION << errorPrefix << "doesn't support 0th input with rank: " << getParentEdgeAt(0)->getDims().ndims();
    }

    if (getParentEdgeAt(1)->getDims().ndims() != 2) {
        THROW_IE_EXCEPTION << errorPrefix << "doesn't support 1st input with rank: " << getParentEdgeAt(1)->getDims().ndims();
    }

    if (getChildEdgeAt(0)->getDims().ndims() != 4) {
        THROW_IE_EXCEPTION << errorPrefix << "doesn't support output with rank: " << getChildEdgeAt(0)->getDims().ndims();
    }

    if (getParentEdgeAt(1)->getDims()[1] != 5) {
        THROW_IE_EXCEPTION << errorPrefix << "has invalid shape on 1st input: ["
                                          << getParentEdgeAt(1)->getDims()[0] << "," << getParentEdgeAt(1)->getDims()[1] << "]";
    }

    pooled_h = genericLayer->GetParamAsInt("pooled_h");
    pooled_w = genericLayer->GetParamAsInt("pooled_w");
    spatial_scale = genericLayer->GetParamAsFloat("spatial_scale");
    std::string m = genericLayer->GetParamAsString("method", "max");
    if (m == "max") {
        opType = ROIPoolingOpType::Max;
    } else if (m == "bilinear") {
        opType = ROIPoolingOpType::Bilinear;
    } else {
        THROW_IE_EXCEPTION << errorPrefix << "doesn't support roi pooling method: " << m;
    }
}

void MKLDNNROIPoolingNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(2);
    config.inConfs[0].constant = false;
    config.inConfs[0].inPlace = -1;
    config.inConfs[1].constant = false;
    config.inConfs[1].inPlace = -1;

    config.outConfs.resize(1);
    config.outConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;

    auto parentDims = getParentEdgeAt(0)->getDims();
    auto format = mayiuse(avx512_common) ? memory::format_tag::nChw16c : memory::format_tag::nChw8c;
    impl_desc_type impl_type;
    if (mayiuse(cpu::x64::avx512_common)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), memory::data_type::f32, format);
    config.inConfs[1].desc = MKLDNNMemoryDesc(getParentEdgeAt(1)->getDims(), memory::data_type::f32, memory::format_tag::nc);
    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), memory::data_type::f32, format);
    supportedPrimitiveDescriptors.push_back({config, impl_type, format});
}

void MKLDNNROIPoolingNode::createPrimitive() {
    auto selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    if (!selectedPrimitiveDescriptor)
        THROW_IE_EXCEPTION << "CPU ROI Pooling node with name '" << getName() << "' doesn't have primitive descriptors.";
    auto config = selectedPrimitiveDescriptor->getConfig();

    const int simd_w = mayiuse(cpu::x64::avx512_common) ? 16 : 8;
    jpp.c_block = simd_w;

    auto inDims = config.inConfs[0].desc.getDims();
    auto outDims = config.outConfs[0].desc.getDims();

    jpp.mb = outDims[0];
    jpp.c = rnd_up(inDims[1], simd_w);
    jpp.ih = inDims[2];
    jpp.iw = inDims[3];
    jpp.oh = outDims[2];
    jpp.ow = outDims[3];

    jpp.spatial_scale = spatial_scale;
    jpp.pooled_h = pooled_h;
    jpp.pooled_w = pooled_w;

    jpp.nb_c = jpp.c / jpp.c_block;

    jpp.nb_c_blocking = mayiuse(cpu::x64::avx512_common) ? 15 : 7;

    jpp.alg = opType;

    if (mayiuse(cpu::x64::avx512_common)) {
        roi_pooling_kernel.reset(new jit_uni_roi_pooling_kernel_f32<cpu::x64::avx512_common>(jpp));
    } else if (mayiuse(cpu::x64::avx2)) {
        roi_pooling_kernel.reset(new jit_uni_roi_pooling_kernel_f32<cpu::x64::avx2>(jpp));
    } else if (mayiuse(cpu::x64::sse41)) {
        roi_pooling_kernel.reset(new jit_uni_roi_pooling_kernel_f32<cpu::x64::sse41>(jpp));
    }

    if (roi_pooling_kernel)
        roi_pooling_kernel->create_ker();
}

void MKLDNNROIPoolingNode::execute(mkldnn::stream strm) {
    auto &srcMemory0 = getParentEdgeAt(0)->getMemory();
    auto &srcMemory1 = getParentEdgeAt(1)->getMemory();
    auto &dstMemory = getChildEdgeAt(0)->getMemory();

    const auto *src_data = reinterpret_cast<const float *>(srcMemory0.GetPtr());
    const auto *src_roi = reinterpret_cast<const float *>(srcMemory1.GetPtr());
    float *dst = reinterpret_cast<float *>(dstMemory.GetPtr());

    auto selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    if (!selectedPrimitiveDescriptor)
        THROW_IE_EXCEPTION << "CPU ROI Pooling node with name '" << getName() << "' doesn't have primitive descriptors.";
    auto config = selectedPrimitiveDescriptor->getConfig();

    auto src_strides = config.inConfs[0].desc.getBlockingDesc().getStrides();
    auto dst_strides = config.outConfs[0].desc.getBlockingDesc().getStrides();

    int cb_work = impl::utils::div_up(jpp.nb_c, jpp.nb_c_blocking);
    int MB = jpp.mb;

    size_t src_roi_step = config.inConfs[1].desc.getBlockingDesc().getStrides()[0];
    int real_rois = 0;
    for (; real_rois < MB; real_rois++) {
        size_t roi_off = real_rois * src_roi_step;

        const float *src_roi_ptr = &src_roi[roi_off];
        int roi_batch_ind = static_cast<int>(src_roi_ptr[0]);
        if (roi_batch_ind == -1) {
            break;
        }
    }

    parallel_for4d(MB, cb_work, jpp.oh, jpp.ow, [&](int n, int cbb, int oh, int ow) {
        auto arg = jit_roi_pooling_call_args();

        int cb = cbb * jpp.nb_c_blocking;
        int cb_num = jpp.nb_c_blocking;
        int c_block = jpp.c_block;

        arg.c_blocks = std::min(cb + cb_num, jpp.nb_c) - cb;

        if (n >= real_rois) {
            if (roi_pooling_kernel) {
                arg.bin_area = 0;
                arg.dst = &dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3]];
            } else {
                for (int c = 0; c < c_block; c++) {
                    dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3] + c] = 0;
                }
            }

            (*roi_pooling_kernel)(&arg);
        } else {
            size_t roi_off = n * src_roi_step;
            const float* src_roi_ptr = &src_roi[roi_off];

            int roi_batch_ind = static_cast<int>(src_roi_ptr[0]);

            if (jpp.alg == ROIPoolingOpType::Max) {
                int roi_start_w = static_cast<int>(round(src_roi_ptr[1] * jpp.spatial_scale));
                int roi_start_h = static_cast<int>(round(src_roi_ptr[2] * jpp.spatial_scale));
                int roi_end_w = static_cast<int>(round(src_roi_ptr[3] * jpp.spatial_scale));
                int roi_end_h = static_cast<int>(round(src_roi_ptr[4] * jpp.spatial_scale));

                int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
                int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);


                int hstart = (oh * roi_height) / jpp.pooled_h;
                if ((hstart * jpp.pooled_h) > (oh * roi_height)) {
                    --hstart;
                }

                int wstart = (ow * roi_width) / jpp.pooled_w;
                if ((wstart * jpp.pooled_w) > (ow * roi_width)) {
                    --wstart;
                }

                int hend = ((oh + 1) * roi_height) / jpp.pooled_h;
                if ((hend * jpp.pooled_h) < ((oh + 1) * roi_height)) {
                    ++hend;
                }

                int wend = ((ow + 1) * roi_width) / jpp.pooled_w;
                if ((wend * jpp.pooled_w) < ((ow + 1) * roi_width)) {
                    ++wend;
                }

                hstart = std::min(std::max(hstart + roi_start_h, 0), jpp.ih);
                hend = std::min(std::max(hend + roi_start_h, 0), jpp.ih);
                wstart = std::min(std::max(wstart + roi_start_w, 0), jpp.iw);
                wend = std::min(std::max(wend + roi_start_w, 0), jpp.iw);

                if (roi_pooling_kernel) {
                    arg.src = &src_data[roi_batch_ind * src_strides[0] + cb * src_strides[1] + hstart * src_strides[2] + wstart * src_strides[3]];
                    arg.dst = &dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3]];

                    arg.bin_area = (hend - hstart) * (wend - wstart);
                    arg.kh = hend - hstart;
                    arg.kw = wend - wstart;
                } else {
                    for (int c = 0; c < c_block; c++) {
                        const size_t pool_index = n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3] + c;
                        if ((hend <= hstart) || (wend <= wstart)) {
                            dst[pool_index] = 0;
                        } else {
                            for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {
                                    float batch_data = src_data[roi_batch_ind * src_strides[0] + cb * src_strides[1] +
                                                                h * src_strides[2] + w * src_strides[3] + c];

                                    if (batch_data > dst[pool_index]) {
                                        dst[pool_index] = batch_data;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                float roi_start_w_ = src_roi_ptr[1];
                float roi_start_h_ = src_roi_ptr[2];
                float roi_end_w_   = src_roi_ptr[3];
                float roi_end_h_   = src_roi_ptr[4];

                float height_scale = (jpp.pooled_h > 1 ? ((roi_end_h_ - roi_start_h_) * (jpp.ih - 1)) / (jpp.pooled_h - 1) : 0);
                float width_scale  = (jpp.pooled_w > 1 ? ((roi_end_w_ - roi_start_w_) * (jpp.iw - 1)) / (jpp.pooled_w - 1) : 0);

                float in_y = (jpp.pooled_h > 1 ? (oh * height_scale + roi_start_h_ * (jpp.ih - 1)) :
                              0.5 * (roi_start_h_ + roi_end_h_) * (jpp.ih - 1));
                float in_x = (jpp.pooled_w > 1 ? (ow * width_scale  + roi_start_w_ * (jpp.iw - 1)) :
                              0.5 * (roi_start_w_ + roi_end_w_) * (jpp.iw - 1));

                if (in_y < 0 || in_y > jpp.ih - 1 || in_x < 0 || in_x > jpp.iw - 1) {
                    if (roi_pooling_kernel) {
                        arg.bin_area = 0;
                        arg.dst = &dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3]];
                    } else {
                        for (int c = 0; c < c_block; c++) {
                            dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3] + c] = 0;
                        }
                    }
                } else {
                    int top_y_index    = static_cast<int>(floorf(in_y));
                    int bottom_y_index = static_cast<int>(ceilf(in_y));
                    int left_x_index   = static_cast<int>(floorf(in_x));
                    int right_x_index  = static_cast<int>(ceilf(in_x));

                    if (right_x_index > jpp.iw - 1)
                        right_x_index = jpp.iw - 1;

                    if (bottom_y_index > jpp.ih - 1)
                        bottom_y_index = jpp.ih - 1;

                    if (roi_pooling_kernel) {
                        arg.dst = &dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3]];

                        arg.xf = in_x - left_x_index;
                        arg.yf = in_y - top_y_index;

                        arg.xoff = sizeof(float) * (right_x_index - left_x_index) * jpp.c_block;
                        arg.yoff = sizeof(float) * (bottom_y_index - top_y_index) * jpp.iw * jpp.c_block;

                        arg.src = &src_data[roi_batch_ind * src_strides[0] + cb * src_strides[1] +
                                            top_y_index * src_strides[2] + left_x_index * src_strides[3]];
                        arg.bin_area = 1;
                    } else {
                        for (int c = 0; c < 1; c++) {
                            const float top_left     = src_data[roi_batch_ind * src_strides[0] + cb * src_strides[1] +
                                                                top_y_index * src_strides[2] + left_x_index * src_strides[3] + c];
                            const float top_right    = src_data[roi_batch_ind * src_strides[0] + cb * src_strides[1] +
                                                                top_y_index * src_strides[2] + right_x_index * src_strides[3] + c];
                            const float bottom_left  = src_data[roi_batch_ind * src_strides[0] + cb * src_strides[1] +
                                                                bottom_y_index * src_strides[2] + left_x_index * src_strides[3] + c];
                            const float bottom_right = src_data[roi_batch_ind * src_strides[0] + cb * src_strides[1] +
                                                                bottom_y_index * src_strides[2] + right_x_index * src_strides[3] + c];

                            const float top    = top_left + (top_right - top_left) * (in_x - left_x_index);
                            const float bottom = bottom_left + (bottom_right - bottom_left) * (in_x - left_x_index);

                            dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3] + c] =
                                    top + (bottom - top) * (in_y - top_y_index);
                        }
                    }
                }
            }

            if (roi_pooling_kernel) {
                (*roi_pooling_kernel)(&arg);
            }
        }
    });
}


bool MKLDNNROIPoolingNode::created() const {
    return getType() == ROIPooling;
}

REG_MKLDNN_PRIM_FOR(MKLDNNROIPoolingNode, ROIPooling);
