// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_quantize_node.h"

#include <legacy/ie_layers.h>
#include <string>
#include <vector>
#include <math.h>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "utils/general_utils.h"

#include <algorithm>
#include <set>
#include <cmath>
#include <cpu/x64/jit_generator.hpp>
#include "ie_parallel.hpp"

#include <ngraph/opsets/opset1.hpp>

// Quantization ranges validation is switched off by default in order to avoid regressions on user side
// #define VALIDATE_QUANTIZATION_RANGES

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace details;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu::x64;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_quantize_call_args, field)

template <cpu_isa_t isa>
struct jit_uni_binarization_kernel : public jit_uni_quantize_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_binarization_kernel)

    explicit jit_uni_binarization_kernel(jit_quantize_params jqp) : jit_uni_quantize_kernel(jqp), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    };

    void generate() override {
        this->preamble();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_thresholds, ptr[param + GET_OFF(thresholds)]);
        mov(reg_output_mask, ptr[param + GET_OFF(output_mask)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        const int nbits = 8;
        int simd_w = isa == avx512_common ? 16 : 8;
        const int C = jqp_.c;
        const int tail_size = C % simd_w;

        Label unrolled_loop_label;
        Label main_loop_label;
        Label tail_label;
        Label exit_label;

        L(unrolled_loop_label); {
            int step = isa == cpu::x64::sse41 ? nbits / 2 : isa == cpu::x64::avx2 ? nbits : 2 * nbits;
            const int ur_ch = isa == cpu::x64::sse41 ? nbits : isa == cpu::x64::avx2 ? nbits / 2 : nbits / 4;
            const int unrolled_loop_step = ur_ch * step;

            cmp(reg_work_amount, unrolled_loop_step);
            jl(main_loop_label, T_NEAR);

            xor_(reg_bin_32, reg_bin_32);
            for (int ch = 0; ch < ur_ch; ch++) {
                uni_vmovups(vmm_src(0), ptr[reg_from + ch*step*sizeof(float)]);
                uni_vmovups(vmm_wei(0), ptr[reg_thresholds + ch*step*sizeof(float)]);
                uni_vmovups(vmm_mask(0), ptr[reg_output_mask + ch*step*sizeof(float)]);
                if (isa == avx512_common) {
                    vcmpps(k_mask0, vmm_src(0), vmm_wei(0), _cmp_gt_os);
                    vptestmd(k_mask1, vmm_mask(0), vmm_mask(0));
                    kxnorw(k_mask0, k_mask0, k_mask1);
                    kmovw(reg_src_32, k_mask0);
                } else {
                    uni_vcmpgtps(vmm_src(0), vmm_src(0), vmm_wei(0));
                    uni_vpcmpeqd(vmm_src(0), vmm_src(0), vmm_mask(0));
                    uni_vmovmskps(reg_src_32, vmm_src(0));
                }
                shl(reg_src_32, ch * step);
                or_(reg_bin_32, reg_src_32);
            }
            mov(ptr[reg_to], reg_bin_32);

            add(reg_from, unrolled_loop_step*sizeof(float));
            add(reg_thresholds, unrolled_loop_step*sizeof(float));
            add(reg_output_mask, unrolled_loop_step*sizeof(float));
            add(reg_to, sizeof(uint32_t));
            sub(reg_work_amount, unrolled_loop_step);

            jmp(unrolled_loop_label, T_NEAR);
        }

        L(main_loop_label); {
            int repeats = isa == cpu::x64::sse41 ? 2 : 1;
            int step = isa == cpu::x64::sse41 ? nbits / 2 : isa == cpu::x64::avx2 ? nbits : nbits * 2;
            const int main_loop_step = step * repeats;

            cmp(reg_work_amount, main_loop_step);
            jl(tail_label, T_NEAR);

            xor_(reg_bin_32, reg_bin_32);
            for (int i = 0; i < repeats; i++) {
                uni_vmovups(vmm_src(0), ptr[reg_from + i*step*sizeof(float)]);
                uni_vmovups(vmm_wei(0), ptr[reg_thresholds + i*step*sizeof(float)]);
                uni_vmovups(vmm_mask(0), ptr[reg_output_mask + i*step*sizeof(float)]);
                if (isa == avx512_common) {
                    vcmpps(k_mask0, vmm_src(0), vmm_wei(0), _cmp_gt_os);
                    vptestmd(k_mask1, vmm_mask(0), vmm_mask(0));
                    kxnorw(k_mask0, k_mask0, k_mask1);
                    kmovw(reg_src_32, k_mask0);
                } else {
                    uni_vcmpgtps(vmm_src(0), vmm_src(0), vmm_wei(0));
                    uni_vpcmpeqd(vmm_src(0), vmm_src(0), vmm_mask(0));
                    uni_vmovmskps(reg_src_32, vmm_src(0));
                }
                shl(reg_src_32, i * step);
                or_(reg_bin_32, reg_src_32);
            }
            if (isa == avx512_common)
                mov(ptr[reg_to], reg_bin_16);
            else
                mov(ptr[reg_to], reg_bin_8);

            add(reg_from, main_loop_step*sizeof(float));
            add(reg_thresholds, main_loop_step*sizeof(float));
            add(reg_output_mask, main_loop_step*sizeof(float));
            add(reg_to, isa == avx512_common ? sizeof(uint16_t) : sizeof(uint8_t));
            sub(reg_work_amount, main_loop_step);

            jmp(main_loop_label, T_NEAR);
        }

        L(tail_label); {
            if (tail_size != 0) {
                xor_(reg_bin_32, reg_bin_32);
                mov(reg_mask, 1);
                for (int c = 0; c < tail_size; c++) {
                    uni_vpxor(xmm_src(0), xmm_src(0), xmm_src(0));
                    uni_vpxor(xmm_wei(0), xmm_wei(0), xmm_wei(0));
                    uni_vpxor(xmm_mask(0), xmm_mask(0), xmm_mask(0));

                    movss(xmm_src(0), ptr[reg_from + c * sizeof(float)]);
                    movss(xmm_wei(0), ptr[reg_thresholds + c * sizeof(float)]);
                    movss(xmm_mask(0), ptr[reg_output_mask + c * sizeof(float)]);
                    uni_vcmpgtps(xmm_src(0), xmm_src(0), xmm_wei(0));
                    uni_vpcmpeqd(xmm_src(0), xmm_src(0), xmm_mask(0));
                    uni_vmovmskps(reg_src_32, xmm_src(0));

                    shl(reg_src_32, c);
                    and_(reg_src_32, reg_mask);
                    or_(reg_bin_32, reg_src_32);
                    shl(reg_mask, 1);
                }
                if (isa == avx512_common && tail_size > nbits)
                    mov(ptr[reg_to], reg_bin_16);
                else
                    mov(ptr[reg_to], reg_bin_8);
            }
        }

        L(exit_label);

        this->postamble();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    inline Vmm vmm_src(int idx) { return Vmm(idx); }
    inline Xmm xmm_src(int idx) { return Xmm(idx); }
    inline Vmm vmm_wei(int idx) { return Vmm(idx + 4); }
    inline Vmm vmm_mask(int idx) { return Vmm(idx + 5); }
    inline Xmm xmm_wei(int idx) { return Xmm(idx + 4); }
    inline Xmm xmm_mask(int idx) { return Xmm(idx + 5); }

    Reg64 param = abi_param1;
    Reg64 reg_from = r8;
    Reg64 reg_to = r9;
    Reg64 reg_work_amount = r10;
    Reg64 reg_thresholds = r11;
    Reg64 reg_output_mask = r14;
    Reg16 reg_bin_16 = r12w;
    Reg32 reg_bin_32 = r12d;
    Reg8 reg_bin_8 = r12b;
    Reg32 reg_src_32 = r13d;
    Reg32 reg_mask = r15d;

    const unsigned char _cmp_gt_os = 6;
    Xbyak::Opmask k_mask0 = Xbyak::Opmask(1);
    Xbyak::Opmask k_mask1 = Xbyak::Opmask(2);
};

template <cpu_isa_t isa>
struct jit_uni_quantization_kernel : public jit_uni_quantize_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_quantization_kernel)

    explicit jit_uni_quantization_kernel(jit_quantize_params jqp) : jit_uni_quantize_kernel(jqp), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    };

    void generate() override {
        do_dequantization = jqp_.op_type == QuantizeOpType::FakeQuantization;
        do_rounding = do_dequantization || jqp_.dst_prc == Precision::FP32;

        this->preamble();

        if (jqp_.src_layout == Layout::CHW || jqp_.src_layout == Layout::NCHW || jqp_.src_layout == Layout::NCDHW)
            compute_planar();
        else
            compute_generic();


        this->postamble();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    inline Vmm vmm_val(int idx) { return Vmm(idx + 0); }
    inline Vmm vmm_crop_low(int idx) { return Vmm(idx + 2); }
    inline Vmm vmm_crop_high(int idx) { return Vmm(idx + 4); }
    inline Vmm vmm_input_scale(int idx) { return Vmm(idx + 6); }
    inline Vmm vmm_input_shift(int idx) { return Vmm(idx + 8); }
    inline Vmm vmm_output_scale(int idx) { return Vmm(idx + 10); }
    inline Vmm vmm_output_shift(int idx) { return Vmm(idx + 12); }

    inline Ymm ymm_val(int idx) { return Ymm(idx + 0); }
    inline Ymm ymm_crop_low(int idx) { return Ymm(idx + 2); }
    inline Ymm ymm_crop_high(int idx) { return Ymm(idx + 4); }
    inline Ymm ymm_input_scale(int idx) { return Ymm(idx + 6); }
    inline Ymm ymm_input_shift(int idx) { return Ymm(idx + 8); }
    inline Ymm ymm_output_scale(int idx) { return Ymm(idx + 10); }
    inline Ymm ymm_output_shift(int idx) { return Ymm(idx + 12); }

    inline Xmm xmm_val(int idx) { return Xmm(idx + 0); }
    inline Xmm xmm_crop_low(int idx) { return Xmm(idx + 2); }
    inline Xmm xmm_crop_high(int idx) { return Xmm(idx + 4); }
    inline Xmm xmm_input_scale(int idx) { return Xmm(idx + 6); }
    inline Xmm xmm_input_shift(int idx) { return Xmm(idx + 8); }
    inline Xmm xmm_output_scale(int idx) { return Xmm(idx + 10); }
    inline Xmm xmm_output_shift(int idx) { return Xmm(idx + 12); }

    Vmm vmm_zero = Vmm(14);

    Reg64 param = abi_param1;
    Reg64 reg_from = rbp;
    Reg64 reg_to = r9;
    Reg64 aux_reg_from = abi_not_param1;
    Reg64 aux_reg_to = r8;
    Reg64 reg_src_step = r10;
    Reg64 reg_dst_step = rsi;
    Reg64 reg_block_size = r11;
    Reg64 reg_work_amount = r12;

    Reg8 reg_tmp_8 = r9b;
    Reg32 reg_tmp_32 = r9d;
    Reg64 reg_tmp_64 = r9;

    Reg64 reg_crop_low = r13;
    Reg64 reg_crop_high = r14;
    Reg64 reg_input_scale = r15;
    Reg64 reg_input_shift = rax;
    Reg64 reg_output_scale = rbx;
    Reg64 reg_output_shift = rdx;

    bool do_rounding = true;
    bool do_dequantization = true;

    inline void compute_planar() {
        int src_type_size = jqp_.src_prc.size();
        int dst_type_size = jqp_.dst_prc.size();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);

        mov(reg_crop_low, ptr[param + GET_OFF(crop_low)]);
        mov(reg_crop_high, ptr[param + GET_OFF(crop_high)]);
        mov(reg_input_scale, ptr[param + GET_OFF(input_scale)]);
        mov(reg_input_shift, ptr[param + GET_OFF(input_shift)]);
        mov(reg_output_scale, ptr[param + GET_OFF(output_scale)]);
        mov(reg_output_shift, ptr[param + GET_OFF(output_shift)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        if (isa == cpu::x64::avx512_common)
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        int simd_w = isa == cpu::x64::avx512_common ? 16 : 8;
        int tail_simd_w = 4;
        int repeats = isa == cpu::x64::sse41 ? 2 : 1;

        Label main_loop_label;
        Label tail_blk4_label;
        Label tail_blk4_loop_label;
        Label tail_blk4_exit_label;
        Label tail_label;
        Label tail_loop_label;
        Label exit_label;

        uni_vbroadcastss(vmm_crop_low(0), ptr[reg_crop_low]);
        uni_vbroadcastss(vmm_crop_high(0), ptr[reg_crop_high]);
        uni_vbroadcastss(vmm_input_scale(0), ptr[reg_input_scale]);
        uni_vbroadcastss(vmm_input_shift(0), ptr[reg_input_shift]);
        if (do_dequantization) {
            uni_vbroadcastss(vmm_output_scale(0), ptr[reg_output_scale]);
            uni_vbroadcastss(vmm_output_shift(0), ptr[reg_output_shift]);
        }

        L(main_loop_label); {
            cmp(reg_work_amount, simd_w);
            jl(tail_blk4_label, T_NEAR);

            for (int i = 0; i < repeats; i++) {
                load_vector(vmm_val(i), ptr[reg_from + i * (simd_w / 2) * src_type_size], jqp_.src_prc);

                uni_vminps(vmm_val(i), vmm_val(i), vmm_crop_high(0));
                uni_vmaxps(vmm_val(i), vmm_val(i), vmm_crop_low(0));
                uni_vfmadd213ps(vmm_val(i), vmm_input_scale(0), vmm_input_shift(0));
                if (do_rounding) uni_vroundps(vmm_val(i), vmm_val(i), 0);
                if (do_dequantization) uni_vfmadd213ps(vmm_val(i), vmm_output_scale(0), vmm_output_shift(0));

                store_vector(ptr[reg_to + i * (simd_w / 2) * dst_type_size], vmm_val(i), jqp_.dst_prc);
            }

            sub(reg_work_amount, simd_w);
            add(reg_from, simd_w * src_type_size);
            add(reg_to, simd_w * dst_type_size);

            jmp(main_loop_label, T_NEAR);
        }

        L(tail_blk4_label); {
            cmp(reg_work_amount, tail_simd_w);
            jl(tail_blk4_exit_label, T_NEAR);

            load_vector(xmm_val(0), ptr[reg_from], jqp_.src_prc);

            uni_vminps(xmm_val(0), xmm_val(0), xmm_crop_high(0));
            uni_vmaxps(xmm_val(0), xmm_val(0), xmm_crop_low(0));
            uni_vfmadd213ps(xmm_val(0), xmm_input_scale(0), xmm_input_shift(0));
            if (do_rounding) uni_vroundps(xmm_val(0), xmm_val(0), 0);
            if (do_dequantization) uni_vfmadd213ps(xmm_val(0), xmm_output_scale(0), xmm_output_shift(0));

            store_vector(ptr[reg_to], xmm_val(0), jqp_.dst_prc);

            sub(reg_work_amount, tail_simd_w);
            add(reg_from, tail_simd_w * src_type_size);
            add(reg_to, tail_simd_w * dst_type_size);
        }

        L(tail_blk4_exit_label);

        mov(aux_reg_from, reg_from);
        mov(aux_reg_to, reg_to);

        L(tail_loop_label); {
            cmp(reg_work_amount, 0);
            jle(exit_label, T_NEAR);

            load_scalar(xmm_val(0), ptr[aux_reg_from], jqp_.src_prc);

            uni_vminps(xmm_val(0), xmm_val(0), xmm_crop_high(0));
            uni_vmaxps(xmm_val(0), xmm_val(0), xmm_crop_low(0));
            uni_vfmadd213ps(xmm_val(0), xmm_input_scale(0), xmm_input_shift(0));
            if (do_rounding) uni_vroundps(xmm_val(0), xmm_val(0), 0);
            if (do_dequantization) uni_vfmadd213ps(xmm_val(0), xmm_output_scale(0), xmm_output_shift(0));

            store_scalar(ptr[aux_reg_to], xmm_val(0), jqp_.dst_prc);

            sub(reg_work_amount, 1);
            add(aux_reg_from, 1 * src_type_size);
            add(aux_reg_to, 1 * dst_type_size);

            jmp(tail_loop_label, T_NEAR);
        }

        L(exit_label);
    }

    inline void compute_generic() {
        int src_type_size = jqp_.src_prc.size();
        int wei_type_size = jqp_.wei_prc.size();
        int dst_type_size = jqp_.dst_prc.size();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);

        mov(reg_crop_low, ptr[param + GET_OFF(crop_low)]);
        mov(reg_crop_high, ptr[param + GET_OFF(crop_high)]);
        mov(reg_input_scale, ptr[param + GET_OFF(input_scale)]);
        mov(reg_input_shift, ptr[param + GET_OFF(input_shift)]);
        if (do_dequantization) {
            mov(reg_output_scale, ptr[param + GET_OFF(output_scale)]);
            mov(reg_output_shift, ptr[param + GET_OFF(output_shift)]);
        }

        mov(reg_src_step, ptr[param + GET_OFF(src_step)]);
        mov(reg_dst_step, ptr[param + GET_OFF(dst_step)]);
        mov(reg_block_size, ptr[param + GET_OFF(block_size)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        if (isa == cpu::x64::avx512_common)
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        int simd_w = isa == cpu::x64::avx512_common ? 16 : 8;
        int tail8_simd_w = 8;
        int tail4_simd_w = 4;
        int repeats = isa == cpu::x64::sse41 ? 2 : 1;

        Label main_loop_label;
        Label tail_blk8_label;
        Label tail_blk8_loop_label;
        Label tail_blk8_exit_label;
        Label tail_blk4_label;
        Label tail_blk4_loop_label;
        Label tail_blk4_exit_label;
        Label tail_label;
        Label tail_loop_label;
        Label exit_label;

        cmp(reg_block_size, simd_w);
        jl(simd_w == 16 ? tail_blk8_label : tail_blk4_label, T_NEAR);

        for (int i = 0; i < repeats; i++) {
            uni_vmovups(vmm_crop_low(i), ptr[reg_crop_low + i * (simd_w / 2) * sizeof(float)]);
            uni_vmovups(vmm_crop_high(i), ptr[reg_crop_high + i * (simd_w / 2) * sizeof(float)]);
            uni_vmovups(vmm_input_scale(i), ptr[reg_input_scale + i * (simd_w / 2) * sizeof(float)]);
            uni_vmovups(vmm_input_shift(i), ptr[reg_input_shift + i * (simd_w / 2) * sizeof(float)]);
            if (do_dequantization) {
                uni_vmovups(vmm_output_scale(i), ptr[reg_output_scale + i * (simd_w / 2) * sizeof(float)]);
                uni_vmovups(vmm_output_shift(i), ptr[reg_output_shift + i * (simd_w / 2) * sizeof(float)]);
            }
        }

        L(main_loop_label); {
            cmp(reg_work_amount, 0);
            jle(exit_label, T_NEAR);

            for (int i = 0; i < repeats; i++) {
                load_vector(vmm_val(i), ptr[reg_from + i * (simd_w / 2) * src_type_size], jqp_.src_prc);

                uni_vminps(vmm_val(i), vmm_val(i), vmm_crop_high(i));
                uni_vmaxps(vmm_val(i), vmm_val(i), vmm_crop_low(i));
                uni_vfmadd213ps(vmm_val(i), vmm_input_scale(i), vmm_input_shift(i));
                if (do_rounding) uni_vroundps(vmm_val(i), vmm_val(i), 0);
                if (do_dequantization) uni_vfmadd213ps(vmm_val(i), vmm_output_scale(i), vmm_output_shift(i));

                store_vector(ptr[reg_to + i * (simd_w / 2) * dst_type_size], vmm_val(i), jqp_.dst_prc);
            }

            dec(reg_work_amount);
            add(reg_from, reg_src_step);
            add(reg_to, reg_dst_step);

            jmp(main_loop_label, T_NEAR);
        }

        if (simd_w == 16) {
            L(tail_blk8_label);

            cmp(reg_block_size, tail8_simd_w);
            jl(tail_blk4_label, T_NEAR);

            mov(aux_reg_to, reg_to);
            mov(aux_reg_from, reg_from);
            mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

            uni_vmovups(ymm_crop_low(0), ptr[reg_crop_low]);
            uni_vmovups(ymm_crop_high(0), ptr[reg_crop_high]);
            uni_vmovups(ymm_input_scale(0), ptr[reg_input_scale]);
            uni_vmovups(ymm_input_shift(0), ptr[reg_input_shift]);
            if (do_dequantization) {
                uni_vmovups(ymm_output_scale(0), ptr[reg_output_scale]);
                uni_vmovups(ymm_output_shift(0), ptr[reg_output_shift]);
            }

            L(tail_blk8_loop_label); {
                cmp(reg_work_amount, 0);
                jle(tail_blk8_exit_label, T_NEAR);

                load_vector(ymm_val(0), ptr[aux_reg_from], jqp_.src_prc);

                uni_vminps(ymm_val(0), ymm_val(0), ymm_crop_high(0));
                uni_vmaxps(ymm_val(0), ymm_val(0), ymm_crop_low(0));
                uni_vfmadd213ps(ymm_val(0), ymm_input_scale(0), ymm_input_shift(0));
                if (do_rounding) uni_vroundps(ymm_val(0), ymm_val(0), 0);
                if (do_dequantization) uni_vfmadd213ps(ymm_val(0), ymm_output_scale(0), ymm_output_shift(0));

                store_vector(ptr[aux_reg_to], ymm_val(0), jqp_.dst_prc);

                dec(reg_work_amount);
                add(aux_reg_from, reg_src_step);
                add(aux_reg_to, reg_dst_step);

                jmp(tail_blk8_loop_label, T_NEAR);
            }

            L(tail_blk8_exit_label);

            add(reg_from, tail8_simd_w * src_type_size);
            add(reg_to, tail8_simd_w * dst_type_size);
            add(reg_crop_low, tail8_simd_w * wei_type_size);
            add(reg_crop_high, tail8_simd_w * wei_type_size);
            add(reg_input_scale, tail8_simd_w * wei_type_size);
            add(reg_input_shift, tail8_simd_w * wei_type_size);
            if (do_dequantization) {
                add(reg_output_scale, tail8_simd_w * wei_type_size);
                add(reg_output_shift, tail8_simd_w * wei_type_size);
            }
            sub(reg_block_size, tail8_simd_w);
        }

        L(tail_blk4_label);

        cmp(reg_block_size, tail4_simd_w);
        jl(tail_label, T_NEAR);

        mov(aux_reg_to, reg_to);
        mov(aux_reg_from, reg_from);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        uni_vmovups(xmm_crop_low(0), ptr[reg_crop_low]);
        uni_vmovups(xmm_crop_high(0), ptr[reg_crop_high]);
        uni_vmovups(xmm_input_scale(0), ptr[reg_input_scale]);
        uni_vmovups(xmm_input_shift(0), ptr[reg_input_shift]);
        if (do_dequantization) {
            uni_vmovups(xmm_output_scale(0), ptr[reg_output_scale]);
            uni_vmovups(xmm_output_shift(0), ptr[reg_output_shift]);
        }

        L(tail_blk4_loop_label); {
            cmp(reg_work_amount, 0);
            jle(tail_blk4_exit_label, T_NEAR);

            load_vector(xmm_val(0), ptr[aux_reg_from], jqp_.src_prc);

            uni_vminps(xmm_val(0), xmm_val(0), xmm_crop_high(0));
            uni_vmaxps(xmm_val(0), xmm_val(0), xmm_crop_low(0));
            uni_vfmadd213ps(xmm_val(0), xmm_input_scale(0), xmm_input_shift(0));
            if (do_rounding) uni_vroundps(xmm_val(0), xmm_val(0), 0);
            if (do_dequantization) uni_vfmadd213ps(xmm_val(0), xmm_output_scale(0), xmm_output_shift(0));

            store_vector(ptr[aux_reg_to], xmm_val(0), jqp_.dst_prc);

            dec(reg_work_amount);
            add(aux_reg_from, reg_src_step);
            add(aux_reg_to, reg_dst_step);

            jmp(tail_blk4_loop_label, T_NEAR);
        }

        L(tail_blk4_exit_label);

        add(reg_from, tail4_simd_w * src_type_size);
        add(reg_to, tail4_simd_w * dst_type_size);
        add(reg_crop_low, tail4_simd_w * wei_type_size);
        add(reg_crop_high, tail4_simd_w * wei_type_size);
        add(reg_input_scale, tail4_simd_w * wei_type_size);
        add(reg_input_shift, tail4_simd_w * wei_type_size);
        if (do_dequantization) {
            add(reg_output_scale, tail4_simd_w * wei_type_size);
            add(reg_output_shift, tail4_simd_w * wei_type_size);
        }

        L(tail_label);

        cmp(reg_block_size, 0);
        jle(exit_label, T_NEAR);

        mov(aux_reg_to, reg_to);
        mov(aux_reg_from, reg_from);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        L(tail_loop_label); {
            cmp(reg_work_amount, 0);
            jle(exit_label, T_NEAR);

            for (int i = 0; i < jqp_.c % tail4_simd_w; i++) {
                movss(xmm_crop_low(0), ptr[reg_crop_low + i * wei_type_size]);
                movss(xmm_crop_high(0), ptr[reg_crop_high + i * wei_type_size]);
                movss(xmm_input_scale(0), ptr[reg_input_scale + i * wei_type_size]);
                movss(xmm_input_shift(0), ptr[reg_input_shift + i * wei_type_size]);
                if (do_dequantization) {
                    movss(xmm_output_scale(0), ptr[reg_output_scale + i * wei_type_size]);
                    movss(xmm_output_shift(0), ptr[reg_output_shift + i * wei_type_size]);
                }

                load_scalar(xmm_val(0), ptr[aux_reg_from + i * src_type_size], jqp_.src_prc);

                uni_vminps(xmm_val(0), xmm_val(0), xmm_crop_high(0));
                uni_vmaxps(xmm_val(0), xmm_val(0), xmm_crop_low(0));
                uni_vfmadd213ps(xmm_val(0), xmm_input_scale(0), xmm_input_shift(0));
                if (do_rounding) uni_vroundps(xmm_val(0), xmm_val(0), 0);
                if (do_dequantization) uni_vfmadd213ps(xmm_val(0), xmm_output_scale(0), xmm_output_shift(0));

                store_scalar(ptr[aux_reg_to + i * dst_type_size], xmm_val(0), jqp_.dst_prc);
            }

            dec(reg_work_amount);
            add(aux_reg_from, reg_src_step);
            add(aux_reg_to, reg_dst_step);

            jmp(tail_loop_label, T_NEAR);
        }

        L(exit_label);
    }

    inline void load_vector(Zmm zmm_src, const Xbyak::Address &op, Precision src_prc) {
        switch (src_prc) {
            case Precision::FP32:
            case Precision::I32:
                uni_vmovups(zmm_src, op);
                break;
            case Precision::I8:
                uni_vpmovsxbd(zmm_src, op);
                break;
            case Precision::U8:
                uni_vpmovzxbd(zmm_src, op);
                break;
            default:
                assert(!"unknown src_prc");
        }

        if (src_prc != Precision::FP32) {
            uni_vcvtdq2ps(zmm_src, zmm_src);
        }
    }

    inline void load_vector(Ymm ymm_src, const Xbyak::Address &op, Precision src_prc) {
        switch (src_prc) {
            case Precision::FP32:
            case Precision::I32:
                uni_vmovups(ymm_src, op);
                break;
            case Precision::I8:
                uni_vpmovsxbd(ymm_src, op);
                break;
            case Precision::U8:
                uni_vpmovzxbd(ymm_src, op);
                break;
            default:
                assert(!"unknown src_prc");
        }

        if (src_prc != Precision::FP32) {
            uni_vcvtdq2ps(ymm_src, ymm_src);
        }
    }

    inline void load_vector(Xmm xmm_src, const Xbyak::Address &op, Precision src_prc) {
        switch (src_prc) {
            case Precision::FP32:
            case Precision::I32:
                uni_vmovups(xmm_src, op);
                break;
            case Precision::I8:
                uni_vpmovsxbd(xmm_src, op);
                break;
            case Precision::U8:
                uni_vpmovzxbd(xmm_src, op);
                break;
            default:
                assert(!"unknown src_prc");
        }

        if (src_prc != Precision::FP32) {
            uni_vcvtdq2ps(xmm_src, xmm_src);
        }
    }

    inline void load_scalar(Xmm xmm_src, const Xbyak::Address &op, Precision src_prc) {
        switch (src_prc) {
            case Precision::FP32:
            case Precision::I32:
                movss(xmm_src, op);
                break;
            case Precision::I8:
                movsx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            case Precision::U8:
                movzx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            default:
                assert(!"unknown src_prc");
        }

        if (src_prc != Precision::FP32) {
            uni_vcvtdq2ps(xmm_src, xmm_src);
        }
    }

    inline void store_vector(const Xbyak::Address &op, Zmm zmm_dst, Precision dst_prc) {
        if (dst_prc != Precision::FP32) {
            uni_vcvtps2dq(zmm_dst, zmm_dst);
        }

        switch (dst_prc) {
            case Precision::FP32:
            case Precision::I32:
                uni_vmovups(op, zmm_dst);
                break;
            case Precision::I8:
                vpmovsdb(op, zmm_dst);
                break;
            case Precision::U8:
                vpmaxsd(zmm_dst, zmm_dst, vmm_zero);
                vpmovusdb(op, zmm_dst);
                break;
            default:
                assert(!"unknown dst_prc");
        }
    }

    inline void store_vector(const Xbyak::Address &op, Ymm ymm_dst, Precision dst_prc) {
        Xmm xmm_dst = Xmm(ymm_dst.getIdx());

        if (dst_prc != Precision::FP32) {
            uni_vcvtps2dq(ymm_dst, ymm_dst);
        }

        switch (dst_prc) {
            case Precision::FP32:
            case Precision::I32:
                uni_vmovups(op, ymm_dst);
                break;
            case Precision::I8:
                uni_vpackssdw(ymm_dst, ymm_dst, ymm_dst);

                vpermq(ymm_dst, ymm_dst, 0x08);

                uni_vpacksswb(ymm_dst, ymm_dst, ymm_dst);

                vmovq(op, xmm_dst);
                break;
            case Precision::U8:
                uni_vpackusdw(ymm_dst, ymm_dst, ymm_dst);

                vpermq(ymm_dst, ymm_dst, 0x08);

                uni_vpackuswb(ymm_dst, ymm_dst, ymm_dst);

                vmovq(op, xmm_dst);
                break;
            default:
                assert(!"unknown dst_prc");
        }
    }

    inline void store_vector(const Xbyak::Address &op, Xmm xmm_dst, Precision dst_prc) {
        if (dst_prc != Precision::FP32) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_prc) {
            case Precision::FP32:
            case Precision::I32:
                uni_vmovups(op, xmm_dst);
                break;
            case Precision::I8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                movd(op, xmm_dst);
                break;
            case Precision::U8:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                movd(op, xmm_dst);
                break;
            default:
                assert(!"unknown dst_prc");
        }
    }

    inline void store_scalar(const Xbyak::Address &op, Xmm xmm_dst, Precision dst_prc) {
        if (dst_prc != Precision::FP32) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_prc) {
            case Precision::FP32:
            case Precision::I32:
                movss(op, xmm_dst);
                break;
            case Precision::I8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case Precision::U8:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            default:
                assert(!"unknown dst_prc");
        }
    }
};

MKLDNNQuantizeNode::MKLDNNQuantizeNode(CNNLayerPtr layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache) {}

void MKLDNNQuantizeNode::init() {
    auto* quantizeLayer = dynamic_cast<QuantizeLayer*>(getCnnLayer().get());
    if (quantizeLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert Quantize layer " << getName();

    levels = quantizeLayer->levels;
    if (levels <= 1)
        THROW_IE_EXCEPTION << "Quantize layer " << getName() << " supports only parameter levels > 1";

    if (getParentEdges().size() != 5)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    for (size_t i = 0; i < getParentEdges().size(); i++) {
        if (getParentEdgesAtPort(i).size() != 1)
            THROW_IE_EXCEPTION << "Quantize layer " << getName() << " has unsupported number of parent edges at port " << i;
    }

    auto initAxisIdx = [&](size_t edgeIdx) {
        auto edge = getParentEdgesAtPort(edgeIdx)[0];

        size_t axisIdx = 0;
        int numberOfNonUnit = 0;
        if (edge->getDims().ndims() > 0) {
            if (edge->getDims()[0] > 1) {
                numberOfNonUnit++;
            }
        }

        for (int i = 1; i < edge->getDims().ndims(); i++) {
            if (edge->getDims()[i] > 1) {
                axisIdx = i;
                numberOfNonUnit++;
            }
        }
        if (numberOfNonUnit > 1) {
            THROW_IE_EXCEPTION << "Quantize layer " << getName() << " supports only per-tensor and per-channel quantizations";
        }

        return axisIdx;
    };

    axis = getParentEdgesAtPort(0)[0]->getDims().ndims() == 1 ? 0 : 1;

    std::set<size_t> quantizationParamsAxisesIdxs;
    std::set<size_t> quantizationParamsAxisesSizes;

    auto inputLowAxis = initAxisIdx(1);
    isInputLowBroadcasted = getParentEdgesAtPort(1)[0]->getDims()[inputLowAxis] == 1;
    if (!isInputLowBroadcasted) {
        quantizationParamsAxisesIdxs.insert(inputLowAxis);
        quantizationParamsAxisesSizes.insert(getParentEdgesAtPort(1)[0]->getDims()[inputLowAxis]);
    }

    auto inputHighAxis = initAxisIdx(2);
    isInputHighBroadcasted = getParentEdgesAtPort(2)[0]->getDims()[inputHighAxis] == 1;
    if (!isInputHighBroadcasted) {
        quantizationParamsAxisesIdxs.insert(inputHighAxis);
        quantizationParamsAxisesSizes.insert(getParentEdgesAtPort(2)[0]->getDims()[inputHighAxis]);
    }

    auto outputLowAxis = initAxisIdx(3);
    isOutputLowBroadcasted = getParentEdgesAtPort(3)[0]->getDims()[outputLowAxis] == 1;
    if (!isOutputLowBroadcasted) {
        quantizationParamsAxisesIdxs.insert(outputLowAxis);
        quantizationParamsAxisesSizes.insert(getParentEdgesAtPort(3)[0]->getDims()[outputLowAxis]);
    }

    auto outputHighAxis = initAxisIdx(4);
    isOutputHighBroadcasted = getParentEdgesAtPort(4)[0]->getDims()[outputHighAxis] == 1;
    if (!isOutputHighBroadcasted) {
        quantizationParamsAxisesIdxs.insert(outputHighAxis);
        quantizationParamsAxisesSizes.insert(getParentEdgesAtPort(4)[0]->getDims()[outputHighAxis]);
    }

    if (quantizationParamsAxisesIdxs.size() > 1 || quantizationParamsAxisesSizes.size() > 1)
        THROW_IE_EXCEPTION << "Unsupported input sizes for Quantize layer with name " << getName();

    if (quantizationParamsAxisesIdxs.size() == 1) {
        axis = *quantizationParamsAxisesIdxs.begin();
    }

    auto inputLowAxisSize = getParentEdgesAtPort(1)[0]->getDims()[inputLowAxis];
    auto inputHighAxisSize = getParentEdgesAtPort(2)[0]->getDims()[inputHighAxis];
    auto outputLowAxisSize = getParentEdgesAtPort(3)[0]->getDims()[outputLowAxis];
    auto outputHighAxisSize = getParentEdgesAtPort(4)[0]->getDims()[outputHighAxis];

    size_t axisRealSize = static_cast<size_t>(getParentEdgesAtPort(0)[0]->getDims()[axis]);
    size_t axisPaddedSize = static_cast<size_t>(rnd_up(getParentEdgesAtPort(0)[0]->getDims()[axis], 16));

    if (quantizationParamsAxisesSizes.size() == 1) {
        if (*quantizationParamsAxisesSizes.begin() != axisRealSize)
            THROW_IE_EXCEPTION << "Unsupported input sizes for Quantize layer with name " << getName();
    }

    for (size_t i = 1; i < getParentEdges().size(); i++) {
        if (!getParentEdgesAtPort(i)[0]->getParent()->isConstant())
            THROW_IE_EXCEPTION << "Quantize layer with name " << getName() << " has non const input on " << i << " port";
        auto prec = getCnnLayer()->insData[i].lock()->getPrecision();
        if (prec != Precision::FP32)
            THROW_IE_EXCEPTION << "Quantize layer with name " << getName() << " has unsupported precision " << prec << " on " << i << " port";
    }

    auto inputLowBlob = dynamic_cast<TBlob<float>*>(getParentEdgesAtPort(1)[0]->getParent()->getCnnLayer()->blobs["custom"].get());
    auto inputLowData = inputLowBlob->buffer().as<float*>();

    auto inputHighBlob = dynamic_cast<TBlob<float>*>(getParentEdgesAtPort(2)[0]->getParent()->getCnnLayer()->blobs["custom"].get());
    auto inputHighData = inputHighBlob->buffer().as<float*>();

    auto outputLowBlob = dynamic_cast<TBlob<float>*>(getParentEdgesAtPort(3)[0]->getParent()->getCnnLayer()->blobs["custom"].get());
    auto outputLowData = outputLowBlob->buffer().as<float*>();

    auto outputHighBlob = dynamic_cast<TBlob<float>*>(getParentEdgesAtPort(4)[0]->getParent()->getCnnLayer()->blobs["custom"].get());
    auto outputHighData = outputHighBlob->buffer().as<float*>();

    bool binarization = levels == 2;

    if (binarization) {
        for (int i = 0; i < outputLowAxisSize; i++) {
            if (outputLowData[i] != 1.f && outputLowData[i] != 0.f) {
                binarization = false;
                break;
            }
        }

        for (int i = 0; i < outputHighAxisSize; i++) {
            if (outputHighData[i] != 1.f && outputHighData[i] != 0.f) {
                binarization = false;
                break;
            }
        }

        for (ptrdiff_t i = 0; i < std::max(inputLowAxisSize, inputHighAxisSize); i++) {
            if (inputLowData[isInputLowBroadcasted ? 0 : i] != inputHighData[isInputHighBroadcasted ? 0 : i]) {
                binarization = false;
                break;
            }
        }
    }

    if (binarization) {
        quantizeOpType = QuantizeOpType::Binarization;

        binarizationThresholds.resize(axisPaddedSize);
        binarizationOutputMask.resize(axisPaddedSize);

        for (int i = 0; i < axisRealSize; i++) {
            binarizationThresholds[i] = inputLowData[isInputLowBroadcasted ? 0 : i];
            binarizationOutputMask[i] = outputHighData[isOutputHighBroadcasted ? 0 : i] == 1.f ? 0xffffffff : 0x00000000;
        }
    } else {
        auto allElementsAreEqual = [&](const float* data, size_t size) {
            if (size == 0)
                return true;

            auto first = data[0];
            for (int i = 1; i < size; i++) {
                if (data[i] != first)
                    return false;
            }

            return true;
        };

        if (allElementsAreEqual(inputLowData, inputLowAxisSize)) {
            inputLowAxisSize = 1;
            isInputLowBroadcasted = true;
        }

        if (allElementsAreEqual(inputHighData, inputHighAxisSize)) {
            inputHighAxisSize = 1;
            isInputHighBroadcasted = true;
        }

        if (allElementsAreEqual(outputLowData, outputLowAxisSize)) {
            outputLowAxisSize = 1;
            isOutputLowBroadcasted = true;
        }

        if (allElementsAreEqual(outputHighData, outputHighAxisSize)) {
            outputHighAxisSize = 1;
            isOutputHighBroadcasted = true;
        }

        cropLow.resize(inputLowAxisSize);
        cropHigh.resize(inputHighAxisSize);
        inputScale.resize(std::max(inputLowAxisSize, inputHighAxisSize));
        inputShift.resize(std::max(inputLowAxisSize, inputHighAxisSize));
        outputScale.resize(std::max(outputLowAxisSize, outputHighAxisSize));
        outputShift.resize(outputLowAxisSize);

        bool quantizationOnly = true;

        for (int i = 0; i < cropLow.size(); i++) {
            float il = inputLowData[isInputLowBroadcasted ? 0 : i];

            cropLow[i] = il;
        }

        for (int i = 0; i < cropHigh.size(); i++) {
            float ih = inputHighData[isInputHighBroadcasted ? 0 : i];

            cropHigh[i] = ih;
        }

        for (int i = 0; i < inputScale.size(); i++) {
            float il = inputLowData[isInputLowBroadcasted ? 0 : i];
            float ih = inputHighData[isInputHighBroadcasted ? 0 : i];

#if defined(VALIDATE_QUANTIZATION_RANGES)
            if ((il == ih && levels != 2) || il > ih || std::isnan(il) || std::isnan(ih) || std::isinf(il) || std::isinf(ih)) {
                THROW_IE_EXCEPTION << "Quantize layer with name '" << getName() << "' has invalid input quantize ranges: "
                                   << "inputLow = " << il << ", inputHigh = " << ih;
            }
#endif

            inputScale[i] = (levels - 1) / (ih - il);
            inputShift[i] = -il * (levels - 1) / (ih - il);
        }

        for (int i = 0; i < outputScale.size(); i++) {
            float ol = outputLowData[isOutputLowBroadcasted ? 0 : i];
            float oh = outputHighData[isOutputHighBroadcasted ? 0 : i];

#if defined(VALIDATE_QUANTIZATION_RANGES)
            if (std::isnan(ol) || std::isnan(oh) || std::isinf(ol) || std::isinf(oh)) {
                THROW_IE_EXCEPTION << "Quantize layer with name '" << getName() << "' has wrong output quantize ranges: "
                                   << "outputLow = " << ol << ", outputHigh = " << oh;
            }
#endif

            outputScale[i] = (oh - ol) / (levels - 1);

            if (outputScale[i] != 1.f)
                quantizationOnly = false;
        }

        for (int i = 0; i < outputShift.size(); i++) {
            float ol = outputLowData[isOutputLowBroadcasted ? 0 : i];

            outputShift[i] = ol;

            if (outputShift[i] != 0.f)
                quantizationOnly = false;
        }

        quantizeOpType = quantizationOnly ? QuantizeOpType::Quantization : QuantizeOpType::FakeQuantization;
    }

    if (binarization) {
        inputPrecision = Precision::FP32;
        outputPrecision = Precision::BIN;
    } else {
        inputPrecision = getCnnLayer()->insData[0].lock()->getPrecision();
        outputPrecision = getCnnLayer()->outData[0]->getPrecision();

        if (inputPrecision != Precision::FP32 && inputPrecision != Precision::U8 && inputPrecision != Precision::I8)
            inputPrecision = Precision::FP32;

        if (outputPrecision != Precision::FP32 && outputPrecision != Precision::U8 && outputPrecision != Precision::I8)
            outputPrecision = Precision::FP32;
    }
}


std::vector<mkldnn::memory::format_tag> MKLDNNQuantizeNode::getDataFormats() const {
    // Special case for first FQ in the network
    if (getParentEdgesAtPort(0)[0]->getDims()[getAxis()] == 3) {
        return { MKLDNNMemory::GetPlainFormat(getParentEdgesAtPort(0)[0]->getDims()) };
    } else {
        if (isBinarization()) {
            return {memory::format_tag::nhwc};
        } else {
            switch (getParentEdgesAtPort(0)[0]->getDims().ndims()) {
                case 4:
                    if (getAxis() == 1) {
                        auto blkFormat = mayiuse(cpu::x64::avx512_common) ? memory::format_tag::nChw16c : memory::format_tag::nChw8c;
                        return {blkFormat, memory::format_tag::nhwc, memory::format_tag::nchw};
                    } else {
                        return {memory::format_tag::nchw};
                    }
                case 5:
                    if (getAxis() == 1) {
                        auto blkFormat = mayiuse(cpu::x64::avx512_common) ? memory::format_tag::nCdhw16c : memory::format_tag::nCdhw8c;
                        return {blkFormat, memory::format_tag::ndhwc, memory::format_tag::ncdhw};
                    } else {
                        return {memory::format_tag::ncdhw};
                    }
                default:
                    return {MKLDNNMemory::GetPlainFormat(getParentEdgesAtPort(0)[0]->getDims())};
            }
        }
    }
}

void MKLDNNQuantizeNode::getSupportedDescriptors() {
    std::string errorPrefix = "Quantize layer with name '" + getName() + "' ";

    if (getParentEdgesAtPort(0)[0]->getDims().ndims() != getChildEdgesAtPort(0)[0]->getDims().ndims()) {
        THROW_IE_EXCEPTION << errorPrefix << "has different ranks for input and output tensors";
    }

    if (getParentEdgesAtPort(0)[0]->getDims().ndims() < 1ul || getParentEdgesAtPort(0)[0]->getDims().ndims() > 5ul) {
        THROW_IE_EXCEPTION << errorPrefix << "has unsupported number of dimensions for input at edge 0";
    }

    if (isBinarization()) {
        if (getParentEdgesAtPort(0)[0]->getDims().ndims() != 4ul) {
            THROW_IE_EXCEPTION << errorPrefix << "doesn't support input/output rank != 4";
        }
    }

    if (getAxis() != 1) {
        if (isBinarization())
            THROW_IE_EXCEPTION << errorPrefix << "doesn't support non per-tensor binarization for axis: " << getAxis();
        if (getAxis() != 0)
            THROW_IE_EXCEPTION << errorPrefix << "doesn't support non per-tensor quantization for axis: " << getAxis();
    }
}

void MKLDNNQuantizeNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

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

    if (!mayiuse(cpu::x64::sse41) || getAxis() != 1) {
        impl_type = impl_desc_type::ref;

        if (!isBinarization()) {
            inputPrecision = Precision::FP32;
            outputPrecision = Precision::FP32;
        }
    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getInputPrecision());
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getOutputPrecision());

    for (auto& fmt : getDataFormats()) {
        LayerConfig config;
        config.dynBatchSupport = true;
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            DataConfig dataConfig;
            dataConfig.inPlace = -1;
            dataConfig.constant = false;

            if (i == 0) {
                dataConfig.desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDataType, fmt);
            } else {
                dataConfig.desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), memory::data_type::f32,
                                                   MKLDNNMemory::GetPlainFormat(getParentEdgeAt(i)->getDims()));
            }
            config.inConfs.push_back(dataConfig);
        }

        DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;
        dataConfig.desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, fmt);
        config.outConfs.push_back(dataConfig);

        supportedPrimitiveDescriptors.push_back({config, impl_type, fmt});
    }
}

void MKLDNNQuantizeNode::createPrimitive() {
    auto config = getSelectedPrimitiveDescriptor()->getConfig();

    auto inDims = config.inConfs[0].desc.getDims();
    jqp.c = inDims.size() > 1 ? inDims[1] : 1;

    jqp.src_prc = config.inConfs[0].desc.getPrecision();
    jqp.wei_prc = Precision::FP32;
    jqp.dst_prc = config.outConfs[0].desc.getPrecision();

    jqp.src_layout = config.inConfs[0].desc.getLayout();

    jqp.op_type = quantizeOpType;

    auto selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    if (!selectedPrimitiveDescriptor)
        THROW_IE_EXCEPTION << "CPU quantize node with name '" << getName() << "' doesn't have primitive descriptors.";

    if (selectedPrimitiveDescriptor->getImplementationType() != impl_desc_type::ref) {
        if (mayiuse(cpu::x64::avx512_common)) {
            if (isBinarization())
                quantize_kernel.reset(new jit_uni_binarization_kernel<cpu::x64::avx512_common>(jqp));
            else
                quantize_kernel.reset(new jit_uni_quantization_kernel<cpu::x64::avx512_common>(jqp));
        } else if (mayiuse(cpu::x64::avx2)) {
            if (isBinarization())
                quantize_kernel.reset(new jit_uni_binarization_kernel<cpu::x64::avx2>(jqp));
            else
                quantize_kernel.reset(new jit_uni_quantization_kernel<cpu::x64::avx2>(jqp));
        } else if (mayiuse(cpu::x64::sse41)) {
            if (isBinarization())
                quantize_kernel.reset(new jit_uni_binarization_kernel<cpu::x64::sse41>(jqp));
            else
                quantize_kernel.reset(new jit_uni_quantization_kernel<cpu::x64::sse41>(jqp));
        }
    }
    if (quantize_kernel)
        quantize_kernel->create_ker();

    size_t axisSize = getParentEdgeAt(0)->getDims()[getAxis()];
    size_t axisPaddedSize = rnd_up(axisSize, 16);

    MKLDNNMemoryDesc weightsDataDesc = {{(uint32_t)axisPaddedSize}, memory::data_type::f32, memory::format_tag::x};

    if (isBinarization()) {
        auto binarizationThresholdsDataMem = std::make_shared<MKLDNNMemory>(getEngine());
        binarizationThresholdsDataMem->Create(weightsDataDesc, getBinarizationTresholdsPtr());
        internalBlobMemory.push_back(binarizationThresholdsDataMem);

        auto binarizationMaskDataMem = std::make_shared<MKLDNNMemory>(getEngine());
        binarizationMaskDataMem->Create(weightsDataDesc, getBinarizationOutputMaskPtr());
        internalBlobMemory.push_back(binarizationMaskDataMem);
    } else if (levels != 2) {
        auto pushInternalBlob = [&](std::vector<float>& data) {
            if (data.size() == 1)
                data.resize(axisPaddedSize, data[0]);
            else
                data.resize(axisPaddedSize);
            auto memory = std::make_shared<MKLDNNMemory>(getEngine());
            memory->Create(weightsDataDesc, &data[0]);
            internalBlobMemory.push_back(memory);
        };

        pushInternalBlob(cropLow);
        pushInternalBlob(cropHigh);
        pushInternalBlob(inputScale);
        pushInternalBlob(inputShift);
        pushInternalBlob(outputScale);
        pushInternalBlob(outputShift);
    }
}

void MKLDNNQuantizeNode::executeReference() {
    auto &srcMemory = getParentEdgeAt(0)->getMemoryPtr();
    auto &dstMemory = getChildEdgeAt(0)->getMemoryPtr();

    auto src = reinterpret_cast<const float *>(srcMemory->GetPtr());

    auto config = getSelectedPrimitiveDescriptor()->getConfig();
    auto srcDims = config.inConfs[0].desc.getDims();
    auto dstDims = config.outConfs[0].desc.getDims();

    auto s_str = config.inConfs[0].desc.getBlockingDesc().getStrides();
    auto d_str = config.outConfs[0].desc.getBlockingDesc().getStrides();

    const int N = srcDims[0];
    const int C = srcDims.size() > 1 ? srcDims[1] : 1;
    const int D = srcDims.size() == 5 ? srcDims[2] : 1;
    const int H = srcDims.size() == 3 ? srcDims[2] : srcDims.size() > 3 ? srcDims[srcDims.size() - 2] : 1;
    const int W = srcDims.size() > 3 ? srcDims[srcDims.size() - 1] : 1;

    if (jqp.op_type == QuantizeOpType::Binarization) {
        size_t tmp = s_str[s_str.size() - 1];
        for (int i = s_str.size() - 1; i > 1; i--) {
            s_str[i] = s_str[i - 1];
        }
        s_str[1] = tmp;

        tmp = d_str[d_str.size() - 1];
        for (int i = d_str.size() - 1; i > 1; i--) {
            d_str[i] = d_str[i - 1];
        }
        d_str[1] = tmp;

        auto dst = reinterpret_cast<uint8_t *>(dstMemory->GetPtr());

        const int nbits = 8;
        const int CB = impl::utils::div_up(C, nbits);

        auto thresholds = reinterpret_cast<const float*>(internalBlobMemory[0]->GetData());
        auto output_mask = reinterpret_cast<const uint32_t*>(internalBlobMemory[1]->GetData());

        parallel_nd(N, CB, D, H, W, [&](int n, int cb, int d, int h, int w) {
            uint8_t bin_val = 0x00;
            for (int c = cb * nbits, shift = 0; c < std::min(C, (cb + 1) * nbits); c++, shift++) {
                size_t src_off = srcDims.size() == 4 ?
                                    n * s_str[0] + c * s_str[1] + h * s_str[2] + w * s_str[3] :
                                 srcDims.size() == 5 ?
                                    n * s_str[0] + c * s_str[1] + d * s_str[2] + h * s_str[3] + w * s_str[4] :
                                    n * s_str[0] + c * s_str[1];

                float val = src[src_off];
                float thr = thresholds[c];
                uint32_t out_mask = output_mask[c];

                uint32_t res = (val > thr) ? 0xffffffff : 0x00000000;

                auto bit = uint8_t(res == out_mask);
                bin_val |= (bit << shift);
            }

            size_t dst_off = dstDims.size() == 4 ?
                                n * d_str[0] + (cb * nbits) * d_str[1] + h * d_str[2] + w * d_str[3] :
                             dstDims.size() == 5 ?
                                 n * d_str[0] + (cb * nbits) * d_str[1] + d * d_str[2] + h * d_str[3] + w * d_str[4] :
                                 n * d_str[0] + (cb * nbits) * d_str[1];

            dst[dst_off / nbits] = bin_val;
        });
    } else {
        auto dst = reinterpret_cast<float *>(dstMemory->GetPtr());

        auto crop_low = reinterpret_cast<const float*>(internalBlobMemory[0]->GetData());
        auto crop_high = reinterpret_cast<const float*>(internalBlobMemory[1]->GetData());
        auto input_scale = reinterpret_cast<const float*>(internalBlobMemory[2]->GetData());
        auto input_shift = reinterpret_cast<const float*>(internalBlobMemory[3]->GetData());
        auto output_scale = reinterpret_cast<const float*>(internalBlobMemory[4]->GetData());
        auto output_shift = reinterpret_cast<const float*>(internalBlobMemory[5]->GetData());

        parallel_nd(N, C, D, H, W, [&](int n, int c, int d, int h, int w) {
            size_t src_off = srcDims.size() == 5 ?
                                n * s_str[0] + c * s_str[1] + d * s_str[2] + h * s_str[3] + w * s_str[4] :
                             srcDims.size() == 4 ?
                                n * s_str[0] + c * s_str[1] + h * s_str[2] + w * s_str[3] :
                             srcDims.size() == 3 ?
                                n * s_str[0] + c * s_str[1] + h * s_str[2] :
                             srcDims.size() == 2 ?
                                n * s_str[0] + c * s_str[1] :
                                n * s_str[0];

            float src_val = src[src_off];

            int wei_idx = getAxis() == 0 ? n : c;
            float cl = crop_low[wei_idx];
            float ch = crop_high[wei_idx];
            float isc = input_scale[wei_idx];
            float ish = input_shift[wei_idx];
            float osc = output_scale[wei_idx];
            float osh = output_shift[wei_idx];

            float dst_val = nstl::min(ch, nstl::max(cl, src_val));
            dst_val = dst_val * isc + ish;
            dst_val = roundf(dst_val);
            dst_val = dst_val * osc + osh;

            size_t dst_off = dstDims.size() == 5 ?
                             n * d_str[0] + c * d_str[1] + d * d_str[2] + h * d_str[3] + w * d_str[4] :
                             dstDims.size() == 4 ?
                             n * d_str[0] + c * d_str[1] + h * d_str[2] + w * d_str[3] :
                             dstDims.size() == 3 ?
                             n * d_str[0] + c * d_str[1] + h * d_str[2] :
                             dstDims.size() == 2 ?
                             n * d_str[0] + c * d_str[1] :
                             n * d_str[0];

            dst[dst_off] = dst_val;
        });
    }
}

void MKLDNNQuantizeNode::executeBinarization() {
    auto &srcMemory = getParentEdgeAt(0)->getMemoryPtr();
    auto &dstMemory = getChildEdgeAt(0)->getMemoryPtr();

    auto src = reinterpret_cast<const uint8_t *>(srcMemory->GetPtr());
    auto dst = reinterpret_cast<uint8_t *>(dstMemory->GetPtr());

    auto thresholds = reinterpret_cast<const float*>(internalBlobMemory[0]->GetData());
    auto output_mask = reinterpret_cast<const float*>(internalBlobMemory[1]->GetData());

    auto config = getSelectedPrimitiveDescriptor()->getConfig();
    auto src_dims = config.inConfs[0].desc.getDims();

    std::vector<size_t> s_str = config.inConfs[0].desc.getBlockingDesc().getStrides();
    size_t tmp = s_str[s_str.size() - 1];
    for (int i = s_str.size() - 1; i > 1; i--) {
        s_str[i] = s_str[i - 1];
    }
    s_str[1] = tmp;

    const int N = src_dims[0];
    const int C = src_dims[1];
    const int H = src_dims[2];
    const int W = src_dims[3];

    int nbits = 8;

    parallel_nd(N, H, W, [&](int n, int h, int w) {
        auto arg = jit_quantize_call_args();

        arg.from    = &src[(n * s_str[0] + h * s_str[2] + w * s_str[3]) * sizeof(float)];
        arg.to      = &dst[(n * s_str[0] + h * s_str[2] + w * s_str[3]) / nbits];
        arg.thresholds = &thresholds[0];
        arg.output_mask = &output_mask[0];
        arg.work_amount = (size_t)C;

        (*quantize_kernel)(&arg);
    });
}

void MKLDNNQuantizeNode::executeQuantization() {
    auto &srcMemory = getParentEdgeAt(0)->getMemoryPtr();
    auto &dstMemory = getChildEdgeAt(0)->getMemoryPtr();

    auto src = reinterpret_cast<const uint8_t *>(srcMemory->GetPtr());
    auto dst = reinterpret_cast<uint8_t *>(dstMemory->GetPtr());

    auto crop_low = reinterpret_cast<const float*>(internalBlobMemory[0]->GetData());
    auto crop_high = reinterpret_cast<const float*>(internalBlobMemory[1]->GetData());
    auto input_scale = reinterpret_cast<const float*>(internalBlobMemory[2]->GetData());
    auto input_shift = reinterpret_cast<const float*>(internalBlobMemory[3]->GetData());
    auto output_scale = reinterpret_cast<const float*>(internalBlobMemory[4]->GetData());
    auto output_shift = reinterpret_cast<const float*>(internalBlobMemory[5]->GetData());

    auto config = getSelectedPrimitiveDescriptor()->getConfig();
    auto srcDims = config.inConfs[0].desc.getDims();

    bool is_blk_format = jqp.src_layout != Layout::NHWC && jqp.src_layout != Layout::NDHWC;
    int blk_size = (jqp.src_layout == Layout::CHW ||
                    jqp.src_layout == Layout::NCHW ||
                    jqp.src_layout == Layout::NCDHW) ? 1 : mayiuse(cpu::x64::avx512_common) ? 16 : 8;

    auto src_type_size = jqp.src_prc.size();
    auto dst_type_size = jqp.dst_prc.size();

    std::vector<size_t> s_str = config.inConfs[0].desc.getBlockingDesc().getStrides();

    if (jqp.src_layout == BLOCKED) {
        s_str[1] /= blk_size;
    }

    if (jqp.src_layout == Layout::NHWC || jqp.src_layout == Layout::NDHWC) {
        size_t tmp = s_str[s_str.size() - 1];
        for (int i = s_str.size() - 1; i > 1; i--) {
            s_str[i] = s_str[i - 1];
        }
        s_str[1] = tmp;
    }

    const int N = srcDims[0];
    const int C = srcDims[1];
    const int CB = div_up(C, blk_size);
    const int D = srcDims.size() == 5 ? srcDims[2] : 1;
    const int H = srcDims.size() == 3 ? srcDims[2] : srcDims.size() > 3 ? srcDims[srcDims.size() - 2] : 1;
    const int W = srcDims.size() > 3 ? srcDims[srcDims.size() - 1] : 1;

    if (jqp.src_layout == Layout::CHW) {
        parallel_nd(N, CB, D, [&](int n, int cb, int d) {
            auto arg = jit_quantize_call_args();

            int c = cb * blk_size;

            size_t data_off = n * s_str[0] + c * s_str[1];

            arg.from = &src[data_off * src_type_size];
            arg.to = &dst[data_off * dst_type_size];
            arg.crop_low = &crop_low[c];
            arg.crop_high = &crop_high[c];
            arg.input_scale = &input_scale[c];
            arg.input_shift = &input_shift[c];
            arg.output_scale = &output_scale[c];
            arg.output_shift = &output_shift[c];

            arg.src_step = (size_t) blk_size * src_type_size;
            arg.dst_step = (size_t) blk_size * dst_type_size;
            arg.block_size = (size_t) blk_size;
            arg.work_amount = (size_t)H;

            (*quantize_kernel)(&arg);
        });
    } else {
        parallel_nd(N, CB, D, H, [&](int n, int cb, int d, int h) {
            auto arg = jit_quantize_call_args();

            int c = cb * blk_size;

            size_t data_off = srcDims.size() == 2 ?
                                    n * s_str[0] + c * s_str[1] :
                              srcDims.size() == 3 || srcDims.size() == 4 ?
                                    n * s_str[0] + c * s_str[1] + h * s_str[2] :
                                    n * s_str[0] + c * s_str[1] + d * s_str[2] + h * s_str[3];

            arg.from = &src[data_off * src_type_size];
            arg.to = &dst[data_off * dst_type_size];
            arg.crop_low = &crop_low[c];
            arg.crop_high = &crop_high[c];
            arg.input_scale = &input_scale[c];
            arg.input_shift = &input_shift[c];
            arg.output_scale = &output_scale[c];
            arg.output_shift = &output_shift[c];

            arg.src_step = is_blk_format ? (size_t) blk_size * src_type_size : (size_t) C * src_type_size;
            arg.dst_step = is_blk_format ? (size_t) blk_size * dst_type_size : (size_t) C * dst_type_size;
            arg.block_size = (is_blk_format && jqp.src_layout != Layout::NC) ? (size_t) blk_size : nstl::min(blk_size, C - c);
            arg.work_amount = (size_t) W;

            (*quantize_kernel)(&arg);
        });
    }
}

void MKLDNNQuantizeNode::execute(mkldnn::stream strm) {
    auto selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    if (!selectedPrimitiveDescriptor)
        THROW_IE_EXCEPTION << "CPU quantize node with name '" << getName() << "' doesn't have primitive descriptors.";

    if (selectedPrimitiveDescriptor->getImplementationType() != impl_desc_type::ref) {
        if (jqp.op_type == QuantizeOpType::Binarization)
            executeBinarization();
        else
            executeQuantization();
    } else {
        executeReference();
    }
}

void MKLDNNQuantizeNode::appendPostOps(mkldnn::post_ops& ops) {
    // MKLDNN quantization_injectors assumes that quantization data memory is always aligned on 16
    // by length of AVX512 vector register which is also enough for AVX2 and SSE42 implementations.
    // Otherwise it can lead to buffer over-read and performance penalties due to denormals.
    const size_t bufferAlignment = 16;

    if (quantizeOpType == QuantizeOpType::Binarization) {
        if (!isPostOpDataInitialized) {
            size_t paddedSize = rnd_up(binarizationThresholds.size(), bufferAlignment);
            binarizationThresholds.resize(paddedSize, 0);
            binarizationOutputMask.resize(paddedSize, 0);
        }

        ops.append_binarization(mkldnn::algorithm::binarization_depthwise, (const float*)&binarizationThresholds[0], (const float*)&binarizationOutputMask[0]);
    } else {
        if (!isPostOpDataInitialized) {
            if (cropLow.size() > 1)
                cropLow.resize(rnd_up(cropLow.size(), bufferAlignment), 0);
            if (cropHigh.size() > 1)
                cropHigh.resize(rnd_up(cropHigh.size(), bufferAlignment), 0);
            if (inputScale.size() > 1)
                inputScale.resize(rnd_up(inputScale.size(), bufferAlignment), 0);
            if (inputShift.size() > 1)
                inputShift.resize(rnd_up(inputShift.size(), bufferAlignment), 0);
            if (outputScale.size() > 1)
                outputScale.resize(rnd_up(outputScale.size(), bufferAlignment), 0);
            if (outputShift.size() > 1)
                outputShift.resize(rnd_up(outputShift.size(), bufferAlignment), 0);

            cropLowData.set(cropLow.size(), 1 << 1, &cropLow[0]);
            cropHighData.set(cropHigh.size(), 1 << 1, &cropHigh[0]);
            inputScaleData.set(inputScale.size(), 1 << 1, &inputScale[0]);
            inputShiftData.set(inputShift.size(), 1 << 1, &inputShift[0]);
            outputScaleData.set(outputScale.size(), 1 << 1, &outputScale[0]);
            outputShiftData.set(outputShift.size(), 1 << 1, &outputShift[0]);
        }

        mkldnn::algorithm alg = quantizeOpType == QuantizeOpType::FakeQuantization ? mkldnn::algorithm::quantization_quantize_dequantize :
                                                                                     mkldnn::algorithm::quantization_quantize;

        ops.append_quantization(alg, &cropLowData, &cropHighData, &inputScaleData, &inputShiftData, &outputScaleData, &outputShiftData);
    }

    if (!isPostOpDataInitialized)
        isPostOpDataInitialized = true;
}

bool MKLDNNQuantizeNode::isNeedToDecompose(const std::shared_ptr<const ngraph::Node>& node) {
    if (const auto fq = std::dynamic_pointer_cast<const ngraph::opset1::FakeQuantize>(node)) {
        for (size_t i = 0; i < fq->get_input_size(); i++) {
            if (fq->get_input_shape(i).size() > 5)
                return true;
        }

        for (size_t i = 1; i < fq->get_input_size(); i++) {
            size_t count_not_unit_axis = 0;
            auto shape = fq->get_input_shape(i);

            if (ngraph::shape_size(shape) != 1) {
                size_t not_unit_axis = 0;
                for (size_t i = 0; i < shape.size(); i++) {
                    if (shape[i] > 1) {
                        not_unit_axis = i;
                        count_not_unit_axis++;
                    }
                }
                if (count_not_unit_axis > 1 || not_unit_axis > 1)
                    return true;
            }
        }
    }
    return false;
}

bool MKLDNNQuantizeNode::created() const {
    return getType() == Quantize;
}

REG_MKLDNN_PRIM_FOR(MKLDNNQuantizeNode, Quantize);
