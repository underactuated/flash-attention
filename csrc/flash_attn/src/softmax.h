/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
//#include "philox.cuh"
#include "utils.h"

#include "kernel_traits.h"
#include "curand_kernel.h"

//#include <cstdio> // For printf
//#include <iostream> // For std::cout (optional, can use printf for messages)
#include <limits>

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}

// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            // The following macro will disable the use of fma.
            // See: https://github.com/pytorch/pytorch/issues/121558 for more details
            // This macro is set in PyTorch and not FlashAttention
            #ifdef UNFUSE_FMA
                tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
            #else
                tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            #endif
        }
    }
}

// Apply the exp to all the elements.
template <bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void max_scale_exp2_sum(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max, Tensor<Engine1, Layout1> &sum, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        MaxOp<float> max_op;
        max(mi) = zero_init ? tensor(mi, 0) : max_op(max(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            max(mi) = max_op(max(mi), tensor(mi, ni));
        }
        max(mi) = Allreduce<4>::run(max(mi), max_op);
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * scale;
        sum(mi) = 0;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            sum(mi) += tensor(mi, ni);
        }
        SumOp<float> sum_op;
        sum(mi) = Allreduce<4>::run(sum(mi), sum_op);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

//#define ww_size 32 * 1 //32 * 8
#define PRINT_BID 30 //1
            
#if 1
template <int kNRows, typename Kernel_traits>
struct StochSparse_simple {

    //const float c = 10; //10; //50; //10; //1; //50; //20; //10; //5; //10; //1; //10; //1e-30; //10;
    //const float overc = 1. / c;

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_sum_tot;

    // do we need it here?
    //static constexpr int kBlockM = Kernel_traits::kBlockM;
    //static constexpr int kBlockN = Kernel_traits::kBlockN;

    ///*
    const int ww_size;
    //float warp_weights [ww_size];
    int ww_count = 0;

    const float softmax_scale = 1./8;

    float prev_max = 0;
    float prev_sum = 0;

    const int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    const int tid = bid * blockDim.x + threadIdx.x;
    //int thread_offset = tid * ww_size;
    const int thread_offset = int(tid / 32) * 32 * ww_size + (tid % 32);

    //////////////// coordinates ////////////////

private:

    __host__ __device__ static auto get_tcaccs() {
        constexpr int kBlockM = Kernel_traits::kBlockM;
        constexpr int kBlockN = Kernel_traits::kBlockN;
        const int tidx = threadIdx.x;
        typename Kernel_traits::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(tidx);
        Tensor caccs = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
        return thr_mma.partition_C(caccs);
    };

public:

    decltype(get_tcaccs()) tcaccs; // = get_tcaccs();

    __device__ StochSparse_simple(int ww_size_): ww_size(ww_size_) {
        tcaccs = get_tcaccs();
    };

    template<typename Tensor0>
    __device__ void original_coordinates(Tensor0 &acc_s) { // ? todo: replace acc_s with scores
        #if 1
        //if (thread(0, PRINT_BID)) {
        //if (thread0()) {
        if (thread(100, 0)) {

            printf("tidx = %d\n", tid);

            auto tcaccs_lo = FLASH_NAMESPACE::convert_layout_acc_rowcol(tcaccs.layout());
            printf("size = %d\n", (int)size(acc_s));
            for (int i = 0; i < size(acc_s); ++i) {
                auto coord1 = tcaccs_lo(i);
                auto coord = tcaccs(i);
                //printf(" coord ");
                //print(coord);
                printf("coord ");
                print(coord);
                printf("\n");
            }
        }
        #endif
    };
    
    //////////////// end coordiantes ////////////

    template <typename Tensor1>
    //__device__ void store_wws (const Tensor1 &row_max, Tensor1 &row_sum) {
    __device__ void store_wws (const Tensor1 &row_max, const Tensor1 &row_sum, float* g_row_sum) {
        /*
        constexpr int mi_max = 4; //decltype(size<0>(scores))::value;
        float s = 0;
        #pragma unroll
        for (int mi = 0; mi < mi_max; ++mi) {
            s += (row_max(mi) + row_sum(mi));
        }
        //*/

        ///*

        //Tensor scores_max_prev = make_fragment_like(row_max); cute::copy(row_max, scores_max_prev);
        
        Tensor1 row_sum_tot;
        cute::copy(row_sum, row_sum_tot);
        SumOp<float> sum_op;
        //quad_allreduce_(row_sum_tot, row_sum, sum_op);
        //cute::copy(row_sum, row_sum_tot);
        quad_allreduce_(row_sum_tot, row_sum_tot, sum_op);
        
        /*
        float curr_max = row_max(0);
        float curr_sum = row_sum_tot(0);
        //float curr_sum = row_sum(0);
        float log_del_sum = logf(curr_sum - prev_sum * expf((prev_max - curr_max) * softmax_scale)) + curr_max * softmax_scale;
        //if (thread(0, PRINT_BID)) printf("log_del_sum = %f curr_max = %f curr_sum = %f prev_max = %f prev_sum = %f\n", log_del_sum, curr_max, curr_sum, prev_max, prev_sum);
        prev_max = curr_max;
        prev_sum = curr_sum;

        float s = log_del_sum;

        int r = threadIdx.x % 4;
        float log_sum = logf(row_sum_tot(r)) + row_max(r) * softmax_scale;
        s = log_sum;
        //*/

        /*
        const int r = threadIdx.x % 4;
        float s = (r == 0) * (logf(row_sum_tot(0)) + row_max(0) * softmax_scale) +
        (r == 1) * (logf(row_sum_tot(1)) + row_max(1) * softmax_scale) +
        (r == 2) * (logf(row_sum_tot(2)) + row_max(2) * softmax_scale) +
        (r == 3) * (logf(row_sum_tot(3)) + row_max(3) * softmax_scale);
        */

        const int r = threadIdx.x % 4;
        float s = 
                (r == 0 ? logf(row_sum_tot(0)) + row_max(0) * softmax_scale : 0) +
                (r == 1 ? logf(row_sum_tot(1)) + row_max(1) * softmax_scale : 0) +
                (r == 2 ? logf(row_sum_tot(2)) + row_max(2) * softmax_scale : 0) +
                (r == 3 ? logf(row_sum_tot(3)) + row_max(3) * softmax_scale : 0);

        for (int th = 0; th < 8; th++) {continue; // comment out continue, to print data
            if (thread(th, PRINT_BID)) {
                int r = th % 4;
                float log_sum = logf(row_sum_tot(r)) + row_max(r) * softmax_scale;
                printf("th = %d, log_sum = %f\n", th, log_sum);
                //printf("th = %d, row_max = %f\n", th, row_max(th % 4));
            }
        }
        //*/

        /*if (thread(0, PRINT_BID) && curr_sum < 1.000001) {
            for(int mi = 0; mi < 4; ++mi) {
                printf("row_sum(%d) = %f, row_sum_tot(%d) = %f\n", mi, row_sum(mi), mi, row_sum_tot(mi));
            }
        }*/
        
        //if (thread(0, PRINT_BID)) printf("log_del_sum = %f\n", log_del_sum);
        
        //warp_weights[ww_count++] = s;
        //if (ww_count % 2 == 0) warp_weights[ww_count/2] = s;
        //ww_count++;

        //if (ww_count < 1) g_row_sum[thread_offset + ww_count] = s;
        if (ww_count < ww_size) g_row_sum[thread_offset + 32 * ww_count] = s;
        ww_count++;

    };//*/

    template<typename Tensor0>
    __device__ void analyze_scores(const Tensor0 &scores, float max){
        float score_sum = 0;
        for (int ni = 0; ni < size<1>(scores); ++ni) {
            score_sum += scores(0, ni);
        }
        
        float res4 = logf(score_sum) + max * softmax_scale;
        if (thread(0, PRINT_BID)) printf("res4 = %f\n", res4);
        //if (thread(0, PRINT_BID)) printf("ln(score_sum) = %f\n", logf(score_sum));
        /*if (thread(0, PRINT_BID) && ww_count < 5) {
            printf("scores: ");
            for (int ni = 0; ni < size<1>(scores); ++ni) printf("%f ", scores(0, ni));
            printf("\n");
        }*/
    };

    template<typename Tensor0, typename Tensor1>
    __device__ float analyze_sae2_helper (const Tensor0 &scores, const Tensor1 &row_max, const float ssl2){
        float scale = ssl2 * logf(2);
        auto scores_copy = make_fragment_like(scores);
        cute::copy(scores, scores_copy);
        FLASH_NAMESPACE::scale_apply_exp2(scores_copy, row_max, ssl2);
        float score_sum = 0;
        for (int ni = 0; ni < size<1>(scores); ++ni) score_sum += scores_copy(0, ni);
        float res = logf(score_sum) + row_max(0) * scale;
        //if (thread(0, PRINT_BID)) printf("ln(score_sum) helper = %f\n", logf(score_sum));
        /*if (thread(0, PRINT_BID) && ww_count < 5) {
            printf("scores check: ");
            for (int ni = 0; ni < size<1>(scores); ++ni) printf("%f ", expf((scores_copy(0, ni) - max) * scale));
            printf("\n");
        }*/
        return res;
    };

    template<typename Tensor0>
    __device__ float sae2_check (const Tensor0 &scores, const float max, const float ssl2){
        //auto scores_copy = make_fragment_like(scores);
        //cute::copy(scores, scores_copy);
        //FLASH_NAMESPACE::scale_apply_exp2(scores_copy, row_max, ssl2);
        float scale = ssl2 * logf(2);
        float score_sum = 0;
        for (int ni = 0; ni < size<1>(scores); ++ni) {
            float score = scores(0, ni);
            score_sum += expf((score - max) * scale);
        }
        float res = logf(score_sum) + max * scale;
        return res;
    };

    template<typename Tensor0, typename Tensor1>
    __device__ void analyze_scale_apply_exp2(const Tensor0 &scores, const Tensor1 &row_max, const float ssl2){
        if (!thread(0, PRINT_BID)) return;
        Tensor1 row_max0;
        for (int mi = 0; mi < 4; ++mi) row_max0(mi) = 0;
        float res1 = analyze_sae2_helper(scores, row_max, ssl2);
        float res2 = analyze_sae2_helper(scores, row_max0, ssl2);
        float res3 = sae2_check(scores, 0, ssl2);
        printf("res1 = %f, res2 = %f, res3 = %f\n", res1, res2, res3);
    };

};

#endif
        
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows>
struct Softmax {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;

    __forceinline__ __device__ Softmax() {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        if (Is_first) {
            FLASH_NAMESPACE::template reduce_max</*zero_init=*/true>(scores, row_max);
            FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            FLASH_NAMESPACE::reduce_sum</*zero_init=*/true>(scores, row_sum);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            FLASH_NAMESPACE::template reduce_max</*zero_init=*/false>(scores, row_max);
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
            }
            FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            FLASH_NAMESPACE::reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
    };

    template<bool Is_dropout=false, bool Split=false, typename Tensor0>
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0) {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
        }
        return lse;
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows, typename Kernel_traits>
struct Softmax_c : public Softmax<kNRows> {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;

    const int store_size;
    //const int store_size = 1;

    StochSparse_simple<kNRows, Kernel_traits> sss {store_size};

    __device__ Softmax_c(int store_size_): store_size(store_size_) {}

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    //__forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2, float* g_row_sum) {
        //if (thread(0, PRINT_BID)) printf("softmax_scale_log2 = %f\n", softmax_scale_log2);
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        if (Is_first) {
            FLASH_NAMESPACE::template reduce_max</*zero_init=*/true>(scores, row_max);
            FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            FLASH_NAMESPACE::reduce_sum</*zero_init=*/true>(scores, row_sum);
            //FLASH_NAMESPACE::reduce_sum_</*zero_init=*/true>(scores, row_sum);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            FLASH_NAMESPACE::template reduce_max</*zero_init=*/false>(scores, row_max);
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
            //if (thread(0, 1)) {printf("acc_o_rowcol:"); print(acc_o_rowcol); printf("\n"); printf("acc_o:"); print(acc_o); printf("\n");}
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;
                /*if (thread(0, PRINT_BID)) {
                    printf("c_bound = %f\n", (row_sum(mi) + 1));
                    if (mi == 3) printf("\n");
                }*/
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
                //log_row_sum[count + mi] = row_sum(mi); // exper
                //ms[count + mi] = row_max(mi); // exper
            }
            //sss.analyze_scale_apply_exp2(scores, row_max, softmax_scale_log2);
            FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            FLASH_NAMESPACE::reduce_sum</*zero_init=*/false>(scores, row_sum);
            //FLASH_NAMESPACE::reduce_sum_</*zero_init=*/false>(scores, row_sum);
            //auto dest_view = make_tensor(log_row_sum + count, make_layout(4));
            //cute::copy(row_sum, dest_view);
            //count = (count + mi_max) % store_size; // exper
        }
        /*
        //__syncthreads();
        //*/
        //sss.store_wws(row_max, row_sum);
        //sss.analyze_scores(scores, row_max(0));
        sss.store_wws(row_max, row_sum, g_row_sum); // uncomment for weight storing
        //sss.original_coordinates(acc_s);
    };

    __device__ void ss_final () {
    };

}; 

}  // namespace FLASH_NAMESPACE
