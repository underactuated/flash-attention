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

#define PRINT_BID 3 //1
#define VERBAL 0 //1

struct SSWeight {
    float score;
    float log_rand; // log of random float from [0, 1]
    char row;
    int col;

    //__device__ SSWeight (float score_, float log_rand_, char row_, int col_) : score(score_), log_rand(log_rand_), row(row_), col(col_) {};
};

template <int kNRows, typename Kernel_traits>
struct StochSparse {

    float c = 10; //1e-30; //10;

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_sum_tot;

    static constexpr int kBlockM = Kernel_traits::kBlockM;
    static constexpr int kBlockN = Kernel_traits::kBlockN;

    SSWeight ssweights [2 * kBlockN];
    int ssw_count = 0;

    curandState local_state;

private:

    __host__ __device__ static auto get_tcaccs() {
        // constexpr int kBlockM = Kernel_traits::kBlockM;
        // constexpr int kBlockN = Kernel_traits::kBlockN;
        const int tidx = threadIdx.x;
        typename Kernel_traits::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(tidx);
        Tensor caccs = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
        return thr_mma.partition_C(caccs);
    };

public:

    decltype(get_tcaccs()) tcaccs; // = get_tcaccs();

    __device__ StochSparse() {
        #if 0
        constexpr int kBlockM = Kernel_traits::kBlockM;
        constexpr int kBlockN = Kernel_traits::kBlockN;
        if (thread(0, PRINT_BID)) print("kBlockM = %d, kBlockN = %d\n", kBlockM, kBlockN);
        #endif
        tcaccs = get_tcaccs();
        int seed = 0;
        curand_init(seed + blockIdx.x * blockDim.x + threadIdx.x, 0, 0, &local_state);
    };

    template<typename Tensor0>
    __device__ void original_coordinates(Tensor0 &acc_s) { // todo: replace acc_s with scores
        #if 0
        if (thread(0, PRINT_BID)) {
        //if (thread(0, 1)) {
            print(acc_s);
            printf(" <-- acc_s\n");
            print(tcaccs);
            printf(" <-- tcaccs\n");
            
            auto accs_lo = FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout());
            print(accs_lo);
            printf(" <-- accs_lo\n");
            auto tcaccs_lo = FLASH_NAMESPACE::convert_layout_acc_rowcol(tcaccs.layout());
            print(tcaccs_lo);
            printf(" <-- tcaccs_lo\n");

            /*int count = 0, max_elements = 20;
            for (int i = 0; i < size(acc_s) && count < max_elements; ++i) {
                printf("[%d] = %f\n", i, float(acc_s(i)));
                //printf("[%d]: %f, %f\n", i, float(acc_s(i)), float(tcaccs));
                count++;
            }*/

            printf("size = %d\n", (int)size(acc_s));

            //for (int i = 0; i < 20; ++i) {
            for (int i = 0; i < size(acc_s); ++i) {
                auto coord = tcaccs(i);  // Get global coordinate upd: incorrect, use coord5
                auto value = acc_s(i);    // Get corresponding data value
                //print(rank(coord));
                printf("[%d] Global coord ", i);
                print(coord);
                auto coord1 = tcaccs_lo(i); printf(" coord1 "); print(coord1);
                //auto coord2 = accs_lo(i); printf(" coord2 "); print(coord2);
                auto coord3 = idx2crd(i, accs_lo.shape()); printf(" coord3 "); print(coord3);
                //auto coord4 = accs_lo(i%4, i/4); printf(" coord4 "); print(coord4);
                auto coord5 = tcaccs_lo(i%4, i/4); printf(" coord5 "); print(coord5); // correct global coordinates
                printf(" -> value: %f\n", (float)value);
                //printf("Global coord (%d,%d) -> value: %f\n", get<0>(coord), get<1>(coord), (float)value);
            }
        }
        
        /*if (thread0()) {
            for(int mi = 0; mi < size<0>(acc_s); ++mi) {
                for(int ni = 0; ni < size<1>(acc_s); ++ni) {
                    printf(" %d", acc_s(mi, ni));
                }
                printf("\n");
            }
        }*/
        #endif
    };

    template <typename Tensor0, typename Tensor1>
    __device__ void store_ssweights (Tensor0 &scores, Tensor1 &row_max, Tensor1 &row_sum) {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum_tot, row_sum, sum_op);
        //float c = 10;
        auto tcaccs_lo = FLASH_NAMESPACE::convert_layout_acc_rowcol(tcaccs.layout());
        #if 1
        #if VERBAL
        if (thread(0, PRINT_BID)) {
            float s = 0;
            for (int mi = 0; mi < 4; ++mi) {
                printf("row_sum_tot(%d) = %f\n", mi, row_sum_tot(mi));
                s += row_sum_tot(mi);
            }
            printf("s = %f\n", s);
            //for (int mi = 0; mi < 4; ++mi) printf("row_sum(%d) = %f\n", mi, row_sum(mi) * exp(row_max(mi)));
        }
        #endif
        #endif
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            //float scores_max_cur = !Check_inf
            //    ? row_max(mi)
            //    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
            //float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
            //row_sum(mi) *= scores_scale;
            //#pragma unroll

            float row_max_mi = row_max(mi);
            float row_sum_tot_mi = row_sum_tot(mi);
            //float row_sum_mi = row_sum(mi);
            // later, osorb c in row_sum_mi
            //if (thread(0, PRINT_BID)) printf("row_sum_mi = %f\n", row_sum_mi);
            
            for (int ni = 0; ni < size<1>(scores); ++ni) {
                //printf("mi = %d, ni = %d\n", mi, ni);
                if (ssw_count == 2 * kBlockN) continue;
                //float score = scores(mi, ni);
                //if (ssw_count > 0) continue; // experimental
                float score = scores(mi, ni);
                float rand_val = curand_uniform(&local_state);
                //float rand_val = score * score + .1; // was to test curand time 
                //if (score <= row_sum_mi * rand_val) continue;
                if (score * c <= row_sum_tot_mi * rand_val) continue;
                //if (score == 0) continue; // experim
                score = logf(score) + row_max_mi;
                float log_rand = logf(rand_val);
                char row = mi; // later, when moving to global memory, should be replaced with get<0>(coord)
                auto coord = tcaccs_lo(mi, ni);
                int col = get<1>(coord);
                #if VERBAL
                if (thread(0, PRINT_BID)) {
                    printf("score = %f, log_rand = %f, row = %d, col = %d\n", score, log_rand, row, col);
                }
                #endif
                //ssweights[ssw_count++] = SSWeight(score, log_rand, row, col);
                ssweights[ssw_count++] = SSWeight{score, log_rand, row, col};
            }
        }
        #if VERBAL
        if (thread(0, PRINT_BID)) printf("ssw_count = %d\n", ssw_count);
        #endif
    };

    template <typename Tensor1>
    __device__ void filter_ssweights (Tensor1 &row_max, Tensor1 &row_sum) {
        // to compare c*score/(row_sum*exp(row_max)) vs rand_val, we can compare:
        // log_c + log_score - log_row_sum - row_max vs log_rand, or
        // log_score - log_rand vs log_row_sum + row_max - log_c
        float log_c = logf(c);
        Tensor1 del_log;
        for (int i = 0; i < size<0>(del_log); ++i) {
            //del_log(i) = logf(row_sum(i)) + row_max(i) - log_c;
            del_log(i) = logf(row_sum_tot(i)) + row_max(i) - log_c;
        }
        int i = 0;
        for (int j = 0; j < ssw_count; ++j) {
            auto ssw = ssweights[j];
            // if weight should be skipped, continue
            if (ssw.score - ssw.log_rand < del_log(ssw.row)) continue;
            if (i < j) ssweights[i] = ssweights[j];
            i++;
        }
        #if 1
        if (thread(0, PRINT_BID)) printf("filtered out: %d\n", ssw_count - i);
        #endif
        ssw_count = i;
    };

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows>
struct Softmax {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;

    __forceinline__ __device__ Softmax() {};
    /*__forceinline__ __device__ Softmax() {
        my_print_test();
    }*/

    /*__device__ void my_print_test() {
        __shared__ int print_lock;
        if (threadIdx.x == 0) print_lock = 0;
        __syncthreads();
        //if (thread0()) {
        //    print(row_max);
        //}
        for (int i=0; i<30; i++) {
            //if (thread(0, i * i)) {
            if (thread(i * i, 0)) {
                bool printed = false;
                while (!printed) {
                    if (atomicCAS(&print_lock, 0, 1) == 0) {
                        print(row_max);
                        printf("block %d thread %d\n", blockIdx.x, threadIdx.x);
                        __threadfence();
                        print_lock = 0;
                        printed = true;
                    }
                    __nanosleep(100);
                }
            }
        }
    };*/

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

    StochSparse<kNRows, Kernel_traits> ss;

    __device__ Softmax_c() {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
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
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
            }
            FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            FLASH_NAMESPACE::reduce_sum</*zero_init=*/false>(scores, row_sum);
            //FLASH_NAMESPACE::reduce_sum_</*zero_init=*/false>(scores, row_sum);
        }
        ///*
        ss.original_coordinates(acc_s);
        ss.store_ssweights(scores, row_max, row_sum);
        ss.filter_ssweights(row_max, row_sum);
        //*/
    };

    /*template<bool Is_dropout=false, bool Split=false, typename Tensor0>
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0) {
        //ss.filter_ssweights(row_max, row_sum);
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
    };*/

}; 

}  // namespace FLASH_NAMESPACE
