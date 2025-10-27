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

#define PRINT_BID 30 //1
#define VERBAL 0 //1
#define VERBAL0 1

#define ssw_size 100 //500 //100 //30 //20

struct SSWeight {
    float score;
    /*float log_rand; // log of random float from [0, 1]
    //int row;
    char row;
    int col;
    //char col;*/
    __half log_rand;
    short row;
    
    //__device__ SSWeight (float score_, float log_rand_, char row_, int col_) : score(score_), log_rand(log_rand_), row(row_), col(col_) {};
};

struct SSWeight0 {
    float score;
    char log_rand;
    char row;
    short col;
};

/*__device__ __forceinline__ void printFloatBits(float f) {
    // Reinterpret the float as an unsigned int
    unsigned int u = *reinterpret_cast<unsigned int*>(&f);

    // Determine the number of bits in an unsigned int (typically 32)
    int numBits = std::numeric_limits<unsigned int>::digits; 

    // Print the bits from most significant to least significant
    for (int i = numBits - 1; i >= 0; --i) {
        // Check if the i-th bit is set
        if ((u >> i) & 1) {
            printf("1");
        } else {
            printf("0");
        }
    }
    printf("\n"); // Newline after printing all bits
}*/

__device__ __forceinline__ void printIntBits(int num) {
    int i;
    // Iterate from the most significant bit to the least significant bit
    for (i = (sizeof(int) * CHAR_BIT) - 1; i >= 0; i--) {
        // Check if the i-th bit is set
        if ((num >> i) & 1) {
            printf("1");
        } else {
            printf("0");
        }
    }
    printf("\n");
}

__device__ __forceinline__ float float2rand (const float& x, const int k = 16) {
  int z = 1 << k;
  int m = z - 1;
  const unsigned int lastkbits = *(reinterpret_cast<const unsigned int*>(&x)) & m;
  return (float) lastkbits / z;
}

template <int kNRows, typename Kernel_traits>
struct StochSparse {

    const float c = 10; //10; //50; //10; //1; //50; //20; //10; //5; //10; //1; //10; //1e-30; //10;
    float overc = 1. / c;

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_sum_tot;

    static constexpr int kBlockM = Kernel_traits::kBlockM;
    static constexpr int kBlockN = Kernel_traits::kBlockN;

    //static constexpr int ssw_size = 2 * kBlockN;

    //SSWeight ssweights [2 * kBlockN];
    SSWeight ssweights [ssw_size];
    SSWeight0 ssweights0 [ssw_size];
    int ssw_count = 0;

    curandState local_state;
    curandStatePhilox4_32_10_t state;

    float scores_subset[10];
    float row_maxs[600];
    int rmi = 0;

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
        //if (threadIdx.x == 0 && blockIdx.x == 0) {printf("MODIFIED VERSION RUNNING\n");} // temp
        if (thread0()) {printf("MODIFIED VERSION RUNNING\n");} // temp
        #if 0
        constexpr int kBlockM = Kernel_traits::kBlockM;
        constexpr int kBlockN = Kernel_traits::kBlockN;
        if (thread(0, PRINT_BID)) print("kBlockM = %d, kBlockN = %d\n", kBlockM, kBlockN);
        #endif
        tcaccs = get_tcaccs();
        int seed = 0;
        curand_init(seed + blockIdx.x * blockDim.x + threadIdx.x, 0, 0, &local_state);
        curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &state);
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

    /*__device__ void store_helper (char row, int col, float score, float rand_val, float row_max) {
        score = logf(score) + row_max;
        float log_rand = logf(rand_val);
        ssweights[ssw_count++] = SSWeight{score, log_rand, row, col};
    };*/

    /*
    template <typename Tensor0, typename array>
    __attribute__((cold)) __device__ void save_ssw(int mi, int ni, float row_max_mi, Tensor0 &scores, array &rand_vals) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j != mi) continue;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            float score = scores(j, i);
            if (i == ni) {
                float rand_val = rand_vals[ni % 8];
                score = logf(score) + row_max_mi;
                float log_rand = logf(rand_val);
                ssweights[ssw_count++] = SSWeight{score, log_rand, (short)mi};
                //ssweights[ssw_count++] = SSWeight{0, log_rand, short(mi)};
            }
        }
        }
    };//*/

    // old version of store ssweights
    // template <typename Tensor0, typename Tensor1>
    // __device__ void store_ssweights (Tensor0 &scores, Tensor1 &row_max, Tensor1 &row_sum) {
    //     SumOp<float> sum_op;
    //     quad_allreduce_(row_sum_tot, row_sum, sum_op);
    //     auto tcaccs_lo = FLASH_NAMESPACE::convert_layout_acc_rowcol(tcaccs.layout());
    //     #if 1
    //     #if VERBAL
    //     if (thread(0, PRINT_BID)) {
    //         float s = 0;
    //         for (int mi = 0; mi < 4; ++mi) {
    //             printf("row_sum_tot(%d) = %f\n", mi, row_sum_tot(mi));
    //             s += row_sum_tot(mi);
    //         }
    //         printf("s = %f\n", s);
    //         //for (int mi = 0; mi < 4; ++mi) printf("row_sum(%d) = %f\n", mi, row_sum(mi) * exp(row_max(mi)));
    //     }
    //     #endif
    //     #endif
    //     const int k = 4; //16; //8; //6; //1; //2; //4;
    //     /*float4 rand_vals0 = curand_uniform4(&state);
    //     float rand_vals[4] = {rand_vals0.x, rand_vals0.y, rand_vals0.z, rand_vals0.w};*/
    //     ///*
    //     float rand_vals[k];
    //     #pragma unroll
    //     for (int i = 0; i < k/4; ++i) {
    //         float4 rand_vals0 = curand_uniform4(&state);
    //         int i4 = i * 4;
    //         rand_vals[i4] = rand_vals0.x;
    //         rand_vals[i4 + 1] = rand_vals0.y;
    //         rand_vals[i4 + 2] = rand_vals0.z;
    //         rand_vals[i4 + 3] = rand_vals0.w;
    //     }//*/
    //     //return;
    //     //#if 0
    //     //float rand_vals[k] = {.5};
    //     //float rand_vals[k] = {.7586751, .3543543};
    //     //float rand_vals[k] = {.8586751, .6543543, .4565756, .2645365};
    //     //float rand_vals[k] = {.9547646, .8586751, .6543543, .4565756, .2645365, .13524};
    //     //float rand_vals[k] = {.8586751, .7546547, .6543543, .5675878, .4565756, .3564365, .2645365, .1342543};
    //     //float rand_vals[k] = {.8586751, .7546547, .6543543, .5675878, .4565756, .3564365, .2645365, .1342543, .05, .1, .2, .4, .5, .6, .7, .8};
    //     // float rand_vals[128];
    //     // //for (int i = 0; i < size<1>(scores); ++i) rand_vals[i] = .5;
    //     // for (int i = 0; i < 128; ++i) rand_vals[i] = .5;
    //     //float drv = c * overc - 1.;
    //     //float drv = 0;
    //     #pragma unroll
    //     for (int mi = 0; mi < size<0>(scores); ++mi) {
    //         //float scores_max_cur = !Check_inf
    //         //    ? row_max(mi)
    //         //    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
    //         //float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
    //         //row_sum(mi) *= scores_scale;
    //         //#pragma unroll

    //         float row_max_mi = row_max(mi);
    //         //float row_sum_tot_mi = row_sum_tot(mi);
    //         float row_sum_tot_mi_oc = row_sum_tot(mi) * overc;
    //         //float row_sum_mi = row_sum(mi);
    //         // later, osorb c in row_sum_mi
    //         //if (thread(0, PRINT_BID)) printf("row_sum_mi = %f\n", row_sum_mi);

    //         /*for (int ni = 0; ni < size<1>(scores); ++ni) {
    //             //printf("mi = %d, ni = %d\n", mi, ni);
    //             if (ssw_count == 2 * kBlockN) continue;
    //             //float score = scores(mi, ni);
    //             //if (ssw_count > 0) continue; // experimental
    //             float score = scores(mi, ni);
    //             //float score = 0.0001; //1./(512 * 32); // experim
    //             float rand_val = curand_uniform(&local_state);
    //             //float rand_val = score * score + .1; // was to test curand time 
    //             //if (score <= row_sum_mi * rand_val) continue;
    //             if (score * c <= row_sum_tot_mi * rand_val) continue;
    //             //if (score == 0) continue; // experim
    //             score = logf(score) + row_max_mi;
    //             float log_rand = logf(rand_val);
    //             char row = mi; // later, when moving to global memory, should be replaced with get<0>(coord)
    //             auto coord = tcaccs_lo(mi, ni);
    //             int col = get<1>(coord);
    //             #if VERBAL
    //             if (thread(0, PRINT_BID)) {
    //                 printf("score = %f, log_rand = %f, row = %d, col = %d\n", score, log_rand, row, col);
    //             }
    //             #endif
    //             //ssweights[ssw_count++] = SSWeight(score, log_rand, row, col);
    //             ssweights[ssw_count++] = SSWeight{score, log_rand, row, col};
    //         }*/
            
    //         #if 0 //1
    //         // experimental block
    //         // slowdown seems to be caused primarily by curand_uniform(), and to lesser degree by logf(),
    //         // not by scores(mi,ni) or conditional
    //         //float p = 0;
    //         //#pragma unroll
    //         for (int ni = 0; ni < size<1>(scores); ++ni) {
    //             //printf("mi = %d, ni = %d\n", mi, ni);
    //             if (ssw_count == 2 * kBlockN) continue;
    //             //if (ssw_count == ssw_size) continue;
    //             float score = scores(mi, ni);

    //             /*//score = logf(score) + row_max_mi;
    //             float rand_val1 = curand_uniform(&local_state);
    //             //float rand_val1 = .5;
    //             //float log_rand1 = logf(rand_val1);
    //             score += rand_val1;
    //             //score += logf(score);
    //             p += score; continue;*/

    //             /*score = logf(score) + row_max_mi;
    //             float rand_val1 = curand_uniform(&local_state);
    //             float log_rand1 = logf(rand_val1);
    //             score += log_rand1;
    //             p += score; continue;*/
                
    //             //float rand_val = curand_uniform(&local_state);
    //             float rand_val = .5;
    //             //float rand_val = .25;
    //             //float rand_val = (ni % 2) ? .5 : .1;
    //             //rand_val += drv;
    //             //float rand_val = (float) ((ni % 10) + 1) / 10; // works much worse than rand_val=.5 or even .1 ???
    //             //float rand_val = .1;
    //             //rand_val += (rand_val > .91) ? -.9 : .1;
    //             //float rand_val = float2rand(score);
    //             //float dlr = -logf((ni % 2) + 1);
    //             if (score * c <= row_sum_tot_mi * rand_val) continue;
    //             //if (score * c * ((ni % 2) + 1) <= row_sum_tot_mi * rand_val) continue;
    //             //if (score * c <= row_sum_tot_mi * rand_val * ((ni % 2) + 1)) continue;
    //             //if (score <= row_sum_tot_mi_oc * rand_val) continue;
    //             //if (thread(0, PRINT_BID)) printFloatBits(score);
    //             //if (thread(0, PRINT_BID)) printf("rand val: %f\n", rand_val);
    //             //int dcount = 1;
    //             //int dcount = !(score * c <= row_sum_tot_mi * rand_val);
    //             score = logf(score) + row_max_mi;
    //             float log_rand = logf(rand_val);
    //             //log_rand -= logf((ni % 2) + 1);
    //             //p += score; continue; // experim
    //             char row = mi; // later, when moving to global memory, should be replaced with get<0>(coord)
    //             //int row = mi;
    //             auto coord = tcaccs_lo(mi, ni);
    //             int col = get<1>(coord);
    //             //char col = get<1>(coord);
    //             //ssweights[ssw_count++] = SSWeight(score, log_rand, row, col);
    //             ssweights[ssw_count++] = SSWeight{score, log_rand, row};
    //             //ssw_count += dcount;
    //         }
    //         //ssweights[ssw_count++] = SSWeight{p, p, 0, 0};
    //         #endif
    //         #if 1 //0
    //         // experimental block
    //         // slowdown seems to be caused primarily by curand_uniform(), and to lesser degree by logf(),
    //         // not by scores(mi,ni) or conditional
    //         //float rand_vals[4] = {.8, .6, .4, .2};
    //         //float4 rand_vals0 = curand_uniform4(&state);
    //         //float rand_vals[4] = {rand_vals0.x, rand_vals0.y, rand_vals0.z, rand_vals0.w};
    //         //float p = 0;
    //         float selected_scores[32];
    //         int iss = 0;
    //         int m = 0;
    //         int ki_max = size<1>(scores) / k;
    //         #pragma unroll
    //         for (int ki = 0; ki < ki_max; ++ki) {
    //             //if (ssw_count > 2 * kBlockN - k) continue;
    //             //if (ssw_count >= ssw_size - 32) continue;
    //             if (ssw_count >= ssw_size - k) continue;
    //             #pragma unroll
    //             for (int i = 0; i < k; ++i) {
    //                 float rand_val = rand_vals[i];
    //                 int ni = k * ki + i;
    //                 float score = scores(mi, ni);
    //                 //p += (row_sum_tot_mi_oc * rand_val - score); continue;
    //                 //if (score * c > row_sum_tot_mi * rand_val) {
    //                 if (score > row_sum_tot_mi_oc * rand_val) {
    //                     //ssweights[ssw_count++] = SSWeight{.1, .1, 0}; continue;
    //                     selected_scores[iss++] = score;
    //                     m += (1 << ni); continue;
    //                     score = logf(score) + row_max_mi;
    //                     float log_rand = logf(rand_val);
    //                     char row = mi; // later, when moving to global memory, should be replaced with get<0>(coord)
    //                     auto coord = tcaccs_lo(mi, ni);
    //                     int col = get<1>(coord);
    //                     //ssweights[ssw_count++] = SSWeight{score, log_rand, row, col};
    //                     ssweights[ssw_count++] = SSWeight{score, log_rand, row};
    //                 }
    //             }
    //         }
    //         //continue;
    //         //ssweights[ssw_count++] = SSWeight{p, p, 0};
    //         int ni = 0;
    //         iss = 0;
    //         while (1) {
    //             int i = __builtin_ffs(m);
    //             //printf("i = %d\n", i);
    //             //if (i == 0) break;
    //             //if (ssw_count > 255) ssw_count = 0;
    //             if (i == 0 || ssw_count >= ssw_size) break;
    //             ni += i;
    //             /*
    //             float score = scores(mi, ni);
    //             float rand_val = rand_vals[ni % k];
    //             score = logf(score) + row_max_mi;
    //             float log_rand = logf(rand_val);
    //             ssweights[ssw_count++] = SSWeight{score, log_rand, (short)mi};
    //             //*/
    //             float score = selected_scores[iss++];
    //             score = logf(score) + row_max_mi;
    //             float rand_val = rand_vals[ni % k];
    //             //float rand_val = .5; // temp
    //             float log_rand = logf(rand_val);
    //             ssweights[ssw_count++] = SSWeight{score, log_rand, (short)mi};
    //             //ssweights[ssw_count++] = SSWeight{score, .1, 0};
    //             //ssweights[ssw_count++] = SSWeight{.1, .1, 0}; //SSWeight{(float)i, (float)i, 0};
    //             //save_ssw(mi, i, row_max_mi, scores, rand_vals);
    //             //if (thread(0, PRINT_BID)) printf("ssw_count = %d\n", ssw_count);
    //             m >>= i;
    //         }
    //         /*while (m != 0) {
    //             //while ((m & 1) == 0 && i < 32) {
    //             while ((m & 1) == 0 && m != 0) {
    //                 m >>= 1;
    //                 i++;
    //             }
    //             if (m == 0) break;
    //             ssweights[ssw_count++] = SSWeight{(float)i, (float)i, 0};
    //             m >>= 1;
    //             i++;
    //         }*/
    //         /*
    //         int i = 0;
    //         while (i < 32) {
    //             while ((m & 1) == 0 && i < 32) {
    //                 m >>= 1;
    //                 i++;
    //             }
    //             if (i == 32) break;
    //             ssweights[ssw_count++] = SSWeight{(float)i, (float)i, 0};
    //             m >>= 1;
    //             i++;
    //         }*/
    //         //ssweights[ssw_count++] = SSWeight{(float)m, (float)m, 0};
    //         #endif
    //     }
    //     #if VERBAL
    //     if (thread(0, PRINT_BID)) printf("ssw_count = %d\n", ssw_count);
    //     #endif
    //     //if (thread(0, PRINT_BID)) printf("last ssweight = %f\n", ssweights[ssw_count-1].score);
    //     //#endif
    // };

    template <typename Tensor0, typename Tensor1>
    __device__ void store_ssweights_test0 (Tensor0 &scores, Tensor1 &row_max, Tensor1 &row_sum) {
        constexpr int mi_max = decltype(size<0>(scores))::value;
        constexpr int ni_max = decltype(size<1>(scores))::value;
        static_assert(decltype(size<0>(scores))::value == mi_max);
        //static_assert(ni_max == 16 || ni_max == 32);
        SumOp<float> sum_op;
        quad_allreduce_(row_sum_tot, row_sum, sum_op);
        auto tcaccs_lo = FLASH_NAMESPACE::convert_layout_acc_rowcol(tcaccs.layout());
        unsigned int m[mi_max] = {};
        unsigned int rand_count = ni_max;
        #pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            //float row_max_mi = row_max(mi);
            float row_sum_tot_mi_oc = row_sum_tot(mi) * overc;
            #pragma unroll
            for (int ni = 0; ni < size<1>(scores); ++ni) {
                float score = scores(mi, ni);
                int bit_pos = __builtin_ffs(rand_count);
                unsigned int over_rand_val = 1 << bit_pos;
                m[mi] += (score * over_rand_val > row_sum_tot_mi_oc)? 1 << ni : 0;
                rand_count++;
            }
        }

        //if (thread(0, PRINT_BID)) printf("rand_count = %d\n", rand_count);

        // #if 0
        // if (thread(0, PRINT_BID)) {
        //     printf("ms: %u %u %u %u\n", m[0], m[1], m[2], m[3]);
        //     for (int i = 0; i < mi_max; i++) printIntBits(m[i]);
        // }
        // #endif

        #if 0 // USE THIS BLOCK FOR H200 (incomplete)
        unsigned int rand_count0 = rand_count - mi_max * ni_max;
        #pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            //float row_max_mi = row_max(mi);
            unsigned int bits = m[mi];
            int ni = -1;
            while (bits && ssw_count < ssw_size) {
                int i = __builtin_ffs(bits);
                ni += i;
                bits >>= i;
                float score = get_score_predicated_mi(scores, mi, ni);
                //float score = get_score_predicated1(scores, mi, ni);
                //float score = get_score_predicated2(scores, row_max, mi, ni);
                //score = logf(score) + row_max_mi;
                char log_rand = __builtin_ffs(rand_count0 + ni);
                auto coord = tcaccs_lo(mi, ni);
                short col = get<1>(coord);
                ssweights0[ssw_count++] = SSWeight0{score, log_rand, (char)mi, col};
            }
            //if (thread(0, PRINT_BID)) printf("ssi = %d\n", ssi);
                //printf("ssi = %d score_sum = %f\n", ssi, score_sum);
            rand_count0 += ni_max;
        }
        #endif

        #if 1 // USE THIS BLOCK FOR A10
        float selected_scores[ni_max];
        unsigned int rand_count0 = rand_count - mi_max * ni_max; 
        #pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            unsigned int bits = m[mi];
            int ssi = 0;
            int ni = -1;
            while (bits && ssw_count < ssw_size) {
                int i = __builtin_ffs(bits);
                ni += i;
                bits >>= i;
                //float score = get_score_predicated_mi(scores, mi, ni);
                //float score = get_score_predicated1(scores, mi, ni);
                selected_scores[ssi++] = get_score_predicated_mi(scores, mi, ni);
            }
            float row_max_mi = row_max(mi);
            bits = m[mi];
            ni = -1;
            for (int j = 0; j < ssi; j++) {
                int i = __builtin_ffs(bits);
                ni += i;
                bits >>= i;
                float score = selected_scores[j];
                score = logf(score) + row_max_mi;
                char log_rand = __builtin_ffs(rand_count0 + ni);
                auto coord = tcaccs_lo(mi, ni);
                short col = get<1>(coord);
                ssweights0[ssw_count++] = SSWeight0{score, log_rand, (char)mi, col};
            }
            //if (thread(0, PRINT_BID)) printf("ssi = %d\n", ssi);
                //printf("ssi = %d score_sum = %f\n", ssi, score_sum);
            rand_count0 += ni_max;
        }
        #endif

        #if 0 // experimental
        int ssw_count0 = ssw_count;
        unsigned int rand_count0 = rand_count - mi_max * ni_max;
        #pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            //float row_max_mi = row_max(mi);
            unsigned int bits = m[mi];
            int ni = -1;
            while (bits && ssw_count < ssw_size) {
                int i = __builtin_ffs(bits);
                ni += i;
                bits >>= i;
                float score = get_score_predicated_mi(scores, mi, ni);
                //float score = get_score_predicated1(scores, mi, ni);
                //float score = get_score_predicated2(scores, row_max, mi, ni);
                //score = logf(score) + row_max_mi;
                score = logf(score);
                char log_rand = __builtin_ffs(rand_count0 + ni);
                auto coord = tcaccs_lo(mi, ni);
                short col = get<1>(coord);
                ssweights0[ssw_count++] = SSWeight0{score, log_rand, (char)mi, col};
            }
            //if (thread(0, PRINT_BID)) printf("ssi = %d\n", ssi);
                //printf("ssi = %d score_sum = %f\n", ssi, score_sum);
            rand_count0 += ni_max;
        }
        #if 0
        // this is needed if we postpone adding row_max to logf(score) 
        if (ssw_count != ssw_count0) {
            #pragma unroll
            for (int mi = 0; mi < mi_max; ++mi) row_maxs[rmi + mi] = row_max(mi);
            rmi += mi_max;
        }
        #endif
        #endif

        #if VERBAL
        if (thread(0, PRINT_BID)) {
            printf("ms: %u %u %u %u\n", m[0], m[1], m[2], m[3]);
            for (int i = 0; i < mi_max; i++) printIntBits(m[i]);
        }
        #endif    
    };

    /*template <typename Tensor0>
    __device__ float get_score_predicated (Tensor0 &scores, int target) {
        constexpr int mi_max = decltype(size<0>(scores))::value;
        constexpr int ni_max = decltype(size<1>(scores))::value;
        float result = 0.0f;
        #pragma unroll
        for (int m = 0; m < mi_max; m++) {
            #pragma unroll
            for (int n = 0; n < ni_max; n++) {
                result = (ni_max * m + n == target) ? scores(m, n) : result;
            }
        }
        return result;
    };*/

    template <typename Tensor0>
    __device__ float get_score_predicated_mi (Tensor0 &scores, int mi, int ni) {
        constexpr int ni_max = decltype(size<1>(scores))::value;
        float result = 0.0f;
        #pragma unroll
        for (int n = 0; n < ni_max; n++) {
            result = (n == ni) ? scores(mi, n) : result;
        }
        return result;
        //return logf(result);
    };

    /*
    template <int mi, typename Tensor0>
    __device__ float get_score_predicated_mi1 (Tensor0 &scores, int ni) {
        float result = 0.0f;
        #pragma unroll
        for (int n = 0; n < 32; n++) {
            result = (n == ni) ? scores(mi, n) : result;
        }
        return result;
        //return logf(result);// + row_max(mi);
    };

    template <typename Tensor0>
    __device__ float get_score_predicated1 (Tensor0 &scores, int mi, int ni) {
        switch (mi) {
            case 0: return get_score_predicated_mi1<0>(scores, ni);
            case 1: return get_score_predicated_mi1<1>(scores, ni);
            case 2: return get_score_predicated_mi1<2>(scores, ni);
            case 3: return get_score_predicated_mi1<3>(scores, ni);
        }
        return 0;
    };//*/

    /*template <typename Tensor0, typename Tensor1>
    __device__ float get_score_predicated2 (Tensor0 &scores, Tensor1 &row_max, int mi, int ni) {
        switch (mi) {
            case 0: return get_score_predicated_mi1<0>(scores, ni) + row_max(0);
            case 1: return get_score_predicated_mi1<1>(scores, ni) + row_max(1);
            case 2: return get_score_predicated_mi1<2>(scores, ni) + row_max(2);
            case 3: return get_score_predicated_mi1<3>(scores, ni) + row_max(3);
        }
        return 0;
    };*/

    // template <typename Tensor0, typename Tensor1>
    // __device__ void store_ssweights_test (Tensor0 &scores, Tensor1 &row_max, Tensor1 &row_sum) {
    //     float p = 0;
    //     //int m = 0;
    //     float selected[32 * 4];
        
    //     /*//__shared__ float shared_selected[128 * 32];
    //     __shared__ int shared_count;
    //     int tid = threadIdx.x;
    //     //int bid = blockIdx.x;
    //     int count = 0;
    //     if (tid == 0) shared_count = 0;//*/
    //     __syncthreads();

    //     /*int start_pos = atomicAdd(&shared_count, local_matches);
    //     for (int i = 0; i < local_matches && start_pos + i < 512; i++) {
    //         shared_indices[start_pos + i] = local_indices[i];*/
            
    //     //#pragma unroll
    //     for (int i = 0; i < 32 * 4; ++i) selected[i] = 0.0f;
    //     //int si = 0;
    //     //#pragma unroll
    //     for (int mi = 0; mi < size<0>(scores); ++mi) {
    //         /*#pragma unroll
    //         for (int ni = 0; ni < size<1>(scores); ++ni) {//{ p += scores(mi, ni); } //{ p += logf(scores(mi, ni)); }
    //             float score = scores(mi, ni);
    //             //score += curand_uniform(&local_state);
    //             //p += score;
    //             //if (p < score) p += score;
    //             p += (p < score)? score : 0;
    //         }*/
    //         //float selected[32];
    //         //int si = 0;
    //         //#pragma unroll
    //         for (int i = 0; i < size<1>(scores)/4; ++i) {
    //             //float4 rand_vals = curand_uniform4(&state);
    //             //float rand_val = curand_uniform(&local_state);
    //             //#pragma unroll
    //             for (int j = 0; j < 4; ++j) {
    //                 int ni = 4 * i + j;
    //                 //for (int ni = 0; ni < size<1>(scores); ++ni) {//{ p += scores(mi, ni); } //{ p += logf(scores(mi, ni)); }
    //                 float score = scores(mi, ni);
    //                 //score += curand_uniform(&local_state);
    //                 //float4 rand_vals = curand_uniform4(&state);
    //                 //score += rand_vals.x;
    //                 //score += rand_val;
    //                 /*switch (j) {
    //                     case 0: score += rand_vals.x; break;
    //                     case 1: score += rand_vals.y; break;
    //                     case 2: score += rand_vals.z; break;
    //                     case 3: score += rand_vals.w; break;
    //                 }*/
    //                 //p += score;
    //                 //if (p < score) p += score;
    //                 //m += (p < score) * (1 << ni);
    //                 //m += (p < score)? 1 << ni : 0;
    //                 //m += (p > score)? 0 : 1 << ni;
    //                 //if (p < score) m += (1 << ni);
    //                 //p += (p < score)? score : 0;
    //                 //selected[ni] = p;
    //                 //selected[ni] += (p < score)? score : 0;
    //                 //selected[ni] += (score > 2)? score : 0;
    //                 //selected[ni] += (score < .1)? score : 0;
    //                 //if (score > .1) selected[ni] += score;
    //                 //selected[ni] = (score < 1e-6)? score : 0;
    //                 p += score; // / 1024;
    //                 if (score > .1) {
    //                     selected[32 * mi + ni] += score;
    //                     /*
    //                     if (count < 1) {
    //                         count = atomicAdd(&shared_count, 1);
    //                         //shared_selected[ind] = score;
    //                     }//*/
    //                     //break;
    //                 }
    //                 /*if (score > .1) {
    //                     switch (si) {
    //                         case (0): selected[0] = score; break;
    //                         case (1): selected[1] = score; break;
    //                         case (2): selected[2] = score; break;
    //                         case (3): selected[3] = score; break;
    //                     }
    //                     si++;
    //                 }*/
    //             }
    //             //if (p < 1 && ssw_count != 5) ssweights[ssw_count++] = SSWeight{p, logf(p), 0};
    //             //if (p < 1 && si != 10 && p > .09) selected[si++] = p;
    //             //selected[si += (p < 1 && si != 10)] = p;
    //         }
    //         //if (p < 5 && ssw_count != 5) ssweights[ssw_count++] = SSWeight{p, logf(p), 0};
    //         //if (p < 3 && ssw_count != 1) ssweights[ssw_count++] = SSWeight{selected[mi], logf(p), 0};
    //     }

    //     float f = 3;
    //     int lmax = (int) (f * curand_uniform(&local_state));
    //     //int lmax = 2;
    //     #pragma unroll
    //     for (int l = 0; l < lmax; l++) {
    //         int target = (int) (128 * curand_uniform(&local_state));
    //         scores_subset[l] = get_score_predicated(scores, target);
    //     }

    //     /* // 158 ms
    //     float f = 3;
    //     int lmax = (int) (f * curand_uniform(&local_state));
    //     #pragma unroll
    //     for (int l = 0; l < lmax; l++) {
    //         int target = (int) (128 * curand_uniform(&local_state));
    //         float result = 0.0f;            
    //         #pragma unroll
    //         for (int m = 0; m < 4; m++) {
    //             #pragma unroll
    //             for (int n = 0; n < 32; n++) {
    //                 int current_flat = (m << __builtin_ctz(32)) | n;
    //                 result = (current_flat == target) ? scores(m, n) : result;
    //             }
    //         }
    //         scores_subset[l] = result;
    //     }*/

    //     /* // 158 ms
    //     float f = 3;
    //     int lmax = (int) (f * curand_uniform(&local_state));
    //     #pragma unroll
    //     for (int l = 0; l < lmax; l++) {
    //         int target = (int) (128 * curand_uniform(&local_state));
    //         float result = 0.0f;
    //         #pragma unroll
    //         for (int m = 0; m < 4; m++) {
    //             #pragma unroll
    //             for (int n = 0; n < 32; n++) {
    //                 result = (32 * m + n == target) ? scores(m, n) : result;
    //             }
    //         }
    //         scores_subset[l] = result;
    //     }*/

    //     /* // 150 ms
    //     float f = 3;
    //     int lmax = (int) (f * curand_uniform(&local_state));
    //     #pragma unroll
    //     for (int l = 0; l < lmax; l++) {
    //         int target = (int) (128 * curand_uniform(&local_state));
    //         #pragma unroll
    //         for (int m = 0; m < 4; m++) {
    //             #pragma unroll
    //             for (int n = 0; n < 32; n++) {
    //                 if (32 * m + n == target) scores_subset[l] = scores(m, n);
    //             }
    //         }
    //     }*/

    //     for (int l=0; l<lmax; l++){
    //         //if (ssw_count != 5 && p > 1) ssweights[ssw_count++] = SSWeight{scores_subset[l], logf(p), 0};
    //         if (ssw_count != 5 && p < 2) ssweights[ssw_count++] = SSWeight{scores_subset[ssw_count], logf(p), 0};
    //     }

    //     /*//#pragma unroll
    //     //for (int l = 0; l < 1; ++l) {
    //         int ind = (int) (10 * curand_uniform(&local_state));
    //         int ind1 = (int) (10 * curand_uniform(&local_state));
    //         float sel = 0, sel1 = 0;
    //         for (int mi = 0; mi < size<0>(scores); ++mi) {
    //             for (int i = 0; i < size<1>(scores)/4; ++i) {
    //                 for (int j = 0; j < 4; ++j) {
    //                     int ni = 4 * i + j;
    //                     float score = scores(mi, ni);
    //                     //p += score; // / 1024;
    //                     if (score > .5) {
    //                         selected[32 * mi + ni] += score;
    //                         p += score;
    //                     }
    //                     sel = (ni == ind)? score : sel;
    //                     sel1 = (ni == ind1)? score : sel1;
    //                     //sel = (ni == ind)? selected[32 * mi + ni] : sel;
    //                 }
    //             }
    //         }
    //         p = p + sel;// + selected[5];
    //         if (ssw_count != 5) ssweights[ssw_count++] = SSWeight{sel, logf(p), 0};
    //     //}*/

    //     /*ind = (int) (10 * curand_uniform(&local_state));
    //     sel = 0;
    //     for (int mi = 0; mi < size<0>(scores); ++mi) {
    //         for (int i = 0; i < size<1>(scores)/4; ++i) {
    //             for (int j = 0; j < 4; ++j) {
    //                 int ni = 4 * i + j;
    //                 float score = scores(mi, ni);
    //                 //p += score; // / 1024;
    //                 if (score > .5) {
    //                     selected[32 * mi + ni] += score;
    //                     p += score;
    //                 }
    //                 sel = (ni == ind)? score : sel;
    //             }
    //         }
    //     }
    //     p = p + sel + selected[5];
    //     if (ssw_count != 5) ssweights[ssw_count++] = SSWeight{sel, logf(p), 0};*/
        
    //     //if (p != 0 && ssw_count == 0) ssweights[ssw_count++] = SSWeight{p, p, 0};
    //     //if (ssw_count != 50 && p != 0) ssweights[ssw_count++] = SSWeight{p, logf(p), 0};
    //     //if (ssw_count != 50 && p > 1) ssweights[ssw_count++] = SSWeight{p, logf(p), 0};
    //     //if (ssw_count != 5 && p > 1) ssweights[ssw_count++] = SSWeight{p, logf(p), 0};
    //     //if (ssw_count != 5 && p > 1) ssweights[ssw_count++] = SSWeight{selected[ssw_count], logf(p), 0};
    //     //if (ssw_count != 5 && p > 1) ssweights[ssw_count++] = SSWeight{selected[5], logf(p), 0};
    //     if (ssw_count != 5 && p > 1e-5) ssweights[ssw_count++] = SSWeight{selected[5], logf(p), 0};
    //     //if (ssw_count != 5 && p > 1e-5) ssweights[ssw_count++] = SSWeight{shared_selected[5], logf(p), 0};
    //     //if (ssw_count != 5 && p > 1) ssweights[ssw_count++] = SSWeight{scores(0, ssw_count), logf(p), 0};
    //     //if (thread(0, PRINT_BID)) printf("m = %d\n", m);
    //     /*SumOp<float> sum_op;
    //     quad_allreduce_(row_sum_tot, row_sum, sum_op);
    //     auto tcaccs_lo = FLASH_NAMESPACE::convert_layout_acc_rowcol(tcaccs.layout());*/
    //     //const int k = 4; //16; //8; //6; //1; //2; //4;
    //     /*float4 rand_vals0 = curand_uniform4(&state);
    //     float rand_vals[4] = {rand_vals0.x, rand_vals0.y, rand_vals0.z, rand_vals0.w};*/
    //     ///*
    //     /*float rand_vals[k];
    //     #pragma unroll
    //     for (int i = 0; i < k/4; ++i) {
    //         float4 rand_vals0 = curand_uniform4(&state);
    //         int i4 = i * 4;
    //         rand_vals[i4] = rand_vals0.x;
    //         rand_vals[i4 + 1] = rand_vals0.y;
    //         rand_vals[i4 + 2] = rand_vals0.z;
    //         rand_vals[i4 + 3] = rand_vals0.w;
    //     }//*/
    //     //return;
    //     #if 0
    //     #pragma unroll
    //     //for (int mi = 0; mi < size<0>(scores); ++mi) {
    //     for (int mi = 0; mi < 4; ++mi) {
    //         //float row_max_mi = row_max(mi);
    //         //float row_sum_tot_mi_oc = row_sum_tot(mi) * overc;

    //         float p = 0;
    //         /*float selected_scores[32];
    //         int iss = 0;
    //         int m = 0;*/
    //         int ki_max = size<1>(scores) / k;
    //         #pragma unroll
    //         for (int ki = 0; ki < ki_max; ++ki) {
    //             //if (ssw_count >= ssw_size - k) continue;
    //             #pragma unroll
    //             for (int i = 0; i < k; ++i) {
    //                 //float rand_val = rand_vals[i];
    //                 int ni = k * ki + i;
    //                 float score = scores(mi, ni);
    //                 p += logf(score);
    //                 //if (score > row_sum_tot_mi_oc * rand_val) {
    //                 //    p += logf(score);
    //                     /*//ssweights[ssw_count++] = SSWeight{.1, .1, 0}; continue;
    //                     selected_scores[iss++] = score;
    //                     m += (1 << ni); continue;
    //                     score = logf(score) + row_max_mi;
    //                     float log_rand = logf(rand_val);
    //                     char row = mi; // later, when moving to global memory, should be replaced with get<0>(coord)
    //                     auto coord = tcaccs_lo(mi, ni);
    //                     int col = get<1>(coord);
    //                     //ssweights[ssw_count++] = SSWeight{score, log_rand, row, col};
    //                     ssweights[ssw_count++] = SSWeight{score, log_rand, row};*/
    //                 //}
    //             }
    //         }
    //         if (p != 0 && ssw_count == 0) ssweights[ssw_count++] = SSWeight{p, p, 0};
    //     }
    //     #endif
    // };

    template <typename Tensor1>
    __device__ void filter_ssweights (Tensor1 &row_max, Tensor1 &row_sum) {
        #if VERBAL0
        if (thread(0, PRINT_BID)) {
            printf("ssw_count: %d\n", ssw_count);
            //if (ssw_count) printf("%f\n", (float)ssweights[ssw_count-1].score);
            if (ssw_count) printf("%f\n", (float)ssweights0[ssw_count-1].score);
        }
        #endif
        //if (ssw_count + 10 < ssw_size) return;
        return;
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
            if (ssw.score - (float)ssw.log_rand < del_log(ssw.row)) continue;
            if (i < j) ssweights[i] = ssweights[j];
            i++;
        }
        #if VERBAL0
        if (thread(0, PRINT_BID)) printf("filtered out: %d\n", ssw_count - i);
        #endif
        ssw_count = i;
    };

    template <typename Tensor1>
    __device__ void filter_ssweights0 (Tensor1 &row_max, Tensor1 &row_sum) {
        #if VERBAL0
        if (thread(0, PRINT_BID)) {
            printf("ssw_count: %d\n", ssw_count);
            if (ssw_count) printf("%f\n", (float)ssweights0[ssw_count-1].score);
        }
        #endif
        //return;
        // to compare c*score/(row_sum*exp(row_max)) vs rand_val, we can compare:
        // log_c + log_score - log_row_sum - row_max vs log_rand, or
        // log_score - log_rand vs log_row_sum + row_max - log_c
        float log_c = logf(c);
        float log_2 = logf(2);
        Tensor1 del_log;
        for (int i = 0; i < size<0>(del_log); ++i) {
            //del_log(i) = logf(row_sum(i)) + row_max(i) - log_c;
            del_log(i) = logf(row_sum_tot(i)) + row_max(i) - log_c;
        }
        int i = 0;
        for (int j = 0; j < ssw_count; ++j) {
            auto ssw = ssweights0[j];
            //if (thread(0, PRINT_BID)) printf("score  = %f log_rand = %f del_log = %f\n", ssw.score,(float)ssw.log_rand, del_log(ssw.row));
            // if weight should be skipped, continue
            if (ssw.score + (float)ssw.log_rand - log_2 < del_log(ssw.row)) continue;
            //if (row_maxs[j] + ssw.score + (float)ssw.log_rand - log_2 < del_log(ssw.row)) continue; // undo
            if (i < j) ssweights0[i] = ssweights0[j];
            i++;
        }
        #if VERBAL0
        if (thread(0, PRINT_BID)) printf("filtered out: %d\n", ssw_count - i);
        #endif
        ssw_count = i;
    };

};

////////////////////////////////////////////////////////////////////////////////////////////////////
/*struct StochSparse {

    const float c = 10; //10; //50; //10; //1; //50; //20; //10; //5; //10; //1; //10; //1e-30; //10;
    float overc = 1. / c;

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_sum_tot;

    static constexpr int kBlockM = Kernel_traits::kBlockM;
    static constexpr int kBlockN = Kernel_traits::kBlockN;

    //static constexpr int ssw_size = 2 * kBlockN;

    //SSWeight ssweights [2 * kBlockN];
    SSWeight ssweights [ssw_size];
    SSWeight0 ssweights0 [ssw_size];
    int ssw_count = 0;

    curandState local_state;
    curandStatePhilox4_32_10_t state;

    float scores_subset[10];
    float row_maxs[600];
    int rmi = 0;*/

struct SparseIndexTracker {

    static constexpr int mi_max = 4;
    static constexpr int store_size = 32; //32 * 32; //8; //32; //32 * 4 * 4; //256; //32;

    //float log_row_sum[32 * 4 * 4];
    /*float log_row_sum[store_size];
    float ms[store_size];*/
    int count = 0;

    // ss = 32*16 mc = 0 ok
    // ss = 32*16 mc = 4 ok
    // ss = 32*16 mc = 8 nok
    // ss = 8 mc = 8 nok
    // ss = 32 * 32 mc = 4 nok, but less

    int thread_offset;

    __device__ SparseIndexTracker () {
        // At the start of your kernel or in your struct's constructor:
        int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
        int tid = bid * blockDim.x + threadIdx.x;
        thread_offset = tid * store_size;
    };

    template <typename Tensor1>
    __device__ void store_log_row_sum (Tensor1 &row_max, Tensor1 &row_sum, float* g_row_sum) {
        //return;
        //if (count >= store_size) return;
        //if (!thread0()) return;
        //if (count >= 32) return;
        if (count >= 8) return;
        int write_idx = thread_offset + count;

        /*// Temporary debug
        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
            int tid = bid * blockDim.x + threadIdx.x;
            printf("First thread: tid=%d, thread_offset=%d, count=%d, write_idx=%d, store_size=%d\n", tid, thread_offset, count, write_idx, store_size);
            printf("store_log_row_sum received pointer: %p\n", g_row_sum);
        }*/
        
        #pragma unroll
        for (int mi = 0; mi < mi_max; ++mi) {
            g_row_sum[write_idx + mi] = row_sum(mi);
            /*log_row_sum[count + mi] = row_sum(mi);
            ms[count + mi] = row_max(mi);*/
            //if (thread(0, PRINT_BID)) printf("rs = %f rm = %f\n", row_sum(mi), row_max(mi));
        }
        count += mi_max;
        //count %= store_size;
    };

    __device__ void final () {
        float s = 0.0f;
        //for (int i = 0; i < count; i++) s += (log_row_sum[i] + ms[i]);
        if (thread(0, PRINT_BID)) printf("s = %f\n", s);
    };

};

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

    //StochSparse<kNRows, Kernel_traits> ss;
    SparseIndexTracker sit;

#if 0
//-----------------
    static constexpr int mi_max = 4;
    static constexpr int store_size = 256; //32; //32 * 4 * 4; //256; //32;

    //float log_row_sum[32 * 4 * 4];
    float log_row_sum[store_size];
    float ms[store_size];
    int count = 0;
    //static constexpr int N = size(row_sum);

    /*//template <typename Tensor1>
    __device__ void store_log_row_sum () {//(Tensor1 &row_max, Tensor1 &row_sum) {
        #pragma unroll
        for (int mi = 0; mi < mi_max; ++mi) {
            log_row_sum[count + mi] = row_sum(mi);
            ms[count + mi] = row_max(mi);
        }
        count += mi_max;
        //count %= store_size;
    };*/
//-----------------
#endif

    __device__ Softmax_c() {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    //__forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2, float* g_row_sum) {
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
            FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            FLASH_NAMESPACE::reduce_sum</*zero_init=*/false>(scores, row_sum);
            //FLASH_NAMESPACE::reduce_sum_</*zero_init=*/false>(scores, row_sum);
            //sit.store_log_row_sum(row_sum);
            //auto dest_view = make_tensor(log_row_sum + count, make_layout(4));
            //cute::copy(row_sum, dest_view);
            //count = (count + mi_max) % store_size; // exper
        }
        sit.store_log_row_sum(row_max, row_sum, g_row_sum);
        /*
        //ss.original_coordinates(acc_s);
        //__syncthreads();
        //ss.store_ssweights(scores, row_max, row_sum);
        //ss.store_ssweights_test(scores, row_max, row_sum);
        ss.store_ssweights_test0(scores, row_max, row_sum);
        //__syncthreads();
        //ss.filter_ssweights(row_max, row_sum);
        //__syncthreads();
        //*/
        
    };

    __device__ void ss_final () {
        //ss.filter_ssweights0(row_max, row_sum);
        sit.final();
    };

}; 

}  // namespace FLASH_NAMESPACE
