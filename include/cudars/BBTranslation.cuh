#ifndef CUDARS_BBTRANSLATION_CUH_
#define CUDARS_BBTRANSLATION_CUH_

#include <cudars/definitions.h>
#include <cudars/utils.h>

// #include <cudars/CuBox.cuh>

#include <cudars/prioqueue.cuh>

static constexpr int DIM = 2;
static constexpr int SPLIT_NUM = (1 << DIM);

/**
 * @brief Main method
 */
__global__ void computeBBTransl_kernel(cudars::Vec2d* ptsSrc_, cudars::Vec2d* ptsDst_,
                                       cudars::Vec2d &translOpt, cudars::Vec2d &translMin_, cudars::Vec2d &translMax_,
                                       double eps_, int numMaxIter_, double res_, int ptsSrcSize, int ptsDstSize);

#endif /*CUDARS_BBTRANSLATION_CUH_*/
