#include <device_launch_parameters.h> //blockIdx.x, threadIdx.x, blockDim, threadDim

#include <ars/functions.h>
#include <ars/utils.h>

struct ParlArsIsoParams { //Isotropic ARS Parallelization Params
    int numPts;
    int numPtsAfterPadding;
    int blockSz;
    int numBlocks;
    int gridTotalSize;
    int gridTotalSizeAfterPadding;
    //depth of mega-matrix
    int coeffsMatNumCols;
    int coeffsMatNumColsPadded;
    int coeffsMatTotalSz;
    //Fourier matrix sum -> parallelization parameters
    int sumBlockSz;
    int sumGridSz;

    //Subdivision into chunks (big input data)
    int chunkMaxSz;
    int numChunks;
    int currChunkSz;

    //time profiling
    double srcExecTime;
    double gpu_srcExecTime;
    double dstExecTime;
    double gpu_dstExecTime;
};

struct ArsIsoParams {
    int arsIsoOrder;
    double arsIsoSigma;
    double arsIsoThetaToll;
    cuars::ArsKernelIso2dComputeMode arsIsoPnebiMode;
};

struct TestParams {
    // ArsIso (Isotropic Angular Radon Spectrum) params
    bool arsIsoEnable;
    bool gpu_arsIsoEnable;

    ArsIsoParams aiPms;

    bool extrainfoEnable;
    int fileSkipper;
};


// --------------------------------------------------------
// DIVISION IN CHUNKS (FOR BIG IMAGES)
// --------------------------------------------------------

/**
 * @return ceil(totNumPts / chunkSz)
 */
//__host__ //commented for ease in IDE
int numChunks(int totNumPts, int chunkSz);

/**
 * Return chunk number "@param round" start and end indices
 */
//__host__
thrust::pair<int, int> chunkStartEndIndices(int round, int totNumPts, int chunkSz);

// --------------------------------------------------------
// 2D->1D INDICIZATION IN FOURIER COEFFICIENT MATRIX
// --------------------------------------------------------

/**
 * When dealing with Fourier coefficient matrix, return index referring to the first point that is being dealt with.
 * In short, when computing Ars(means[i], means[j]) -> this function returns i.
 */
__device__
int getIfromTid(int tid, int n);

/**
 * When dealing with Fourier coefficient matrix, return index referring to the second point that is being dealt with.
 * In short, when computing Ars(means[i], means[j]) -> this function returns j.
 */
__device__
int getJfromTid(int tid, int n, int i);

// --------------------------------------------------------
// PNEBI FUNCTIONS
// PNEBI stands for Product of Negative Exponential and Bessel I, which is defined as
// 
//    PNEBI(k,x) = 2.0 * exp(-x) * besseli(k,x)
// 
// where besseli(k,x) is the modified Bessel function of the First Kind with order k. 
// --------------------------------------------------------

/** 
 * Computes the value of function of PNEBI(0,x) = 2.0 * exp(-x) * besseli(0,x) 
 * using the polynomial approximation of Abramowitz-Stegun (9.8.1)-(9.8.2). 
 * Common library functions computing besseli(0,x) leads to numeric overflow 
 * or exploits inaccurate (and substantially flat in our interval!) Hankel 
 * approximation. 
 */
__device__
double evaluatePnebi0Polynom(double x);

/** 
 * Evaluates PNEBI function in point x for different orders from 0 to n. 
 * This implementation is based on downward recurring formula as suggested in
 *  
 * Aa Vv, Numerical Recipes in C. The Art of Scientific Computing, 2nd edition, 1992. 
 * 
 * Used when ArsKernelIso2dComputeMode = PNEBI_DOWNWARD
 */
__device__
void evaluatePnebiVectorGPU(int n, double x, double* pnebis, int pnebisSz);

// --------------------------------------------------------
// GLOBAL CUDA KERNELS
// --------------------------------------------------------

/**
 * Insert Isotropic Gaussians Kernel function that uses Downward method for partial coefficients computing
 * @param means
 * @param sigma1
 * @param sigma2
 * @param numPts
 * @param fourierOrder
 * @param numColsPadded
 * @param pnebiMode
 * @param coeffsMat
 */
__global__
void iigDw(cuars::Vec2d* means, double sigma1, double sigma2, int numPts, int fourierOrder, int numColsPadded, cuars::ArsKernelIso2dComputeMode pnebiMode, double* coeffsMat);

/**
 * !! UNFINISHED
 * Insert Isotropic Gaussians Kernel, that uses LUT table method for partial coefficients computing
 * @param means
 * @param sigma1
 * @param sigma2
 * @param numPts
 * @param numPtsAfterPadding
 * @param fourierOrder
 * @param numColsPadded
 * @param pnebiMode
 * @param pnebiLUT
 * @param coeffsMat
 */
__global__
void iigLut(cuars::Vec2d* means, double sigma1, double sigma2, int numPts, int numPtsAfterPadding, int fourierOrder, int numColsPadded, cuars::ArsKernelIso2dComputeMode pnebiMode, cuars::PnebiLUT& pnebiLUT, double* coeffsMat);

/**
 * Compute partial sums of subsets of rows taken from iig output matrix
 * @param matIn
 * @param nrowsIn
 * @param ncols
 * @param matOut
 */
__global__
void makePartialSums(double* matIn, int nrowsIn, int ncols, double *matOut);

/**
 * Sum partial sums that result from makePartialSums()
 * @param matIn
 * @param nrows
 * @param ncols
 * @param vecOut
 */
__global__
void sumColumnsPartialSums(double* matIn, int nrows, int ncols, double* vecOut);

/**
 * Initialization of all parallelization parameters as members of struct @param pp;
 * @param blockSz and @param chunkMaxSz are supposed to be set "at runtime" as they can be set by the user
 * @param fourierOrder is the ARS Iso Fourier Order
 * and
 * @param numPts is the number of input points
 */
//__host__
void initParallelizationParams(ParlArsIsoParams& pp, int fourierOrder, int numPts, int blockSz, int chunkMaxSz);

/**
 * Same as above, but numPts is taken as max(numPtsSrc, numPtsDst)
 */
//__host__
void initParallelizationParams(ParlArsIsoParams& pp, int fourierOrder, int numPtsSrc, int numPtsDst, int blockSz, int chunkMaxSz);

/**
 * When input data need to be split into more than 1 chunk, this function helps with updating 
 * pp internal params according to current @param currChunkSz
 */
//__host__
void updateParallelizationParams(ParlArsIsoParams& pp, int currChunkSz);

/**
 * Compute Ars Iso (i.e. on SRC or DST data set) storing Ars output into @param coeffsArs host vector (d_coeffsArs device equivalent is created internally)
 * Ars Iso params and Parallelization params are specified in @param arsPms and @param paip respectively
 * @param points is the input vector of data points
 * @param startSrc, @param stopSrc and @param execTime are used for profiling execution times of the GPU kernel calls
 */
//__host__
void computeArsIsoGpu(ParlArsIsoParams& paip, ArsIsoParams& arsPms, const cuars::VecVec2d& points, double* coeffsArs, cudaEvent_t start, cudaEvent_t stop, double& execTime);