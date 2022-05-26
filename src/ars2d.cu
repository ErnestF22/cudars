/**
 * CudARS: Angular Radon Spectrum - CUDA version
 * Copyright (C) 2017-2020 Dario Lodi Rizzini.
 * Copyright (C) 2021- Dario Lodi Rizzini, Ernesto Fontana.
 *
 * CudARS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CudARS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with CudARS.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "cudars/ars2d.cuh"

// --------------------------------------------------------
// DIVISION IN CHUNKS (FOR BIG IMAGES)
// --------------------------------------------------------

//__host__

int numChunks(int totNumPts, int chunkSz)
{
    if (totNumPts % chunkSz > 1024)
        return (totNumPts / chunkSz) + 1;
    else
        return max(1, totNumPts / chunkSz);
}

//__host__

thrust::pair<int, int> chunkStartEndIndices(int round, int totNumPts, int chunkSz)
{
    thrust::pair<int, int> pr;

    // TODO: adaptation for cases when last chunk is small (< ~1000 pts)
    if (totNumPts <= chunkSz)
    {
        pr.first = 0;
        pr.second = totNumPts - 1;

        return pr;
    }
    pr.first = chunkSz * round;
    pr.second = min(totNumPts - 1, chunkSz * (round + 1) - 1);

    if (totNumPts - pr.second < 1025)
        pr.second = totNumPts - 1;

    return pr;
}

// --------------------------------------------------------
// 2D->1D INDICIZATION IN FOURIER COEFFICIENT MATRIX
// --------------------------------------------------------

__device__ int getIfromTid(int tid, int n)
{
    if (tid < 0 || n < 0)
        return -1;
    const int nId = n - 1; // max n in ids (indices start from 0)
    //    const int tidStart = tid; //useful for debugging
    /*i is equal to the number of times that we can subtract NID, NID-1, NID-2, ...
     from tid before tid goes below 0*/
    int i = 0;
    if (tid < nId)
    {
        //        printf("tid %d i %d       n %d\n", tidStart, i, n);
        return i;
    }

    if (tid >= 0.5 * nId * (nId + 1))
    { // maybe change this to improve efficiency?
        //        printf("tid %d i %d       n %d\n", tidStart, i, n);
        i = n;
        return n;
    }

    while (tid >= 0)
    {
        tid -= (nId - i);
        i++;
    }
    //    printf("tid %d i %d       n %d\n", tidStart, i - 1, n);
    return i - 1;
}

__device__ int getJfromTid(int tid, int n, int i)
{
    if (tid < 0 || n < 0 || i < 0)
        return -1;

    const int nId = n - 1; // max n in ids (indices start from 0)
    //    const int tidStart = tid; //useful for debugging
    //    const int iStart = i; //useful for debugging
    /*i is equal to the number of times that we can subtract NID, NID-1, NID-2, ...
     from tid before tid goes below 0*/
    int j = -1;

    if (tid >= 0.5 * nId * (nId + 1) || i > nId)
    { // maybe change this to improve efficiency?
        j = n;
        //        printf("tid %d i %d j %d       n %d\n", tidStart, iStart, j, n);
        return n;
    }

    while (i > 0)
    {
        tid -= (nId - i);
        i--;
    }
    j = tid + 1;
    //    printf("tid %d i %d j %d       n %d\n", tidStart, iStart, j, n);
    return j;
}

// --------------------------------------------------------
// PNEBI FUNCTIONS
// --------------------------------------------------------

__device__ double evaluatePnebi0Polynom(double x)
{
    double t, t2, tinv, val;

    if (x < 0.0)
        x = -x;
    t = x / 3.75;

    if (t < 1.0)
    {
        t2 = t * t;
        val = 1.0 + t2 * (3.5156229 + t2 * (3.0899424 + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.360768e-1 + t2 * 0.45813e-2)))));
        val = 2.0 * exp(-x) * val;
    }
    else
    {
        tinv = 1 / t;
        val = (0.39894228 + tinv * (0.1328592e-1 + tinv * (0.225319e-2 + tinv * (-0.157565e-2 + tinv *
                                                                                                    (0.916281e-2 + tinv * (-0.2057706e-1 + tinv * (0.2635537e-1 + tinv * (-0.1647633e-1 + tinv * 0.392377e-2))))))));
        val = 2.0 * val / sqrt(x);
    }

    return val;
}

__device__ void evaluatePnebiVectorGPU(int n, double x, double *pnebis, int pnebisSz)
{
    double factor, seqPrev, seqCurr, seqNext;
    //    if (pnebis.size() < n + 1) { //questa condizione dovrebbe essere già garantita prima della chiamata di evaluatePnebiVectorGPU
    //        pnebis.resize(n + 1); //ovvero: il questo resizing non dovrebbe essere necessario
    //    }

    if (x < 0.0)
        x = -x;

    // If x~=0, then BesselI(0,x) = 1.0 and BesselI(k,x) = 0.0 for k > 0.
    // Thus, PNEBI(0,x) = 2.0 and PNEBI(k,x) = 0.0 for k > 0.
    if (x < 1e-6)
    {
        pnebis[0] = 2.0;
        for (int i = 1; i < pnebisSz; ++i)
            pnebis[i] = 0.0;
        return;
    }

    // Computes bessel function using back recursion
    factor = 2.0 / x;
    seqPrev = 0.0; // bip
    seqCurr = 1.0; // bi
    seqNext = 0.0; // bim
    for (int k = 2 * (n + (int)sqrt(40.0 * n)); k >= 0; --k)
    {
        seqNext = seqPrev + factor * k * seqCurr;
        seqPrev = seqCurr;
        seqCurr = seqNext;
        if (k <= n)
        {
            pnebis[k] = seqPrev;
        }
        // To avoid overflow!
        if (seqCurr > cudars::BIG_NUM)
        {
            seqPrev *= cudars::SMALL_NUM;
            seqCurr *= cudars::SMALL_NUM;
            for (int i = 0; i < pnebisSz; ++i)
            {
                pnebis[i] *= cudars::SMALL_NUM;
            }
            // std::cerr << __FILE__ << "," << __LINE__ << ": ANTI-OVERFLOW!" << std::endl;
        }
    }

    double scaleFactor = evaluatePnebi0Polynom(x) / pnebis[0];
    for (int i = 0; i < pnebisSz; ++i)
    {
        pnebis[i] = scaleFactor * pnebis[i];
    }
}

// --------------------------------------------------------
// GLOBAL CUDA KERNELS
// --------------------------------------------------------

__global__ void iigDw(cudars::Vec2d *means, double sigma1, double sigma2, int numPts, int fourierOrder, int numColsPadded, cudars::ArsKernelIso2dComputeMode pnebiMode, double *coeffsMat)
{
    //    a.insertIsotropicGaussians(points, sigma);

    int index = blockIdx.x * blockDim.x + threadIdx.x; // index runs through a single block
    int stride = blockDim.x * gridDim.x;               // total number of threads in the grid

    const int totalNumComparisons = gridDim.x * blockDim.x;

    for (int tid = index; tid < totalNumComparisons; tid += stride)
    {

        int i = getIfromTid(tid, numPts);
        int j = getJfromTid(tid, numPts, i);
        //        printf("i %d j %d\n", i, j);

        if (i >= numPts || j >= numPts || j <= i)
        {
            //            printf("BAD INDEXING!!!!\n"); //could actually be ok because of padding... maybe optimization can be further improved in this regard
            continue;
        }

        cudars::Vec2d vecI = means[i];
        cudars::Vec2d vecJ = means[j];

        //            isotropicKer_.init(means[i], means[j], sigma);
        double dx, dy;
        dx = vecJ.x - vecI.x;
        dy = vecJ.y - vecI.y;
        double phi;

        //        if (dx == 0 && dy == 0) {
        //                        phi = 0.0; //mathematically undefined
        //            //            for (int k = 0; k <= numColsPadded; ++k) {
        //            //                int rowIndex = (i * numPtsAfterPadding) + j; //it's more a block index rather than row
        //            //                coeffsMat[rowIndex * numColsPadded + k] = 0.0;
        //            //            }
        ////            continue;
        //
        //        } else
        phi = atan2(dy, dx);

        double sigmaValSq = sigma1 * sigma1 + sigma2 * sigma2;
        double lambdaSqNorm = 0.25 * (dx * dx + dy * dy) / sigmaValSq;

        //            isotropicKer_.updateFourier(arsfOrder_, coeffs_, w);
        double wNorm = 1.0 / (numPts * numPts);
        double weight = wNorm / sqrt(2.0 * M_PI * sigmaValSq);

        // updating Fourier coefficients (2 modes)
        if (pnebiMode == cudars::ArsKernelIso2dComputeMode::PNEBI_DOWNWARD)
        {
            //                updateARSF2CoeffRecursDown(lambdaSqNorm, phi, w2, nFourier, coeffs);

            double cth2, sth2;
            cth2 = cos(2.0 * phi);
            sth2 = sin(2.0 * phi);
            //                updateARSF2CoeffRecursDown(lambda, cth2, sth2, factor, n, coeffs);

            int pnebisSz = fourierOrder + 1;
            // TODO: find a better solution instead of hard-coding 21
            double pnebis[21]; // Fourier Order + 1
            if (pnebis == nullptr)
                printf("ERROR ALLOCATING WITH NEW[]!\n");
            for (int pn = 0; pn < pnebisSz; ++pn)
                pnebis[pn] = 0.0;

            double sgn, cth, sth, ctmp, stmp;

            // Fourier Coefficients
            //                if (coeffs.size() != 2 * n + 2) {
            //                    std::cerr << __FILE__ << "," << __LINE__ << ": invalid size of Fourier coefficients vector " << coeffs.size() << " should be " << (2 * n + 2) << std::endl;
            //                    return;
            //                }

            evaluatePnebiVectorGPU(fourierOrder, lambdaSqNorm, pnebis, pnebisSz);
            //                ARS_PRINT(pnebis[0]);

            //!!!! factor = w2
            double factor = weight;
            int rowIndex = tid; // = tid
            coeffsMat[rowIndex * numColsPadded + 0] += 0.5 * factor * pnebis[0];
            //            printf("coeff 0 %f\n", 0.5 * factor * pnebis[0]);

            sgn = -1.0;
            cth = cth2;
            sth = sth2;
            //!!!! n in the for below is fourierOrder
            for (int k = 1; k <= fourierOrder; ++k)
            {
                //                printf("coeff %d %f\n", 2 * k, factor * pnebis[k] * sgn * cth);
                //                printf("coeff %d %f\n", 2 * k + 1, factor * pnebis[k] * sgn * sth);
                coeffsMat[(rowIndex * numColsPadded) + (2 * k)] += factor * pnebis[k] * sgn * cth;
                coeffsMat[(rowIndex * numColsPadded) + ((2 * k) + 1)] += factor * pnebis[k] * sgn * sth;
                sgn = -sgn;
                ctmp = cth2 * cth - sth2 * sth;
                stmp = sth2 * cth + cth2 * sth;
                cth = ctmp;
                sth = stmp;
            }

            //            delete pnebis;
        }
        else
            printf("ERROR: pnebi mode is NOT Downward!\n");
    }
}

__global__ void iigLut(cudars::Vec2d *means, double sigma1, double sigma2, int numPts, int numPtsAfterPadding, int fourierOrder, int numColsPadded, cudars::ArsKernelIso2dComputeMode pnebiMode, cudars::PnebiLUT &pnebiLUT, double *coeffsMat)
{
    //    a.insertIsotropicGaussians(points, sigma);

    int index = blockIdx.x * blockDim.x + threadIdx.x; // index runs through a single block
    int stride = blockDim.x * gridDim.x;               // total number of threads in the grid

    const int totalNumComparisons = numPtsAfterPadding * numPtsAfterPadding;

    for (int tid = index; tid < totalNumComparisons; tid += stride)
    {

        int j = tid % numPtsAfterPadding;
        int i = (tid - j) / numPtsAfterPadding;
        //        printf("i %d j %d\n", i, j);
        //        printf("tid %d i %d j %d tidIJ %d --- numPts %d numPtsAfterPadding %d numColsPadded %d totNumComp %d index %d\n", tid, i, j, i * numPtsAfterPadding + j, numPts, numPtsAfterPadding, numColsPadded, totalNumComparisons, index);

        if (i >= numPts || j >= numPts || j <= i)
            continue;

        cudars::Vec2d vecI = means[i];
        cudars::Vec2d vecJ = means[j];

        //            isotropicKer_.init(means[i], means[j], sigma);
        double dx, dy;
        dx = vecJ.x - vecI.x;
        dy = vecJ.y - vecI.y;
        double phi;

        //        if (dx == 0 && dy == 0) {
        //                        phi = 0.0; //mathematically undefined
        //            //            for (int k = 0; k <= numColsPadded; ++k) {
        //            //                int rowIndex = (i * numPtsAfterPadding) + j; //it's more a block index rather than row
        //            //                coeffsMat[rowIndex * numColsPadded + k] = 0.0;
        //            //            }
        ////            continue;
        //
        //        } else
        phi = atan2(dy, dx);

        double sigmaValSq = sigma1 * sigma1 + sigma2 * sigma2;
        double lambdaSqNorm = 0.25 * (dx * dx + dy * dy) / sigmaValSq;
        printf("lambdaSqNorm %f\n", lambdaSqNorm); // just to avoid seeing warning of unused variable when compiling

        //            isotropicKer_.updateFourier(arsfOrder_, coeffs_, w);
        double wNorm = 1.0 / (numPts * numPts);
        double weight = wNorm / sqrt(2.0 * M_PI * sigmaValSq);

        // updating Fourier coefficients (2 modes)
        if (pnebiMode == cudars::ArsKernelIso2dComputeMode::PNEBI_LUT)
        {
            printf("Method not fully implemented!\n");
            continue;

            //                updateARSF2CoeffRecursDownLUT(lambdaSqNorm_, phi_, w2, nFourier, pnebiLut_, coeffs);
            double cth2, sth2;
            // fastCosSin(2.0 * phi, cth2, sth2); //già commentata nell'originale
            cth2 = cos(2.0 * phi);
            sth2 = sin(2.0 * phi);

            int pnebisSz = fourierOrder + 1;
            double *pnebis = new double[pnebisSz];
            if (pnebis == nullptr)
                printf("ERROR ALLOCATING WITH NEW[]!\n");
            double sgn, cth, sth, ctmp, stmp;

            coeffsMat[0] = 0.5 * weight * pnebis[0]; // factor = w2

            sgn = -1.0;
            cth = cth2;
            sth = sth2;
            for (int k = 1; k <= fourierOrder; ++k)
            {

                coeffsMat[2 * k] = pnebis[k] * weight * sgn * cth;
                coeffsMat[2 * k + 1] = pnebis[k] * weight * sgn * sth;
                sgn = -sgn;
                ctmp = cth2 * cth - sth2 * sth;
                stmp = sth2 * cth + cth2 * sth;
                cth = ctmp;
                sth = stmp;
            }

            //            delete pnebis;
        }
        else
            printf("ERROR: pnebi mode is not LUT!\n");
    }
}

__global__ void makePartialSums(double *matIn, int nrowsIn, int ncols, double *matOut)
{
    //    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int index = threadIdx.x * gridDim.x + blockIdx.x; //!!! indexing is done "column-major" (in terms of kernel grid)
    int stride = blockDim.x * gridDim.x;

    int totalSzIn = nrowsIn * ncols; // matrix is considered of size nrows*ncols, with nrows = sumNaturalsUpToN(numPts)

    // 1 thread of the kernel composes 1 box of matOut
    int rowOutId = threadIdx.x;
    int colOutId = blockIdx.x;
    int idOut = rowOutId * ncols + colOutId;
    //    printf("rowOutId %d colOutId %d idOut %d\n", rowOutId, colOutId, idOut);

    int nOps = 0;
    for (int idx = index; idx < totalSzIn; idx += stride)
    {
        //        printf("nOps %d\n", nOps);

        matOut[idOut] += matIn[idx];
        nOps++;
    }
}

__global__ void sumColumnsPartialSums(double *matPartialSums, int nrows, int ncols, double *sums)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index runs through a single block
    int stride = blockDim.x * gridDim.x;               // total number of threads in the grids

    int totalSz = nrows * ncols; // matrix is considered of size nrows*ncols, with nrows = sumNaturalsUpToN(numPts)

    for (int idx = index; idx < totalSz; idx += stride)
    {
        //        int totalIndex = (((i * nrows) + j) * ncols) + k;
        int k = idx % ncols;
        //        int rowIdx = (idx - k) / ncols; //useful for debugging
        //        printf("k %d rowIdx %d; accessing mat[%d]\n", k, rowIdx, idx);
        sums[k] += matPartialSums[idx];
    }
}

//__host__

void initParallelizationParams(ParlArsIsoParams &pp, int fourierOrder, int numPts, int blockSz, int chunkMaxSz)
{

    pp.numPts = numPts;
    int numPtsAfterPadding = numPts;
    pp.numPtsAfterPadding = numPtsAfterPadding;

    pp.chunkMaxSz = chunkMaxSz;
    int nc = numChunks(numPts, chunkMaxSz);
    pp.numChunks = nc;

    pp.blockSz = blockSz;

    const int coeffsMatNumCols = 2 * fourierOrder + 2;
    pp.coeffsMatNumCols = coeffsMatNumCols;
    const int coeffsMatNumColsPadded = coeffsMatNumCols;
    pp.coeffsMatNumColsPadded = coeffsMatNumColsPadded;
}

//__host__

void initParallelizationParams(ParlArsIsoParams &pp, int fourierOrder, int numPtsSrc, int numPtsDst, int blockSz, int chunkMaxSz)
{

    int numPts = std::max<int>(numPtsSrc, numPtsDst);
    pp.numPts = numPts;
    int numPtsAfterPadding = numPts;
    pp.numPtsAfterPadding = numPtsAfterPadding;

    pp.chunkMaxSz = chunkMaxSz;
    int nc = numChunks(numPts, chunkMaxSz);
    pp.numChunks = nc;

    pp.blockSz = blockSz;

    const int coeffsMatNumCols = 2 * fourierOrder + 2;
    pp.coeffsMatNumCols = coeffsMatNumCols;
    const int coeffsMatNumColsPadded = coeffsMatNumCols;
    pp.coeffsMatNumColsPadded = coeffsMatNumColsPadded;
}

//__host__

void updateParallelizationParams(ParlArsIsoParams &pp, int currChunkSz)
{
    // Setting up parallelization
    // Parallelization parameters
    pp.currChunkSz = currChunkSz;
    // Fourier coefficients mega-matrix computation
    const int gridTotalSize = cudars::sumNaturalsUpToN(pp.currChunkSz - 1); // total number of threads in grid Fourier coefficients grid - BEFORE PADDING
    pp.gridTotalSize = gridTotalSize;

    const int numBlocks = floor(gridTotalSize / pp.blockSz) + 1; // number of blocks in grid (each block contains blockSize threads)
    pp.numBlocks = numBlocks;
    const int gridTotalSizeAfterPadding = pp.blockSz * numBlocks;
    pp.gridTotalSizeAfterPadding = gridTotalSizeAfterPadding;

    const int coeffsMatTotalSz = gridTotalSizeAfterPadding * pp.coeffsMatNumColsPadded; // sumNaturalsUpToN(numPts - 1) * coeffsMatNumColsPadded
    pp.coeffsMatTotalSz = coeffsMatTotalSz;

    // Fourier matrix sum -> parallelization parameters
    const int sumGridSz = pp.coeffsMatNumColsPadded; // = 2 * fourierOrder + 2
    pp.sumGridSz = sumGridSz;
    const int sumBlockSz = pp.blockSz;
    pp.sumBlockSz = sumBlockSz;

    //    std::cout << "Parallelization params:" << std::endl;
    //    std::cout << "numPts " << pp.numPts << " blockSize " << pp.blockSz << " numBlocks " << numBlocks
    //            << " gridTotalSize " << gridTotalSize << " gridTotalSizeAP " << gridTotalSizeAfterPadding << std::endl;
    //    std::cout << "sumBlockSz " << sumBlockSz << " sumGridSz " << sumGridSz << std::endl;
    //        std::cout << "sum parallelization params: " << std::endl
    //            << "coeffMatNumCols " << pp.coeffsMatNumCols << " coeffsMatTotalSz " << coeffsMatTotalSz << std::endl;

    std::cout << "\nCalling kernel functions on GPU...\n"
              << std::endl;
}

//__host__

void computeArsIsoGpu(ParlArsIsoParams &paip, ArsIsoParams &arsPms, const cudars::VecVec2d &points, double *coeffsArs, cudaEvent_t start, cudaEvent_t stop, double &execTime)
{

    std::cout << "\n---Estimating Ars Iso---\n"
              << std::endl;

    //    double* coeffsArs = new double [paip.coeffsMatNumColsPadded];
    double *d_coeffsArs;
    cudaMalloc((void **)&d_coeffsArs, paip.coeffsMatNumColsPadded * sizeof(double));
    cudaMemset(d_coeffsArs, 0.0, paip.coeffsMatNumColsPadded * sizeof(double));

    for (int i = 0; i < paip.numChunks; ++i)
    {
        std::cout << "NUMPTS " << paip.numPts << std::endl;
        thrust::pair<int, int> indicesStartEnd = chunkStartEndIndices(i, paip.numPts, paip.chunkMaxSz);
        int currChunkSz = (indicesStartEnd.second - indicesStartEnd.first) + 1;

        updateParallelizationParams(paip, currChunkSz); // TODO: put chunkMaxSz as TestParams struct member?

        cudars::Vec2d *kernelInput;
        cudaMalloc((void **)&kernelInput, currChunkSz * sizeof(cudars::Vec2d));
        //        cudaMemcpy(kernelInput, points.data(), numPtsAfterPadding * sizeof (cudars::Vec2d), cudaMemcpyHostToDevice);
        std::cout << "round " << i + 1 << "/" << paip.numChunks << " -> "
                  << "chunk-beg " << indicesStartEnd.first << " chunk-end " << indicesStartEnd.second << " --- chunk-size " << currChunkSz << std::endl;
        cudars::VecVec2d dataChunk(points.begin() + indicesStartEnd.first, points.begin() + (indicesStartEnd.first + currChunkSz));
        cudaMemcpy(kernelInput, dataChunk.data(), (dataChunk.size()) * sizeof(cudars::Vec2d), cudaMemcpyHostToDevice);

        // Fourier matrix sum -> parallelization parameters
        std::cout << "Parallelization params:" << std::endl;
        std::cout << "numPts " << paip.numPts << " blockSize " << paip.blockSz << " numBlocks " << paip.numBlocks
                  << " gridTotalSize " << paip.gridTotalSize << " gridTotalSizeAP " << paip.gridTotalSizeAfterPadding << std::endl;
        std::cout << "sumBlockSz " << paip.sumBlockSz << " sumGridSz " << paip.sumGridSz << std::endl;

        std::cout << "sum parallelization params: " << std::endl
                  << "coeffMatNumCols " << paip.coeffsMatNumCols << " coeffsMatTotalSz " << paip.coeffsMatTotalSz << std::endl;

        double *d_coeffsMat;
        cudaMalloc((void **)&d_coeffsMat, paip.coeffsMatTotalSz * sizeof(double));
        cudaMemset(d_coeffsMat, 0.0, paip.coeffsMatTotalSz * sizeof(double));

        double *d_partsums;
        cudaMalloc((void **)&d_partsums, paip.blockSz * paip.coeffsMatNumColsPadded * sizeof(double));
        cudaMemset(d_partsums, 0.0, paip.blockSz * paip.coeffsMatNumColsPadded * sizeof(double));

        cudaEventRecord(start);
        iigDw<<<paip.numBlocks, paip.blockSz>>>(kernelInput, arsPms.arsIsoSigma, arsPms.arsIsoSigma, currChunkSz, arsPms.arsIsoOrder, paip.coeffsMatNumColsPadded, arsPms.arsIsoPnebiMode, d_coeffsMat);
        //    sumColumnsNoPadding << <1, sumBlockSz>> >(coeffsMat, gridTotalSizeAfterPadding, coeffsMatNumColsPadded, d_coeffsArs);
        makePartialSums<<<paip.coeffsMatNumColsPadded, paip.blockSz>>>(d_coeffsMat, paip.gridTotalSizeAfterPadding, paip.coeffsMatNumColsPadded, d_partsums);
        sumColumnsPartialSums<<<paip.coeffsMatNumColsPadded, 1>>>(d_partsums, paip.blockSz, paip.coeffsMatNumColsPadded, d_coeffsArs);
        cudaEventRecord(stop);

        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

        //    for (int i = 0; i < coeffsMatNumColsPadded; ++i) {
        //        std::cout << "coeffsArs[" << i << "] " << coeffsArs[i] << std::endl;
        //    }

        cudaFree(d_partsums);
        cudaFree(d_coeffsMat);
        cudaFree(kernelInput);
    }

    cudaMemcpy(coeffsArs, d_coeffsArs, paip.coeffsMatNumColsPadded * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_coeffsArs);

    cudaEventSynchronize(stop);
    float millisecondsExecTime = 0.0f;
    cudaEventElapsedTime(&millisecondsExecTime, start, stop);
    std::cout << "\ninsertIsotropicGaussians() -> exec time: " << millisecondsExecTime << " ms" << std::endl;
    execTime = millisecondsExecTime;
}

void gpu_estimateRotationArsIso(const ArsImgTests::PointReaderWriter &pointsSrc, const ArsImgTests::PointReaderWriter &pointsDst, TestParams &tp, ParlArsIsoParams &paip, double &rotOut)
{
    // ARS SRC -> preparation for kernel calls and kernel calls
    cudaEvent_t startSrc, stopSrc; // timing using CUDA events
    cudaEventCreate(&startSrc);
    cudaEventCreate(&stopSrc);

    const cudars::VecVec2d &inputSrc = pointsSrc.points();
    initParallelizationParams(paip, tp.aiPms.arsIsoOrder, inputSrc.size(), paip.blockSz, paip.chunkMaxSz); // cudarsIso.init()
    double *coeffsArsSrc = new double[paip.coeffsMatNumColsPadded];
    computeArsIsoGpu(paip, tp.aiPms, inputSrc, coeffsArsSrc, startSrc, stopSrc, paip.gpu_srcExecTime); // cudarsIso.compute()

    cudaEventDestroy(startSrc);
    cudaEventDestroy(stopSrc);
    // END OF ARS SRC

    //    std::cout << "\n------\n" << std::endl; //"pause" between ars src and ars dst

    // ARS DST -> preparation for kernel calls and kernel calls
    cudaEvent_t startDst, stopDst; // timing using CUDA events
    cudaEventCreate(&startDst);
    cudaEventCreate(&stopDst);

    const cudars::VecVec2d &inputDst = pointsDst.points();
    initParallelizationParams(paip, tp.aiPms.arsIsoOrder, inputDst.size(), paip.blockSz, paip.chunkMaxSz); // cudarsIso.init()
    double *coeffsArsDst = new double[paip.coeffsMatNumColsPadded];
    computeArsIsoGpu(paip, tp.aiPms, inputDst, coeffsArsDst, startDst, stopDst, paip.gpu_dstExecTime); // cudarsIso.compute()

    cudaEventDestroy(startDst);
    cudaEventDestroy(stopDst);
    // END OF ARS DST

    std::cout << std::endl
              << "---Computing corelation---" << std::endl;

    // Final computations (correlation, ...) on CPU
    //     std::cout << "\nARS Coefficients:\n";
    //     std::cout << "Coefficients: Src, Dst, Cor" << std::endl;

    double thetaMax, corrMax, fourierTol;
    fourierTol = 1.0; // TODO: check for a proper tolerance

    std::vector<double> coeffsCor;
    {
        cudars::ScopedTimer("ars.correlation()");
        std::vector<double> tmpSrc;
        tmpSrc.assign(coeffsArsSrc, coeffsArsSrc + paip.coeffsMatNumColsPadded);
        std::vector<double> tmpDst;
        tmpDst.assign(coeffsArsDst, coeffsArsDst + paip.coeffsMatNumColsPadded);
        cudars::computeFourierCorr(tmpSrc, tmpDst, coeffsCor);
        cudars::findGlobalMaxBBFourier(coeffsCor, 0.0, M_PI, tp.aiPms.arsIsoThetaToll, fourierTol, thetaMax, corrMax);
        rotOut = thetaMax; //!! rotOut is passed to the function as reference
    }

    //  Output coeffs check: CPU version
    //    arsSrc.setCoefficients(coeffsArsSrc, paip.coeffsMatNumCols);
    //    //    for (int i = 0; i < coeffsVectorMaxSz; i++) {
    //    //        std::cout << "arsSrc - coeff_d[" << i << "] " << d_coeffsMat1[i] << std::endl;
    //    //    }
    //    arsDst.setCoefficients(coeffsArsDst, paip.coeffsMatNumCols);
    //    for (int i = 0; i < arsSrc.coefficients().size() && i < arsDst.coefficients().size(); ++i) {
    //        std::cout << "\t" << i << " \t" << arsSrc.coefficients().at(i) << " \t" << arsDst.coefficients().at(i) << " \t" << coeffsCor[i] << std::endl;
    //    }

    //  Output coeffs check: GPU version
    //    for (int i = 0; i < paip.coeffsMatNumCols; ++i) {
    //        std::cout << "\t" << i << " \t" << coeffsArsSrc[i] << " \t" << coeffsArsDst[i] << " \t" << coeffsCor[i] << std::endl;
    //    }
    //    std::cout << std::endl;

    std::cout << std::endl
              << "ROT OUT " << rotOut << std::endl;

    // Computes the rotated points,centroid, affine transf matrix between src and dst
    ArsImgTests::PointReaderWriter pointsRot(pointsSrc.points());
    cudars::Vec2d centroidSrc = pointsSrc.computeCentroid();
    cudars::Vec2d centroidDst = pointsDst.computeCentroid();
    cudars::Affine2d rotSrcDst = ArsImgTests::PointReaderWriter::coordToTransform(0.0, 0.0, rotOut);
    //    cudars::Vec2d translSrcDst = centroidDst - rotSrcDst * centroidSrc;
    cudars::Vec2d translSrcDst;
    cudars::vec2diff(translSrcDst, centroidDst, cudars::aff2TimesVec2WRV(rotSrcDst, centroidSrc));
    //    std::cout << "centroidSrc " << centroidSrc.x << " \t" << centroidSrc.y << "\n"
    //            << "centroidDst " << centroidDst.x << " \t" << centroidDst.y << "\n"
    //            << "rotSrcDst\n" << rotSrcDst << "\n"
    //            << "translation: [" << translSrcDst.x << " \t" << translSrcDst.y << "] rotation[deg] " << (180.0 / M_PI * rotOut) << "\n";
    pointsRot.applyTransform(translSrcDst.x, translSrcDst.y, rotOut);

    //    double rotTrue = pointsDst.getRotTheta() - pointsSrc.getRotTheta();
    //    std::cout << "\n***\npointsDst.getrotTheta() [deg]" << (180 / M_PI * pointsDst.getRotTheta())
    //            << ", pointsSrc.getrotTheta() [deg] " << (180.0 / M_PI * pointsSrc.getRotTheta()) << "\n";
    //    std::cout << "rotTrue[deg] \t" << (180.0 / M_PI * rotTrue) << " \t" << (180.0 / M_PI * cudars::mod180(rotTrue)) << std::endl;
    //    std::cout << "rotArs[deg] \t" << (180.0 / M_PI * rotOut) << " \t" << (180.0 / M_PI * cudars::mod180(rotOut)) << std::endl;

    // Free CPU memory
    delete coeffsArsSrc;
    delete coeffsArsDst;
}