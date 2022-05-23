
/**
 * ARS - Angular Radon Spectrum
 * Copyright (C) 2017 Dario Lodi Rizzini.
 *
 * ARS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ARS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with ARS.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <chrono>

#include "ars/Profiler.h"

#include "ars/ars2d.cuh"
#include <ars/ars2d.h>
#include <ars/ConsensusTranslationEstimator.cuh>
#include <ars/ConsensusTranslationEstimator.h>

#include "ars/mpeg7RW.h"

void plotGrid(const cuars::ConsensusTranslationEstimator2d::Grid &grid, const cuars::Vec2d &translMin, double translRes, const std::string &filename, double factor);

void gpu_estimateRotationArsIso(const ArsImgTests::PointReaderWriter &pointsSrc, const ArsImgTests::PointReaderWriter &pointsDst, TestParams &tp, ParlArsIsoParams &paip, double &rotOut);

void computeArsTec(cuars::VecVec2d &translCandidates, const cuars::VecVec2d &pointsSrc, const cuars::VecVec2d &pointsDst, cuars::ArsTecParams &translParams);

int main(int argc, char **argv)
{
    cuars::AngularRadonSpectrum2d arsSrc;
    cuars::AngularRadonSpectrum2d arsDst;
    ArsImgTests::PointReaderWriter pointsSrc;
    ArsImgTests::PointReaderWriter pointsDst;

    TestParams testParams;
    ParlArsIsoParams paiParams;
    cuars::ArsTecParams translParams;

    rofl::ParamMap params;
    std::string filenameCfg;
    std::string filenameSrc;
    std::string filenameDst;
    std::string filenameRot;
    std::string filenameArsSrc;
    std::string filenameArsDst;
    std::string filenameArsRot;
    std::string filenameArsCor;
    std::string filenameCovSrc;
    std::string filenameCovDst;
    int arsOrder;
    double arsSigma, arsThetaToll;
    double rotTrue, rotArs;
    // The variables below are for I/O related functionalities (plot, etc.) that are highly Eigen-based and are present in the CPU-only ArsImgTests...
    // Maybe implement them later
    //     double sampleRes, sampleAng;
    //     int sampleNum;
    //     bool saveOn;
    //     bool saveCov;

    params.read(argc, argv);
    params.getParam<std::string>("cfg", filenameCfg, "");
    std::cout << "config filename: " << filenameCfg << std::endl;
    if (filenameCfg != "")
    {
        params.read(filenameCfg);
    }

    params.read(argc, argv);
    params.getParam<std::string>("src", filenameSrc, "/home/rimlab/Downloads/mpeg7_point_tests/noise000_occl00_rand000/apple-1_xp0686_yp0967_t059_sigma0001_occl000.txt");
    params.getParam<std::string>("dst", filenameDst, "/home/rimlab/Downloads/mpeg7_point_tests/noise000_occl00_rand000/apple-1_xp0749_yn0521_t090_sigma0001_occl000.txt");
    params.getParam<int>("arsOrder", arsOrder, 20);
    params.getParam<double>("arsSigma", arsSigma, 1.0);
    params.getParam<double>("arsTollDeg", arsThetaToll, 1.0);
    arsThetaToll *= M_PI / 180.0;
    //    params.getParam<double>("sampleResDeg", sampleRes, 0.5);
    //    sampleRes *= M_PI / 180.0;
    //    params.getParam<bool>("saveOn", saveOn, false);
    //    params.getParam<bool>("saveCov", saveCov, false);

    params.getParam<bool>("extrainfoEnable", testParams.extrainfoEnable, bool(true));

    // ArsIso params (CPU and GPU)
    params.getParam<bool>("arsisoEnable", testParams.arsIsoEnable, false);
    params.getParam<bool>("gpu_arsisoEnable", testParams.gpu_arsIsoEnable, true);
    params.getParam<int>("arsisoOrder", testParams.aiPms.arsIsoOrder, 20);
    params.getParam<double>("arsisoSigma", testParams.aiPms.arsIsoSigma, 1.0);
    params.getParam<double>("arsisoTollDeg", testParams.aiPms.arsIsoThetaToll, 0.5);
    testParams.aiPms.arsIsoThetaToll *= M_PI / 180.0;
    //    params.getParam<unsigned int>("arsisoPnebiMode", tp.arsIsoPnebiMode, cuars::ArsKernelIsotropic2d::ComputeMode::PNEBI_DOWNWARD);

    arsSrc.setARSFOrder(testParams.aiPms.arsIsoOrder);
    //    arsSrc.initLUT(0.0001);
    //    arsSrc.setComputeMode(ars::ArsKernelIsotropic2d::ComputeMode::PNEBI_LUT);
    arsSrc.setComputeMode(cuars::ArsKernelIso2dComputeMode::PNEBI_DOWNWARD);
    arsDst.setARSFOrder(testParams.aiPms.arsIsoOrder);
    arsDst.setComputeMode(cuars::ArsKernelIso2dComputeMode::PNEBI_DOWNWARD);

    // parallelization parameters
    params.getParam<int>("blockSz", paiParams.blockSz, 256);
    params.getParam<int>("chunkMaxSz", paiParams.chunkMaxSz, 4096);

    params.getParam<double>("translRes", translParams.translRes, 2.0);
    // params.getParamContainer("translMin", translMin.data(), translMin.data() + translMin.size(), "[-10.0,-10.0]", double(0.0), "[,]"); //TODO: adapt ParamContainer to Cuda types
    params.getParam<double>("translMin-x", translParams.translMin.x, 0);
    params.getParam<double>("translMin-y", translParams.translMin.y, 0);
    params.getParam<double>("translMax-x", translParams.translMax.x, 1000.0);
    params.getParam<double>("translMax-y", translParams.translMax.y, 1000.0);
    // params.getParamContainer("translGt", translGt.data(), translGt.data() + translGt.size(), "[-4.2,5.0]", double(1.0), "[,]");
    params.getParam<double>("translGt-x", translParams.translGt.x, -4.2);
    params.getParam<double>("translGt-y", translParams.translGt.y, 5.0);
    params.getParamContainer("gridSize", translParams.gridSize.data(), translParams.gridSize.data() + translParams.gridSize.size(), "[1000,1000]", int(0), "[,]");
    params.getParamContainer("gridWin", translParams.gridWin.data(), translParams.gridWin.data() + translParams.gridWin.size(), "[10,10]", int(1), "[,]");
    params.getParam<bool>("adaptive", translParams.adaptiveGrid, true);
    params.getParam<bool>("plot", translParams.plot, false);

    std::cout << "\nParameter values:\n";
    params.write(std::cout);
    std::cout << std::endl;

    // Loads files and computes the rotation
    std::cout << "\n*****\nLoading file \"" << filenameSrc << "\"" << std::endl;
    pointsSrc.load(filenameSrc);
    std::cout << "\n*****\nLoading file \"" << filenameDst << "\"" << std::endl;
    pointsDst.load(filenameDst);
    std::cout << "  points src " << pointsSrc.points().size() << ", points dst " << pointsDst.points().size() << std::endl;

    int numPts = std::min<int>(pointsSrc.points().size(), pointsDst.points().size()); // the two should normally be equals
    //    int numPtsAfterPadding = numPts;

    // ARS parameters setting
    arsSrc.setARSFOrder(arsOrder);
    arsDst.setARSFOrder(arsOrder);
    cuars::ArsKernelIso2dComputeMode pnebiMode = cuars::ArsKernelIso2dComputeMode::PNEBI_DOWNWARD;
    arsSrc.setComputeMode(pnebiMode);
    arsDst.setComputeMode(pnebiMode);

    std::cout << "\n------\n"
              << std::endl;

    std::cout << "\n\nCalling kernel functions on GPU\n"
              << std::endl;

    gpu_estimateRotationArsIso(pointsSrc, pointsDst, testParams, paiParams, rotArs);

    rotTrue = pointsDst.getRotTheta() - pointsSrc.getRotTheta();
    std::cout << "\n***\npointsDst.getrotTheta() [deg]" << (180 / M_PI * pointsDst.getRotTheta())
              << ", pointsSrc.getrotTheta() [deg] " << (180.0 / M_PI * pointsSrc.getRotTheta()) << "\n";
    std::cout << "rotTrue[deg] \t" << (180.0 / M_PI * rotTrue) << " \t" << (180.0 / M_PI * cuars::mod180(rotTrue)) << std::endl;
    std::cout << "rotArs[deg] \t" << (180.0 / M_PI * rotArs) << " \t" << (180.0 / M_PI * cuars::mod180(rotArs)) << std::endl;

    // APPLY COMPUTED ROTATION
    //  Computes the rotated points, centroid, affine transf matrix between src and dst
    ArsImgTests::PointReaderWriter pointsRot(pointsSrc.points());
    cuars::Vec2d centroidSrc = pointsSrc.computeCentroid();
    cuars::Vec2d centroidDst = pointsDst.computeCentroid();
    cuars::Affine2d rotSrcDst = ArsImgTests::PointReaderWriter::coordToTransform(0.0, 0.0, rotArs);
    //    cuars::Vec2d translSrcDst = centroidDst - rotSrcDst * centroidSrc;
    cuars::Vec2d translSrcDst;
    cuars::vec2diff(translSrcDst, centroidDst, cuars::aff2TimesVec2WRV(rotSrcDst, centroidSrc));
    pointsRot.applyTransform(translSrcDst.x, translSrcDst.y, rotArs);

    if (translParams.adaptiveGrid)
    {
        cuars::Vec2d translMin;
        translMin.x = pointsDst.xmin() - pointsSrc.xmax();
        translMin.y = pointsDst.ymin() - pointsSrc.ymax();
        //        std::cout << std::endl << "tmin [m]\n" << translMin << std::endl;
        translParams.translMin = translMin;

        cuars::Vec2d translMax;
        translMax.x = pointsDst.xmax() - pointsSrc.xmin();
        translMax.y = pointsDst.ymax() - pointsSrc.ymin();
        //        std::cout << "tmax [m]\n" << translMax << std::endl;
        translParams.translMax = translMax;
    }

    cuars::Grid2d grid; //!! the only 2 remaining external classes used are here
    cuars::PeakFinder2d pf;
    cuars::VecVec2d translCandidates;
    computeArsTec(translCandidates, pointsRot.points(), pointsDst.points(), translParams);

    std::cout << "Estimated translation values:\n";
    // cuars::ConsensusTranslationEstimator2d translEstimOutput(...) //constructor can be used for example to fill the class with the outputs
    for (auto &pt : translCandidates)
    {
        std::cout << "  [";
        // cuars::printVec2d(pt);
        std::cout << pt.x << "\t" << pt.y;
        std::cout << "]\n";
    }

    return 0;
}

void plotGrid(const cuars::ConsensusTranslationEstimator2d::Grid &grid, const cuars::Vec2d &translMin, double translRes, const std::string &filename, double factor)
{
    int dim0 = grid.dimensions()[0];
    int dim1 = grid.dimensions()[1];
    int dim0Reduced = round(dim0 / factor);
    int dim1Reduced = round(dim1 / factor);
    double v0, v1;

    std::ofstream file(filename);
    if (!file)
    {
        std::cerr << "Cannot open \"" << filename << "\"" << std::endl;
        return;
    }

    std::cout << "plotting grid with size " << dim0 << " x " << dim1 << std::endl;

    file << "set grid nopolar\n"
         << "set style data lines\n"
         << "set dgrid3d " << dim0Reduced << "," << dim1Reduced << "\n"
         << "set hidden3d\n";

    file << "splot '-'\n";
    for (int i0 = 0; i0 < dim0; ++i0)
    {
        for (int i1 = 0; i1 < dim1; ++i1)
        {
            // v0 = translMin(0) + translRes * i0;
            v0 = translMin.x + translRes * i0;
            // v1 = translMin(1) + translRes * i1;
            v1 = translMin.y + translRes * i1;

            file << v0 << " " << v1 << " " << grid.value({i0, i1}) << "\n";
        }
    }
    file << "e\n";

    file.close();
}

void gpu_estimateRotationArsIso(const ArsImgTests::PointReaderWriter &pointsSrc, const ArsImgTests::PointReaderWriter &pointsDst, TestParams &testParams, ParlArsIsoParams &paip, double &rotOut)
{
    // ARS SRC -> preparation for kernel calls and kernel calls
    cudaEvent_t startSrc, stopSrc; // timing using CUDA events
    cudaEventCreate(&startSrc);
    cudaEventCreate(&stopSrc);

    const cuars::VecVec2d &inputSrc = pointsSrc.points();
    initParallelizationParams(paip, testParams.aiPms.arsIsoOrder, inputSrc.size(), paip.blockSz, paip.chunkMaxSz); // cudarsIso.init()
    double *coeffsArsSrc = new double[paip.coeffsMatNumColsPadded];
    computeArsIsoGpu(paip, testParams.aiPms, inputSrc, coeffsArsSrc, startSrc, stopSrc, paip.gpu_srcExecTime); // cudarsIso.compute()

    cudaEventDestroy(startSrc);
    cudaEventDestroy(stopSrc);
    // END OF ARS SRC

    //    std::cout << "\n------\n" << std::endl; //"pause" between ars src and ars dst

    // ARS DST -> preparation for kernel calls and kernel calls
    cudaEvent_t startDst, stopDst; // timing using CUDA events
    cudaEventCreate(&startDst);
    cudaEventCreate(&stopDst);

    const cuars::VecVec2d &inputDst = pointsDst.points();
    initParallelizationParams(paip, testParams.aiPms.arsIsoOrder, inputDst.size(), paip.blockSz, paip.chunkMaxSz); // cudarsIso.init()
    double *coeffsArsDst = new double[paip.coeffsMatNumColsPadded];
    computeArsIsoGpu(paip, testParams.aiPms, inputDst, coeffsArsDst, startDst, stopDst, paip.gpu_dstExecTime); // cudarsIso.compute()

    cudaEventDestroy(startDst);
    cudaEventDestroy(stopDst);
    // END OF ARS DST

    // Final computations (correlation, ...) on CPU
    //     std::cout << "\nARS Coefficients:\n";
    //     std::cout << "Coefficients: Src, Dst, Cor" << std::endl;

    double thetaMax, corrMax, fourierTol;
    fourierTol = 1.0; // TODO: check for a proper tolerance

    std::vector<double> coeffsCor;
    {
        cuars::ScopedTimer("ars.correlation()");
        std::vector<double> tmpSrc;
        tmpSrc.assign(coeffsArsSrc, coeffsArsSrc + paip.coeffsMatNumColsPadded);
        std::vector<double> tmpDst;
        tmpDst.assign(coeffsArsDst, coeffsArsDst + paip.coeffsMatNumColsPadded);
        cuars::computeFourierCorr(tmpSrc, tmpDst, coeffsCor);
        cuars::findGlobalMaxBBFourier(coeffsCor, 0.0, M_PI, testParams.aiPms.arsIsoThetaToll, fourierTol, thetaMax, corrMax);
        rotOut = thetaMax; //!! rotOut is passed to the function as reference
    }

    // Free CPU memory
    delete coeffsArsSrc;
    delete coeffsArsDst;
}

void computeArsTec(cuars::VecVec2d &translCandidates, const cuars::VecVec2d &pointsSrc, const cuars::VecVec2d &pointsDst, cuars::ArsTecParams &translParams)
{
    cuars::ArsTec<cuars::Grid2d, cuars::Indices2d, cuars::PeakFinder2d, 2> translObj(translParams); //ArsTec 2D object

    // translEstim.init(translMin, translRes, gridSize);
    // translEstim.setNonMaximaWindowDim(gridWin);    
    translObj.init(translParams.gridSize, translParams.gridWin);

    std::cout << "Inserting pair source-destination:\n";
    // translEstim.insert(pointsSrc, pointsDst);
    translObj.insert(pointsSrc, pointsDst, translParams.adaptiveGrid); // adaptive = false for the dummy example

    // if (translParams.plot)
    // {
    //     cuars::ConsensusTranslationEstimator2d translEstimPlot(grid, pf, translParams.translMin, translParams.translRes, translParams.gridSize);
    //     plotGrid(translEstimPlot.getGrid(), translParams.translMin, translParams.translRes, "consensus_transl_grid.plot", 1.0);
    // }

    std::cout << "Computing maxima:\n";
    // translEstim.computeMaxima(translCandidates); //TODO: adapt computeMaxima() for CUDA GPU parallelization
    // cuars::computeMaxima<cuars::Grid2d, cuars::Indices2d, cuars::PeakFinder2d, 2>(translCandidates, grid, peakF, translMin, translRes);
    translObj.computeMaxima(translCandidates); // TODO: adapt computeMaxima() for CUDA GPU parallelization
}