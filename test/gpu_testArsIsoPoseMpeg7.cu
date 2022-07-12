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

#include <iostream>
#include <chrono>

#include "cudars/Profiler.h"

#include "cudars/ars2d.cuh"
#include "cudars/ars2d.h"

#include "cudars/ConsensusTranslationEstimator.cuh"

#include "cudars/tls_scalar_consensus.cuh"

#include "cudars/mpeg7RW.h"

int main(int argc, char **argv)
{
    cudars::AngularRadonSpectrum2d arsSrc;
    cudars::AngularRadonSpectrum2d arsDst;
    ArsImgTests::PointReaderWriter pointsSrc;
    ArsImgTests::PointReaderWriter pointsDst;

    TestParams testParams;
    ParlArsIsoParams paiParams;
    cudars::ArsTec2dParams translParams;

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

    double rotTrue, rotArs;
    // cudars::VecVec2d translCandidates;
    cudars::Vec2d translTrue, translArs;

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
    //    params.getParam<double>("sampleResDeg", sampleRes, 0.5);
    //    sampleRes *= M_PI / 180.0;
    //    params.getParam<bool>("saveOn", saveOn, false);
    //    params.getParam<bool>("saveCov", saveCov, false);

    params.getParam<bool>("extrainfoEnable", testParams.extrainfoEnable, bool(true));

    // ArsIso params (CPU and GPU)
    params.getParam<bool>("arsisoEnable", testParams.arsIsoEnable, false);
    params.getParam<bool>("gpu_arsisoEnable", testParams.gpu_arsIsoEnable, true);
    params.getParam<int>("arsisoOrder", testParams.aiPms.arsIsoOrder, 32);
    params.getParam<double>("arsisoSigma", testParams.aiPms.arsIsoSigma, 1.0);
    params.getParam<double>("arsisoTollDeg", testParams.aiPms.arsIsoThetaToll, 0.5);
    testParams.aiPms.arsIsoThetaToll *= M_PI / 180.0;
    //    params.getParam<unsigned int>("arsisoPnebiMode", tp.arsIsoPnebiMode, cudars::ArsKernelIsotropic2d::ComputeMode::PNEBI_DOWNWARD);

    arsSrc.setARSFOrder(testParams.aiPms.arsIsoOrder);
    //    arsSrc.initLUT(0.0001);
    //    arsSrc.setComputeMode(ars::ArsKernelIsotropic2d::ComputeMode::PNEBI_LUT);
    arsSrc.setComputeMode(cudars::ArsKernelIso2dComputeMode::PNEBI_DOWNWARD);
    arsDst.setARSFOrder(testParams.aiPms.arsIsoOrder);
    arsDst.setComputeMode(cudars::ArsKernelIso2dComputeMode::PNEBI_DOWNWARD);

    // parallelization parameters
    params.getParam<int>("blockSz", paiParams.blockSz, 256);
    params.getParam<int>("chunkMaxSz", paiParams.chunkMaxSz, 4096);

    std::string translModeStr;
    params.getParam<std::string>("translMode", translModeStr, "tls");
    params.getParam<double>("translTlsRange", translParams.translTlsRange, 5.0);
    cudars::setupTranslMode(translParams.translMode, translModeStr);
    params.getParam<double>("translRes", translParams.translRes, 3.0);
    // params.getParamContainer("translMin", translMin.data(), translMin.data() + translMin.size(), "[-10.0,-10.0]", double(0.0), "[,]"); //TODO: adapt ParamContainer to Cuda types
    params.getParam<double>("translMin-x", translParams.translMin.x, 0);
    params.getParam<double>("translMin-y", translParams.translMin.y, 0);
    params.getParam<double>("translMax-x", translParams.translMax.x, 1000.0);
    params.getParam<double>("translMax-y", translParams.translMax.y, 1000.0);
    // params.getParamContainer("translGt", translGt.data(), translGt.data() + translGt.size(), "[-4.2,5.0]", double(1.0), "[,]");
    // params.getParam<double>("translGt-x", translParams.translGt.x, -4.2); //OSS. translGt is made useless
    // params.getParam<double>("translGt-y", translParams.translGt.y, 5.0);
    params.getParamContainer("gridSize", translParams.gridSize.data(), translParams.gridSize.data() + translParams.gridSize.size(), "[200,200]", int(0), "[,]");
    params.getParamContainer("gridWin", translParams.gridWin.data(), translParams.gridWin.data() + translParams.gridWin.size(), "[5,5]", int(1), "[,]");
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

    int numPts = std::min<int>(pointsSrc.points().size(), pointsDst.points().size()); // the two should normally be equal
    //    int numPtsAfterPadding = numPts;

    // ARS parameters setting
    arsSrc.setARSFOrder(testParams.aiPms.arsIsoOrder);
    arsDst.setARSFOrder(testParams.aiPms.arsIsoOrder);
    cudars::ArsKernelIso2dComputeMode pnebiMode = cudars::ArsKernelIso2dComputeMode::PNEBI_DOWNWARD;
    arsSrc.setComputeMode(pnebiMode);
    arsDst.setComputeMode(pnebiMode);

    std::cout << "\n------\n"
              << std::endl;

    std::cout << "\n\nCalling kernel functions on GPU\n"
              << std::endl;

    // Eigen::Affine2d transfSrcToDst = pointsDst.getTransform() * pointsSrc.getTransform().inverse();
    cudars::Affine2d transfSrcToDst;
    aff2Prod(transfSrcToDst, pointsDst.getTransform(), pointsSrc.getTransform().inverse());
    std::cout << "transfSrcToDst" << std::endl
              << transfSrcToDst << std::endl;
    translTrue = transfSrcToDst.translation();

    gpu_estimateRotationArsIso(pointsSrc, pointsDst, testParams, paiParams, rotArs);

    rotTrue = pointsDst.getRotTheta() - pointsSrc.getRotTheta();
    std::cout << "\n***\npointsDst.getrotTheta() [deg]" << (180 / M_PI * pointsDst.getRotTheta())
              << ", pointsSrc.getrotTheta() [deg] " << (180.0 / M_PI * pointsSrc.getRotTheta()) << "\n";
    std::cout << "rotTrue[deg] \t" << (180.0 / M_PI * rotTrue) << " \t" << (180.0 / M_PI * cudars::mod180(rotTrue)) << std::endl;
    std::cout << "rotArs[deg] \t" << (180.0 / M_PI * rotArs) << " \t" << (180.0 / M_PI * cudars::mod180(rotArs)) << std::endl;

    if (translParams.translMode == cudars::TranslMode::GRID)
        cudars::computeArsTec2d(translArs, rotArs, pointsSrc, pointsDst, translParams);
    else if (translParams.translMode == cudars::TranslMode::TLS)
        cudars::estimateTranslTls(translArs, rotArs, pointsSrc, pointsDst, translParams);

    // std::cout << "translTrue:" << std::endl;
    cudars::printVec2d(translTrue, "translTrue");

    // std::cout << "translArs:" << std::endl;
    cudars::printVec2d(translArs, "translArs");

    return 0;
}
