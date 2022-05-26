
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

#include "ars/ConsensusTranslationEstimator.cuh"

#include "ars/mpeg7RW.h"

void plotGrid2d(const cuars::ArsTec<cuars::Grid2d, cuars::Indices2d, cuars::PeakFinder2d, 2> &arsTec, const cuars::Vec2d &translMin, double translRes, const std::string &filename, double factor);

int main(int argc, char **argv)
{
    cuars::AngularRadonSpectrum2d arsSrc;
    cuars::AngularRadonSpectrum2d arsDst;
    ArsImgTests::PointReaderWriter pointsSrc;
    ArsImgTests::PointReaderWriter pointsDst;

    TestParams testParams;
    ParlArsIsoParams paiParams;
    cuars::ArsTec2dParams translParams;

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
    // cuars::VecVec2d translCandidates;
    cuars::Vec2d translTrue, translArs;

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
    params.getParam<int>("arsisoOrder", testParams.aiPms.arsIsoOrder, 32);
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

    // // Eigen::Affine2d transfSrcToDst = pointsDst.getTransform() * pointsSrc.getTransform().inverse();
    cuars::Affine2d transfSrcToDst;
    aff2Prod(transfSrcToDst, pointsDst.getTransform(), pointsSrc.getTransform().inverse());
    std::cout << "diff transform" << std::endl
              << transfSrcToDst << std::endl;
    translTrue = transfSrcToDst.translation();

    cuars::computeArsTec2d(translArs, rotArs, pointsSrc, pointsDst, translParams);

    // std::cout << "translTrue:" << std::endl;
    cuars::printVec2d(translTrue, "translTrue");

    // std::cout << "translArs:" << std::endl;
    cuars::printVec2d(translArs, "translArs");

    return 0;
}
