
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

#include <ars/ars2d.h>

#include "ars/ars2d.cuh"
#include "ars/mpeg7RW.h"

int main(int argc, char **argv) {
    cuars::AngularRadonSpectrum2d arsSrc;
    cuars::AngularRadonSpectrum2d arsDst;
    ArsImgTests::PointReaderWriter pointsSrc;
    ArsImgTests::PointReaderWriter pointsDst;


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

    //    int arsOrder;
    //    double arsSigma, arsThetaToll;
    //    int blockSz, chunkMaxSz;
    ArsIsoParams aiPms;
    ParlArsIsoParams paip;

    double rotTrue, rotArs;


    params.read(argc, argv);
    params.getParam<std::string>("cfg", filenameCfg, "");
    std::cout << "config filename: " << filenameCfg << std::endl;
    if (filenameCfg != "") {
        params.read(filenameCfg);
    }

    params.read(argc, argv);
    params.getParam<std::string>("src", filenameSrc, "/home/rimlab/Downloads/acld0000_global.pcd");
    params.getParam<std::string>("dst", filenameDst, "/home/rimlab/Downloads/acld0001_global.pcd");
    params.getParam<int>("arsOrder", aiPms.arsIsoOrder, 20);
    params.getParam<double>("arsSigma", aiPms.arsIsoSigma, 0.05);
    params.getParam<double>("arsTollDeg", aiPms.arsIsoThetaToll, 0.5);
    aiPms.arsIsoThetaToll *= M_PI / 180.0;

    params.getParam<int>("blockSz", paip.blockSz, 256);
    params.getParam<int>("chunkMaxSz", paip.chunkMaxSz, 4096);


    std::cout << "\nParameter values:\n";
    params.write(std::cout);
    std::cout << std::endl;


    // Loads files and computes the rotation
    std::cout << "\n*****\nLoading file \"" << filenameSrc << "\"" << std::endl;
    pointsSrc.loadPcdAscii(filenameSrc);
    std::cout << "\n*****\nLoading file \"" << filenameDst << "\"" << std::endl;
    pointsDst.loadPcdAscii(filenameDst);
    std::cout << "  points src " << pointsSrc.points().size() << ", points dst " << pointsDst.points().size() << std::endl;

    int numPts = std::max<int>(pointsSrc.points().size(), pointsDst.points().size());
    //    int numPtsAfterPadding = numPts;

    //ARS parameters setting
    arsSrc.setARSFOrder(aiPms.arsIsoOrder);
    arsDst.setARSFOrder(aiPms.arsIsoOrder);
    cuars::ArsKernelIso2dComputeMode pnebiMode = cuars::ArsKernelIso2dComputeMode::PNEBI_DOWNWARD;
    arsSrc.setComputeMode(pnebiMode);
    arsDst.setComputeMode(pnebiMode);


    std::cout << "\n------\n" << std::endl;


    //    aiPms.arsIsoOrder = arsOrder;
    //    aiPms.arsIsoPnebiMode = pnebiMode;
    //    aiPms.arsIsoSigma = arsSigma;
    //    aiPms.arsIsoThetaToll = arsThetaToll;
    //
    //    paip.blockSz = blockSz;
    //    paip.chunkMaxSz = chunkMaxSz;


    //ARS SRC -> preparation for kernel calls and kernel calls
    cudaEvent_t startSrc, stopSrc; //timing using CUDA events
    cudaEventCreate(&startSrc);
    cudaEventCreate(&stopSrc);

    const cuars::VecVec2d& inputSrc = pointsSrc.points();
    initParallelizationParams(paip, aiPms.arsIsoOrder, inputSrc.size(), paip.blockSz, paip.chunkMaxSz); //cudarsIso.init()
    double* coeffsArsSrc = new double [paip.coeffsMatNumColsPadded];
    computeArsIsoGpu(paip, aiPms, inputSrc, coeffsArsSrc, startSrc, stopSrc, paip.gpu_srcExecTime); //cudarsIso.compute()

    cudaEventDestroy(startSrc);
    cudaEventDestroy(stopSrc);
    //END OF ARS SRC


    std::cout << "\n------\n" << std::endl; //"pause" between ars src and ars dst


    //ARS DST -> preparation for kernel calls and kernel calls
    cudaEvent_t startDst, stopDst; //timing using CUDA events
    cudaEventCreate(&startDst);
    cudaEventCreate(&stopDst);

    const cuars::VecVec2d& inputDst = pointsDst.points();
    initParallelizationParams(paip, aiPms.arsIsoOrder, inputDst.size(), paip.blockSz, paip.chunkMaxSz); //cudarsIso.init()
    double* coeffsArsDst = new double [paip.coeffsMatNumColsPadded];
    computeArsIsoGpu(paip, aiPms, inputDst, coeffsArsDst, startDst, stopDst, paip.gpu_srcExecTime); //cudarsIso.compute()

    cudaEventDestroy(startDst);
    cudaEventDestroy(stopDst);
    //END OF ARS DST


    //Computation final computations (correlation, ...) on CPU
    std::cout << "\nARS Coefficients:\n";
    std::cout << "Coefficients: Src, Dst, Cor" << std::endl;

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
        cuars::findGlobalMaxBBFourier(coeffsCor, 0.0, M_PI, aiPms.arsIsoThetaToll, fourierTol, thetaMax, corrMax);
        rotArs = thetaMax;
    }



    arsSrc.setCoefficients(coeffsArsSrc, paip.coeffsMatNumCols);
    //    for (int i = 0; i < coeffsVectorMaxSz; i++) {
    //        std::cout << "arsSrc - coeff_d[" << i << "] " << d_coeffsMat1[i] << std::endl;
    //    }
    arsDst.setCoefficients(coeffsArsDst, paip.coeffsMatNumCols);
    for (int i = 0; i < arsSrc.coefficients().size() && i < arsDst.coefficients().size(); ++i) {
        std::cout << "\t" << i << " \t" << arsSrc.coefficients().at(i) << " \t" << arsDst.coefficients().at(i) << " \t" << coeffsCor[i] << std::endl;
    }
    std::cout << std::endl;



    // Computes the rotated points,centroid, affine transf matrix between src and dst
    ArsImgTests::PointReaderWriter pointsRot(pointsSrc.points());
    cuars::Vec2d centroidSrc = pointsSrc.computeCentroid();
    cuars::Vec2d centroidDst = pointsDst.computeCentroid();
    cuars::Affine2d rotSrcDst = ArsImgTests::PointReaderWriter::coordToTransform(0.0, 0.0, rotArs);
    //    cuars::Vec2d translSrcDst = centroidDst - rotSrcDst * centroidSrc;
    cuars::Vec2d translSrcDst;
    cuars::vec2diff(translSrcDst, centroidDst, cuars::aff2TimesVec2WRV(rotSrcDst, centroidSrc));
    //    std::cout << "centroidSrc " << centroidSrc.transpose() << "\n"
    //            << "rotSrcDst\n" << rotSrcDst.matrix() << "\n"
    //            << "translation: [" << translSrcDst.transpose() << "] rotation[deg] " << (180.0 / M_PI * rotArs) << "\n";
    std::cout << "centroidSrc " << centroidSrc.x << " \t" << centroidSrc.y << "\n"
            << "centroidDst " << centroidDst.x << " \t" << centroidDst.y << "\n"
            << "rotSrcDst\n" << rotSrcDst << "\n"
            << "translation: [" << translSrcDst.x << " \t" << translSrcDst.y << "] rotation[deg] " << (180.0 / M_PI * rotArs) << "\n";
    pointsRot.applyTransform(translSrcDst.x, translSrcDst.y, rotArs);



    rotTrue = pointsDst.getRotTheta() - pointsSrc.getRotTheta();
    std::cout << "\n***\npointsDst.getrotTheta() [deg]" << (180 / M_PI * pointsDst.getRotTheta())
            << ", pointsSrc.getrotTheta() [deg] " << (180.0 / M_PI * pointsSrc.getRotTheta()) << "\n";
    std::cout << "rotTrue[deg] \t" << (180.0 / M_PI * rotTrue) << " \t" << (180.0 / M_PI * cuars::mod180(rotTrue)) << std::endl;
    std::cout << "rotArs[deg] \t" << (180.0 / M_PI * rotArs) << " \t" << (180.0 / M_PI * cuars::mod180(rotArs)) << std::endl;

    //Free CPU memory
    delete coeffsArsSrc;
    delete coeffsArsDst;


    return 0;
}





