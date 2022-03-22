
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

#include "ars/mpeg7RW.h"
#include "ars/mpeg7_io.h"
#include "ars/ars2d.cuh"
#include <ars/ars2d.h>




void gpu_estimateRotationArsIso(const ArsImgTests::PointReaderWriter& pointsSrc, const ArsImgTests::PointReaderWriter& pointsDst, TestParams& tp, ParlArsIsoParams& paip, double& rotOut);

int main(int argc, char **argv) {
    cuars::AngularRadonSpectrum2d arsSrc;
    cuars::AngularRadonSpectrum2d arsDst;
    ArsImgTests::PointReaderWriter pointsSrc;
    ArsImgTests::PointReaderWriter pointsDst;
    TestParams tparams;
    ParlArsIsoParams paiParams;


    rofl::ParamMap params;
    std::string inputGlob;

    std::string filenameCfg;


    //The variables below are for I/O related functionalities (plot, etc.) that are highly Eigen-based and are present in the CPU-only ArsImgTests...
    //Maybe implement them later
    //    double sampleRes, sampleAng; 
    //    int sampleNum;
    //    bool saveOn;
    //    bool saveCov;
    std::string resumeFilename;
    std::string filenameOut;

    std::string prefixName;
    double rotTrue, rotArsIso, rotArsIso_gpu;
    int srcNumPts, dstNumPts;


    params.read(argc, argv);
    params.getParam<std::string>("cfg", filenameCfg, "");
    std::cout << "config filename: " << filenameCfg << std::endl;
    if (filenameCfg != "") {
        params.read(filenameCfg);
    }

    params.read(argc, argv);
    params.getParam<std::string>("in", inputGlob, std::experimental::filesystem::current_path().string() + "/*");
    params.getParam<std::string>("out", filenameOut, mpeg7io::generateStampedString("results_", ".txt"));
    params.getParam<std::string>("resume", resumeFilename, "");
    params.getParam<bool>("extrainfoEnable", tparams.extrainfoEnable, bool(true));


    // ArsIso params (CPU and GPU)
    params.getParam<bool>("arsisoEnable", tparams.arsIsoEnable, false);
    params.getParam<bool>("gpu_arsisoEnable", tparams.gpu_arsIsoEnable, true);
    params.getParam<int>("arsisoOrder", tparams.aiPms.arsIsoOrder, 20);
    params.getParam<double>("arsisoSigma", tparams.aiPms.arsIsoSigma, 1.0);
    params.getParam<double>("arsisoTollDeg", tparams.aiPms.arsIsoThetaToll, 0.5);
    tparams.aiPms.arsIsoThetaToll *= M_PI / 180.0;
    //    params.getParam<unsigned int>("arsisoPnebiMode", tparams.arsIsoPnebiMode, cuars::ArsKernelIsotropic2d::ComputeMode::PNEBI_DOWNWARD);


    arsSrc.setARSFOrder(tparams.aiPms.arsIsoOrder);
    //    arsSrc.initLUT(0.0001);
    //    arsSrc.setComputeMode(ars::ArsKernelIsotropic2d::ComputeMode::PNEBI_LUT);
    arsSrc.setComputeMode(cuars::ArsKernelIso2dComputeMode::PNEBI_DOWNWARD);
    arsDst.setARSFOrder(tparams.aiPms.arsIsoOrder);
    arsDst.setComputeMode(cuars::ArsKernelIso2dComputeMode::PNEBI_DOWNWARD);


    //parallelization parameters
    params.getParam<int>("blockSz", paiParams.blockSz, 256);
    params.getParam<int>("chunkMaxSz", paiParams.chunkMaxSz, 4096);

    params.getParam<int>("fileSkipper", tparams.fileSkipper, int(1));

    std::cout << "\nParameter values:\n";
    params.write(std::cout);
    std::cout << std::endl;


    /* Reading files from folder */
    std::vector<std::string> inputFilenames;
    std::vector<std::pair<int, int> > allPairs;
    std::vector<std::pair<int, int> > outPairs;

    mpeg7io::getDirectoryFiles(inputGlob, inputFilenames);
    std::cout << "\nFilenames:\n";
    size_t numFiles = 0;
    for (auto& filename : inputFilenames) {
        if (numFiles < 30)
            std::cout << "  " << mpeg7io::getPrefix(filename) << " " << mpeg7io::getShortName(filename) << " " << filename << "\n";
        else if (numFiles == 30)
            std::cout << "..." << std::endl;

        numFiles++;
    }
    std::cout << std::endl;



    if (!inputFilenames.empty()) {
        std::string leafDir = mpeg7io::getLeafDirectory(inputFilenames[0]);
        std::cout << "leafDir: \"" << leafDir << "\"" << std::endl;
        std::string methodSuffix;
        if (tparams.arsIsoEnable) {
            methodSuffix = methodSuffix + "_arsiso";
        }
        if (tparams.gpu_arsIsoEnable) {
            methodSuffix = methodSuffix + "_gpuarsiso";
        }
        if (tparams.extrainfoEnable) {
            methodSuffix = methodSuffix + "_extrainfo";
        }
        filenameOut = mpeg7io::generateStampedString("results_" + leafDir + methodSuffix + "_", ".txt");
        std::cout << "outputFilename: \"" << filenameOut << "\"" << std::endl;

    }

    // Open output results file
    std::ofstream outfile(filenameOut.c_str());
    if (!outfile) {
        std::cerr << "Cannot open file \"" << filenameOut << "\"" << std::endl;
        return -1;
    }

    mpeg7io::findComparisonPair(inputFilenames, allPairs);
    std::cout << "Processing " << inputFilenames.size() << " files, " << allPairs.size() << " comparisons\n" << std::endl;
    mpeg7io::filterComparisonPair(resumeFilename, outfile, inputFilenames, allPairs, outPairs);
    std::cout << "Remaining comparisons " << outPairs.size() << " comparisons\n" << std::endl;

    outfile << "# Parameters:\n";
    params.write(outfile, "#  ");
    outfile << "# \n";
    outfile << "# file1 numpts1 noise1 occl1 rand1 file2 numpts2 noise2 occl2 rand2 rotTrue rotTrue[deg] ";

    if (tparams.arsIsoEnable) {
        outfile << "arsIso rotArsIso[deg] ";
    }
    if (tparams.gpu_arsIsoEnable) {
        outfile << "gpuarsIso rotGpuArsIso[deg] ";
    }
    if (tparams.extrainfoEnable)
        outfile << "srcNumPts srcNumKers srcExecTime dstNumPts dstNumKers dstExecTime "; //Kers stands for kernels

    outfile << "\n";
    //End of outfile header setup


    //execution couple-by-couple (of files) of ARS
    int countPairs = 0;
    for (auto& comp : outPairs) {
        if (countPairs % tparams.fileSkipper) {
            countPairs++;
            continue;
        }
        pointsSrc.load(inputFilenames[comp.first]);
        pointsDst.load(inputFilenames[comp.second]);
        prefixName = mpeg7io::getPrefix(inputFilenames[comp.first]);


        std::cout << "[" << countPairs << "/" << outPairs.size() << "]\n" << "  * \"" << inputFilenames[comp.first] << "\"\n    \"" << inputFilenames[comp.second] << "\"" << std::endl;
        rotTrue = pointsDst.getRotTheta() - pointsSrc.getRotTheta();

        //    if (rotTrue < 0.0) rotTrue += M_PI;
        //    else if (rotTrue > M_PI) rotTrue -= M_PI;
        std::cout << " angle dst " << (180.0 / M_PI * pointsDst.getRotTheta()) << " [deg], src " << (180.0 / M_PI * pointsSrc.getRotTheta()) << " [deg]" << std::endl;
        std::cout << std::fixed << std::setprecision(2) << std::setw(10)
                << "  rotTrue \t\t" << (180.0 / M_PI * rotTrue) << " deg\t\t" << (180.0 / M_PI * cuars::mod180(rotTrue)) << " deg [mod 180]\n" << std::endl;

        outfile
                << std::setw(20) << mpeg7io::getShortName(inputFilenames[comp.first]) << " "
                << std::setw(6) << pointsSrc.getNumIn() << " "
                << std::fixed << std::setprecision(1) << std::setw(6) << pointsSrc.getNoiseSigma() << " "
                << std::setw(6) << pointsSrc.getNumOccl() << " "
                << std::setw(6) << pointsSrc.getNumRand() << " "
                << std::setw(20) << mpeg7io::getShortName(inputFilenames[comp.second]) << " "
                << std::setw(6) << pointsSrc.getNumIn() << " "
                << std::fixed << std::setprecision(1) << std::setw(6) << pointsDst.getNoiseSigma() << " "
                << std::setw(6) << pointsDst.getNumOccl() << " "
                << std::setw(6) << pointsDst.getNumRand() << " "
                << "rotTrue" << std::fixed << std::setprecision(2) << std::setw(8) << (180.0 / M_PI * cuars::mod180(rotTrue)) << " ";


        if (tparams.arsIsoEnable) {
            //                    estimateRotationArsIso(pointsSrc.points(), pointsDst.points(), tparams, rotArsIso);
            std::cout << std::fixed << std::setprecision(2) << std::setw(10)
                    << "  rotArsIso \t\t" << (180.0 / M_PI * rotArsIso) << " deg\t\t" << (180.0 / M_PI * cuars::mod180(rotArsIso)) << " deg [mod 180]\n";
            outfile << std::setw(6) << "arsIso " << std::fixed << std::setprecision(2) << std::setw(6) << (180.0 / M_PI * cuars::mod180(rotArsIso)) << " ";
        }
        if (tparams.gpu_arsIsoEnable) {
            gpu_estimateRotationArsIso(pointsSrc.points(), pointsDst.points(), tparams, paiParams, rotArsIso_gpu);
            std::cout << std::fixed << std::setprecision(2) << std::setw(10)
                    << "  gpu_rotArsIso \t" << (180.0 / M_PI * rotArsIso_gpu) << " deg\t\t" << (180.0 / M_PI * cuars::mod180(rotArsIso_gpu)) << " deg [mod 180]\n";
            outfile << std::setw(6) << "gpu_arsIso " << std::fixed << std::setprecision(2) << std::setw(6) << (180.0 / M_PI * cuars::mod180(rotArsIso_gpu)) << " ";
        }
        if (tparams.extrainfoEnable) {
            srcNumPts = paiParams.numPts;
            dstNumPts = paiParams.numPts; //all couples on mpeg7 have the same number of points
            std::cout << std::fixed << std::setprecision(2) << std::setw(10)
                    << "  srcNumPts \t" << srcNumPts << "  srcExecTime \t" << paiParams.srcExecTime << "  gpu_srcExecTime \t" << paiParams.gpu_srcExecTime
                    << "  dstNumPts \t" << dstNumPts << "  dstExecTime \t" << paiParams.dstExecTime << "  gpu_dstExecTime \t" << paiParams.gpu_dstExecTime
                    << std::endl;

            outfile << std::setw(8) << "sPts " << std::fixed << std::setprecision(2) << std::setw(6) << srcNumPts << " "
                    << std::setw(8) << "sTm " << std::fixed << std::setprecision(2) << std::setw(6) << paiParams.srcExecTime << " "
                    << std::setw(8) << "gpu_sTm " << std::fixed << std::setprecision(2) << std::setw(12) << paiParams.gpu_srcExecTime << " "
                    << std::setw(8) << "dPts " << std::fixed << std::setprecision(2) << std::setw(6) << dstNumPts << " "
                    << std::setw(8) << "dTm " << std::fixed << std::setprecision(2) << std::setw(6) << paiParams.dstExecTime << " "
                    << std::setw(8) << "gpu_dTm " << std::fixed << std::setprecision(2) << std::setw(12) << paiParams.gpu_dstExecTime << " ";
        }

        outfile << std::endl;


        countPairs++;
        std::cout << "\n\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n";
    }

    //    std::cout << "maxNumPts " << maxNumPts << " maxNumPtsPadded " << maxNumPtsPadded << std::endl;

    outfile.close();

    return 0;
}



void gpu_estimateRotationArsIso(const ArsImgTests::PointReaderWriter& pointsSrc, const ArsImgTests::PointReaderWriter& pointsDst, TestParams& tp, ParlArsIsoParams& paip, double& rotOut) {
    //ARS SRC -> preparation for kernel calls and kernel calls
    cudaEvent_t startSrc, stopSrc; //timing using CUDA events
    cudaEventCreate(&startSrc);
    cudaEventCreate(&stopSrc);

    const cuars::VecVec2d& inputSrc = pointsSrc.points();
    double* coeffsArsSrc = new double [paip.coeffsMatNumColsPadded];
    initParallelizationParams(paip, tp.aiPms.arsIsoOrder, inputSrc.size(), paip.blockSz, paip.chunkMaxSz); //cudarsIso.init()
    computeArsIsoGpu(paip, tp.aiPms, inputSrc, coeffsArsSrc, startSrc, stopSrc); //cudarsIso.compute()

    cudaEventDestroy(startSrc);
    cudaEventDestroy(stopSrc);
    //END OF ARS SRC

    //    std::cout << "\n------\n" << std::endl; //"pause" between ars src and ars dst

    //ARS DST -> preparation for kernel calls and kernel calls
    cudaEvent_t startDst, stopDst; //timing using CUDA events
    cudaEventCreate(&startDst);
    cudaEventCreate(&stopDst);

    const cuars::VecVec2d& inputDst = pointsDst.points();
    double* coeffsArsDst = new double [paip.coeffsMatNumColsPadded];
    initParallelizationParams(paip, tp.aiPms.arsIsoOrder, inputDst.size(), paip.blockSz, paip.chunkMaxSz); //cudarsIso.init()
    computeArsIsoGpu(paip, tp.aiPms, inputDst, coeffsArsDst, startDst, stopDst); //cudarsIso.compute()

    cudaEventDestroy(startDst);
    cudaEventDestroy(stopDst);
    //END OF ARS DST


    //Final computations (correlation, ...) on CPU
    //    std::cout << "\nARS Coefficients:\n";
    //    std::cout << "Coefficients: Src, Dst, Cor" << std::endl;

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
        cuars::findGlobalMaxBBFourier(coeffsCor, 0.0, M_PI, tp.aiPms.arsIsoThetaToll, fourierTol, thetaMax, corrMax);
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


    // Computes the rotated points,centroid, affine transf matrix between src and dst
    ArsImgTests::PointReaderWriter pointsRot(pointsSrc.points());
    cuars::Vec2d centroidSrc = pointsSrc.computeCentroid();
    cuars::Vec2d centroidDst = pointsDst.computeCentroid();
    cuars::Affine2d rotSrcDst = ArsImgTests::PointReaderWriter::coordToTransform(0.0, 0.0, rotOut);
    //    cuars::Vec2d translSrcDst = centroidDst - rotSrcDst * centroidSrc;
    cuars::Vec2d translSrcDst;
    cuars::vec2diff(translSrcDst, centroidDst, cuars::aff2TimesVec2WRV(rotSrcDst, centroidSrc));
    //    std::cout << "centroidSrc " << centroidSrc.x << " \t" << centroidSrc.y << "\n"
    //            << "centroidDst " << centroidDst.x << " \t" << centroidDst.y << "\n"
    //            << "rotSrcDst\n" << rotSrcDst << "\n"
    //            << "translation: [" << translSrcDst.x << " \t" << translSrcDst.y << "] rotation[deg] " << (180.0 / M_PI * rotOut) << "\n";
    pointsRot.applyTransform(translSrcDst.x, translSrcDst.y, rotOut);


    //    double rotTrue = pointsDst.getRotTheta() - pointsSrc.getRotTheta();
    //    std::cout << "\n***\npointsDst.getrotTheta() [deg]" << (180 / M_PI * pointsDst.getRotTheta())
    //            << ", pointsSrc.getrotTheta() [deg] " << (180.0 / M_PI * pointsSrc.getRotTheta()) << "\n";
    //    std::cout << "rotTrue[deg] \t" << (180.0 / M_PI * rotTrue) << " \t" << (180.0 / M_PI * cuars::mod180(rotTrue)) << std::endl;
    //    std::cout << "rotArs[deg] \t" << (180.0 / M_PI * rotOut) << " \t" << (180.0 / M_PI * cuars::mod180(rotOut)) << std::endl;

    //Free CPU memory
    delete coeffsArsSrc;
    delete coeffsArsDst;
}
