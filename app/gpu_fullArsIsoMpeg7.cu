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

#include "cudars/mpeg7RW.h"
#include "cudars/mpeg7_io.h"
#include "cudars/ars2d.cuh"
#include "cudars/ars2d.h"

namespace expfs = std::experimental::filesystem;

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
    params.getParam<std::string>("in", inputGlob, expfs::current_path().string() + "/*");
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

    params.getParam<int>("fileSkipper", tparams.fileSkipper, 1);

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
        outfile << "gpu_arsIso rotGpuArsIso[deg] ";
    }
    if (tparams.extrainfoEnable)
        outfile << "srcNumPts srcExecTime gpu_srcExecTime dstNumPts dstExecTime gpu_dstExecTime";


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
            srcNumPts = pointsSrc.points().size();
            dstNumPts = pointsDst.points().size();
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


