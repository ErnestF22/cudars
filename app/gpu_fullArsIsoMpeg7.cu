
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

#include <unordered_map>

#include "ars/Profiler.h"

#include "ars/mpeg7RW.h"
#include "ars/mpeg7_io.h"
#include "ars/ars2d.cuh"
#include <ars/ars2d.h>

struct TestParams {
    // ArsIso (Isotropic Angular Radon Spectrum) params
    bool arsIsoEnable;
    bool gpu_arsIsoEnable;
    int arsIsoOrder;
    double arsIsoSigma;
    double arsIsoThetaToll;
    cuars::ArsKernelIso2dComputeMode arsIsoPnebiMode;


    bool extrainfoEnable;
    int fileSkipper;
};

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



void findComparisonPair(const std::vector<std::string>& inputFilenames, std::vector<std::pair<int, int> >& comPairs);

void filterComparisonPair(std::string resumeFilename, std::ostream& outputfile,
        const std::vector<std::string>& inputFilenames, std::vector<std::pair<int, int> >& inputPairs,
        std::vector<std::pair<int, int> >& outputPairs);

std::string getPrefix(std::string filename);

std::string getShortName(std::string filename);

std::string getLeafDirectory(std::string filename);

void initParallelizationParams(ParlArsIsoParams& pp, int fourierOrder, int numPtsSrc, int numPtsDst, int blockSz, int chunkMaxSz);

void updateParallelizationParams(ParlArsIsoParams& pp, int currChunkSz);

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
    params.getParam<int>("arsisoOrder", tparams.arsIsoOrder, 20);
    params.getParam<double>("arsisoSigma", tparams.arsIsoSigma, 1.0);
    params.getParam<double>("arsisoTollDeg", tparams.arsIsoThetaToll, 0.5);
    tparams.arsIsoThetaToll *= M_PI / 180.0;
    //    params.getParam<unsigned int>("arsisoPnebiMode", tparams.arsIsoPnebiMode, cuars::ArsKernelIsotropic2d::ComputeMode::PNEBI_DOWNWARD);


    arsSrc.setARSFOrder(tparams.arsIsoOrder);
    //    arsSrc.initLUT(0.0001);
    //    arsSrc.setComputeMode(ars::ArsKernelIsotropic2d::ComputeMode::PNEBI_LUT);
    arsSrc.setComputeMode(cuars::ArsKernelIso2dComputeMode::PNEBI_DOWNWARD);
    arsDst.setARSFOrder(tparams.arsIsoOrder);
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
            std::cout << "  " << getPrefix(filename) << " " << getShortName(filename) << " " << filename << "\n";
        else if (numFiles == 30)
            std::cout << "..." << std::endl;

        numFiles++;
    }
    std::cout << std::endl;




    if (!inputFilenames.empty()) {
        std::string leafDir = getLeafDirectory(inputFilenames[0]);
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


    findComparisonPair(inputFilenames, allPairs);
    std::cout << "Processing " << inputFilenames.size() << " files, " << allPairs.size() << " comparisons\n" << std::endl;
    filterComparisonPair(resumeFilename, outfile, inputFilenames, allPairs, outPairs);
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
        prefixName = getPrefix(inputFilenames[comp.first]);


        std::cout << "[" << countPairs << "/" << outPairs.size() << "]\n" << "  * \"" << inputFilenames[comp.first] << "\"\n    \"" << inputFilenames[comp.second] << "\"" << std::endl;
        rotTrue = pointsDst.getRotTheta() - pointsSrc.getRotTheta();

        //    if (rotTrue < 0.0) rotTrue += M_PI;
        //    else if (rotTrue > M_PI) rotTrue -= M_PI;
        std::cout << " angle dst " << (180.0 / M_PI * pointsDst.getRotTheta()) << " [deg], src " << (180.0 / M_PI * pointsSrc.getRotTheta()) << " [deg]" << std::endl;
        std::cout << std::fixed << std::setprecision(2) << std::setw(10)
                << "  rotTrue \t\t" << (180.0 / M_PI * rotTrue) << " deg\t\t" << (180.0 / M_PI * cuars::mod180(rotTrue)) << " deg [mod 180]\n" << std::endl;

        outfile
                << std::setw(20) << getShortName(inputFilenames[comp.first]) << " "
                << std::setw(6) << pointsSrc.getNumIn() << " "
                << std::fixed << std::setprecision(1) << std::setw(6) << pointsSrc.getNoiseSigma() << " "
                << std::setw(6) << pointsSrc.getNumOccl() << " "
                << std::setw(6) << pointsSrc.getNumRand() << " "
                << std::setw(20) << getShortName(inputFilenames[comp.second]) << " "
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

void findComparisonPair(const std::vector<std::string>& inputFilenames, std::vector<std::pair<int, int> >& comPairs) {
    std::string prefix;
    int idx1, idx2;

    idx1 = 0;
    while (idx1 < inputFilenames.size()) {
        // Finds the prefix of inputFilenames[idx1] and finds adjacent filenames 
        // with the same prefix 
        prefix = getPrefix(inputFilenames[idx1]);
        idx2 = idx1 + 1;
        while (idx2 < inputFilenames.size() && getPrefix(inputFilenames[idx2]) == prefix) {
            idx2++;
        }
        // Computes all index pairs
        //        std::cout << "Group \"" << prefix << "\" with " << (idx2 - idx1) << " items: ";
        for (int i1 = idx1; i1 < idx2; ++i1) {
            //            std::cout << "\"" << getShortName(inputFilenames[i1]) << "\" [" << i1 << "], ";
            for (int i2 = i1 + 1; i2 < idx2; ++i2) {
                comPairs.push_back(std::make_pair(i1, i2));
            }
        }
        //        std::cout << "\n";
        idx1 = idx2;
    }
}

// Reads outputFilename for the list of already processed files

void filterComparisonPair(std::string resumeFilename, std::ostream& outputFile,
        const std::vector<std::string>& inputFilenames, std::vector<std::pair<int, int> >& inputPairs,
        std::vector<std::pair<int, int> >& outputPairs) {
    std::unordered_multimap<std::string, int> indicesMap;
    std::vector<std::pair<int, int> > visitedPairs;
    std::string filenameShort, line, label1, label2;
    int numIn1, numOccl1, numRand1, i1, i2;

    outputPairs.clear();
    // Visits all the lines/items of the output file
    for (int i = 0; i < inputFilenames.size(); ++i) {
        filenameShort = getShortName(inputFilenames[i]);
        indicesMap.insert(std::make_pair(filenameShort, i));
    }

    // Finds all the pairs already visited
    std::ifstream resumeFile(resumeFilename.c_str());
    if (!resumeFile) {
        std::cerr << "Cannot open file \"" << resumeFilename << "\": nothing to resume" << std::endl;
        outputPairs.insert(outputPairs.begin(), inputPairs.begin(), inputPairs.end());
        return;
    }
    while (!resumeFile.eof()) {
        std::getline(resumeFile, line);
        outputFile << line << "\n";
        // Strips comment from line
        size_t pos = line.find_first_of('#');
        if (pos != std::string::npos) {
            line = line.substr(0, pos);
        }
        // Reads the labels of the two files from items 
        std::stringstream ssline(line);
        if (ssline >> label1 >> numIn1 >> numOccl1 >> numRand1 >> label2) {
            // Finds the indices of label1 and label2
            auto iter1 = indicesMap.find(label1);
            if (iter1 == indicesMap.end()) i1 = -1;
            else i1 = iter1->second;
            auto iter2 = indicesMap.find(label2);
            if (iter2 == indicesMap.end()) i2 = -1;
            else i2 = iter2->second;
            std::cout << "  visited \"" << label1 << "\" [" << i1 << "] \"" << label2 << "\" [" << i2 << "]\n";
            // If both labels are found, it inserts the pair
            if (i1 >= 0 && i2 >= 0) {
                if (i1 != i2) {
                    visitedPairs.push_back(std::make_pair(i1, i2));
                } else {
                    // two files with the same short name are handled...
                    std::cout << "  homonymous \"" << label1 << "\": ";
                    auto range = indicesMap.equal_range(label1);
                    for (iter1 = range.first; iter1 != range.second; ++iter1) {
                        iter2 = iter1;
                        std::advance(iter2, 1);
                        for (; iter2 != range.second; ++iter2) {
                            i1 = iter1->second;
                            i2 = iter2->second;
                            if (i1 > i2) std::swap(i1, i2);
                            visitedPairs.push_back(std::make_pair(i1, i2));
                            std::cout << " (" << i1 << "," << i2 << ") ";
                        }
                    }
                    std::cout << std::endl;
                }
            }
        }
    }
    resumeFile.close();
    outputFile << "# RESUMING" << std::endl;

    // Finds the set difference
    std::sort(inputPairs.begin(), inputPairs.end());
    std::sort(visitedPairs.begin(), visitedPairs.end());
    std::set_difference(inputPairs.begin(), inputPairs.end(),
            visitedPairs.begin(), visitedPairs.end(),
            std::back_inserter(outputPairs));

    std::cout << "Remaining pairs:\n";
    for (auto& p : outputPairs) {
        std::cout << " " << p.first << ", " << p.second << ": \"" << getShortName(inputFilenames[p.first]) << "\", \"" << getShortName(inputFilenames[p.second]) << "\"\n";
    }
    std::cout << "All pairs " << inputPairs.size() << ", visited pairs " << visitedPairs.size() << ", remaining pairs " << outputPairs.size() << std::endl;
}

std::string getPrefix(std::string filename) {
    // Strips filename of the path 
    std::experimental::filesystem::path filepath(filename);
    std::string name = filepath.filename().string();
    std::string prefix;
    //  std::cout << "  name: \"" << name << "\"\n";

    // Finds the prefix
    size_t pos = name.find_first_of('_');
    if (pos != std::string::npos) {
        prefix = name.substr(0, pos);
    } else {
        prefix = name;
    }
    return prefix;
}

std::string getShortName(std::string filename) {
    std::stringstream ss;
    std::string prefix = getPrefix(filename);
    std::experimental::filesystem::path filenamePath = filename;
    filename = filenamePath.filename().string();
    // Computes a digest on the string
    unsigned int h = 19;
    for (int i = 0; i < filename.length(); ++i) {
        h = ((h * 31) + (unsigned int) filename[i]) % 97;
    }
    //  std::cout << "\nglob \"" << filenamePath.string() << "\" filename \"" << filename << "\" hash " << h << std::endl;
    ss << prefix << "_" << std::setw(2) << std::setfill('0') << h;
    return ss.str();
}

std::string getLeafDirectory(std::string filename) {
    std::experimental::filesystem::path filenamePath = filename;
    std::string parent = filenamePath.parent_path().string();
    size_t pos = parent.find_last_of('/');
    std::string leafDir = "";
    if (pos != std::string::npos) {
        leafDir = parent.substr(pos + 1, parent.length());
    }
    return leafDir;
}

void initParallelizationParams(ParlArsIsoParams& pp, int fourierOrder, int numPtsSrc, int numPtsDst, int blockSz, int chunkMaxSz) {

    int numPts = std::max<int>(numPtsSrc, numPtsDst);
    pp.numPts = numPts;
    int numPtsAfterPadding = numPts;
    pp.numPtsAfterPadding = numPtsAfterPadding;

    pp.chunkMaxSz = chunkMaxSz;
    int nc = numChunks(numPts, chunkMaxSz);
    pp.numChunks = nc;

    const int blockSize = blockSz;
    pp.blockSz = blockSize;

    const int coeffsMatNumCols = 2 * fourierOrder + 2;
    pp.coeffsMatNumCols = coeffsMatNumCols;
    const int coeffsMatNumColsPadded = coeffsMatNumCols;
    pp.coeffsMatNumColsPadded = coeffsMatNumColsPadded;
}

void updateParallelizationParams(ParlArsIsoParams& pp, int currChunkSz) {
    //Setting up parallelization
    //Parallelization parameters
    pp.currChunkSz = currChunkSz;
    //Fourier coefficients mega-matrix computation
    const int gridTotalSize = cuars::sumNaturalsUpToN(pp.currChunkSz - 1); //total number of threads in grid Fourier coefficients grid - BEFORE PADDING
    pp.gridTotalSize = gridTotalSize;

    const int numBlocks = floor(gridTotalSize / pp.blockSz) + 1; //number of blocks in grid (each block contains blockSize threads)
    pp.numBlocks = numBlocks;
    const int gridTotalSizeAfterPadding = pp.blockSz * numBlocks;
    pp.gridTotalSizeAfterPadding = gridTotalSizeAfterPadding;


    const int coeffsMatTotalSz = gridTotalSizeAfterPadding * pp.coeffsMatNumColsPadded; //sumNaturalsUpToN(numPts - 1) * coeffsMatNumColsPadded
    pp.coeffsMatTotalSz = coeffsMatTotalSz;

    //Fourier matrix sum -> parallelization parameters
    const int sumGridSz = pp.coeffsMatNumColsPadded; // = 2 * fourierOrder + 2
    pp.sumGridSz = sumGridSz;
    const int sumBlockSz = pp.blockSz;
    pp.sumBlockSz = sumBlockSz;

    //    std::cout << "Parallelization params:" << std::endl;
    //    std::cout << "numPts " << pp.numPts << " blockSize " << pp.blockSz << " numBlocks " << numBlocks
    //            << " gridTotalSize " << gridTotalSize << " gridTotalSizeAP " << gridTotalSizeAfterPadding << std::endl;
    //    std::cout << "sumSrcBlockSz " << sumBlockSz << " sumGridSz " << sumGridSz << std::endl;
    //        std::cout << "sum parallelization params: " << std::endl
    //            << "coeffMatNumCols " << pp.coeffsMatNumCols << " coeffsMatTotalSz " << coeffsMatTotalSz << std::endl;


    std::cout << "\nCalling kernel functions on GPU...\n" << std::endl;
}

void gpu_estimateRotationArsIso(const ArsImgTests::PointReaderWriter& pointsSrc, const ArsImgTests::PointReaderWriter& pointsDst, TestParams& tp, ParlArsIsoParams& paip, double& rotOut) {
    //ARS SRC -> preparation for kernel calls and kernel calls
    cudaEvent_t startSrc, stopSrc; //timing using CUDA events
    cudaEventCreate(&startSrc);
    cudaEventCreate(&stopSrc);

    initParallelizationParams(paip, tp.arsIsoOrder, pointsSrc.points().size(), pointsDst.points().size(), paip.blockSz, paip.chunkMaxSz);

    std::cout << "\n---Estimating Ars Iso on SRC---\n" << std::endl;

    double* coeffsArsSrc = new double [paip.coeffsMatNumColsPadded];
    double* d_coeffsArsSrc;
    cudaMalloc((void**) &d_coeffsArsSrc, paip.coeffsMatNumColsPadded * sizeof (double));
    cudaMemset(d_coeffsArsSrc, 0.0, paip.coeffsMatNumColsPadded * sizeof (double));

    for (int i = 0; i < paip.numChunks; ++i) {
        std::cout << "NUMPTS " << paip.numPts << std::endl;
        thrust::pair<int, int> indicesStartEnd = chunkStartEndIndices(i, paip.numPts, paip.chunkMaxSz);
        int currChunkSz = (indicesStartEnd.second - indicesStartEnd.first) + 1;

        updateParallelizationParams(paip, currChunkSz); //TODO: put chunkMaxSz as TestParams struct member?


        cuars::Vec2d * kernelInputSrc;
        cudaMalloc((void**) &kernelInputSrc, currChunkSz * sizeof (cuars::Vec2d));
        //        cudaMemcpy(kernelInputSrc, pointsSrc.points().data(), numPtsAfterPadding * sizeof (cuars::Vec2d), cudaMemcpyHostToDevice);
        std::cout << "round " << i + 1 << "/" << paip.numChunks << " -> "
                << "chunk-beg " << indicesStartEnd.first << " chunk-end " << indicesStartEnd.second << " --- chunk-size " << currChunkSz << std::endl;
        cuars::VecVec2d dataChunk(pointsSrc.points().begin() + indicesStartEnd.first, pointsSrc.points().begin() + (indicesStartEnd.first + currChunkSz));
        cudaMemcpy(kernelInputSrc, dataChunk.data(), (dataChunk.size()) * sizeof (cuars::Vec2d), cudaMemcpyHostToDevice);


        //Fourier matrix sum -> parallelization parameters
        std::cout << "Parallelization params:" << std::endl;
        std::cout << "numPts " << paip.numPts << " blockSize " << paip.blockSz << " numBlocks " << paip.numBlocks
                << " gridTotalSize " << paip.gridTotalSize << " gridTotalSizeAP " << paip.gridTotalSizeAfterPadding << std::endl;
        std::cout << "sumSrcBlockSz " << paip.sumBlockSz << " sumGridSz " << paip.sumGridSz << std::endl;

        std::cout << "sum parallelization params: " << std::endl
                << "coeffMatNumCols " << paip.coeffsMatNumCols << " coeffsMatTotalSz " << paip.coeffsMatTotalSz << std::endl;

        double *d_coeffsMatSrc;
        cudaMalloc((void**) &d_coeffsMatSrc, paip.coeffsMatTotalSz * sizeof (double));
        cudaMemset(d_coeffsMatSrc, 0.0, paip.coeffsMatTotalSz * sizeof (double));
        //    for (int i = 0; i < coeffsMatTotalSz; ++i) {
        //        coeffsMaSrc1[i] = 0.0;
        //    }

        double* d_partsumsSrc;
        cudaMalloc((void**) &d_partsumsSrc, paip.blockSz * paip.coeffsMatNumColsPadded * sizeof (double));
        cudaMemset(d_partsumsSrc, 0.0, paip.blockSz * paip.coeffsMatNumColsPadded * sizeof (double));



        cudaEventRecord(startSrc);
        iigDw << <paip.numBlocks, paip.blockSz >> >(kernelInputSrc, tp.arsIsoSigma, tp.arsIsoSigma, currChunkSz, tp.arsIsoOrder, paip.coeffsMatNumColsPadded, tp.arsIsoPnebiMode, d_coeffsMatSrc);
        //    sumColumnsNoPadding << <1, sumBlockSz>> >(coeffsMatSrc, gridTotalSizeAfterPadding, coeffsMatNumColsPadded, d_coeffsArsSrc);
        makePartialSums << < paip.coeffsMatNumColsPadded, paip.blockSz >>> (d_coeffsMatSrc, paip.gridTotalSizeAfterPadding, paip.coeffsMatNumColsPadded, d_partsumsSrc);
        sumColumnsPartialSums << <paip.coeffsMatNumColsPadded, 1 >> >(d_partsumsSrc, paip.blockSz, paip.coeffsMatNumColsPadded, d_coeffsArsSrc);
        cudaEventRecord(stopSrc);



        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

        //    for (int i = 0; i < coeffsMatNumColsPadded; ++i) {
        //        std::cout << "coeffsArsSrc[" << i << "] " << coeffsArsSrc[i] << std::endl;
        //    }

        cudaFree(d_partsumsSrc);
        cudaFree(d_coeffsMatSrc);
        cudaFree(kernelInputSrc);

    }

    cudaMemcpy(coeffsArsSrc, d_coeffsArsSrc, paip.coeffsMatNumColsPadded * sizeof (double), cudaMemcpyDeviceToHost);
    cudaFree(d_coeffsArsSrc);

    cudaEventSynchronize(stopSrc);
    float millisecondsSrc = 0.0f;
    cudaEventElapsedTime(&millisecondsSrc, startSrc, stopSrc);
    std::cout << "\nSRC -> insertIsotropicGaussians() " << millisecondsSrc << " ms" << std::endl;
    paip.gpu_srcExecTime = millisecondsSrc;

    cudaEventDestroy(startSrc);
    cudaEventDestroy(stopSrc);
    //END OF ARS SRC


    //    std::cout << "\n------\n" << std::endl; //"pause" between ars src and ars dst


    //ARS DST -> preparation for kernel calls and kernel calls
    cudaEvent_t startDst, stopDst; //timing using CUDA events
    cudaEventCreate(&startDst);
    cudaEventCreate(&stopDst);

    //    initParallelizationParams(paip, tp.arsIsoOrder, pointsSrc.points().size(), pointsDst.points().size(), paip.chunkMaxSz); //not needed for dst if it's done for src

    std::cout << "\n---Estimating Ars Iso on DST---\n" << std::endl;

    double* coeffsArsDst = new double [paip.coeffsMatNumColsPadded];
    double* d_coeffsArsDst;
    cudaMalloc((void**) &d_coeffsArsDst, paip.coeffsMatNumColsPadded * sizeof (double));
    cudaMemset(d_coeffsArsDst, 0.0, paip.coeffsMatNumColsPadded * sizeof (double));

    for (int i = 0; i < paip.numChunks; ++i) {
        std::cout << "NUMPTS " << paip.numPts << std::endl;
        thrust::pair<int, int> indicesStartEnd = chunkStartEndIndices(i, paip.numPts, paip.chunkMaxSz);
        int currChunkSz = (indicesStartEnd.second - indicesStartEnd.first) + 1;

        updateParallelizationParams(paip, currChunkSz); //TODO: put chunkMaxSz as TestParams struct member?


        cuars::Vec2d * kernelInputDst;
        cudaMalloc((void**) &kernelInputDst, currChunkSz * sizeof (cuars::Vec2d));
        //        cudaMemcpy(kernelInputDst, pointsDst.points().data(), numPtsAfterPadding * sizeof (cuars::Vec2d), cudaMemcpyHostToDevice);
        std::cout << "round " << i + 1 << "/" << paip.numChunks << " -> "
                << "chunk-beg " << indicesStartEnd.first << " chunk-end " << indicesStartEnd.second << " --- chunk-size " << currChunkSz << std::endl;
        cuars::VecVec2d dataChunk(pointsDst.points().begin() + indicesStartEnd.first, pointsDst.points().begin() + (indicesStartEnd.first + currChunkSz));
        cudaMemcpy(kernelInputDst, dataChunk.data(), (dataChunk.size()) * sizeof (cuars::Vec2d), cudaMemcpyHostToDevice);


        //Fourier matrix sum -> parallelization parameters
        std::cout << "Parallelization params:" << std::endl;
        std::cout << "numPts " << paip.numPts << " blockSize " << paip.blockSz << " numBlocks " << paip.numBlocks
                << " gridTotalSize " << paip.gridTotalSize << " gridTotalSizeAP " << paip.gridTotalSizeAfterPadding << std::endl;
        std::cout << "sumDstBlockSz " << paip.sumBlockSz << " sumGridSz " << paip.sumGridSz << std::endl;

        std::cout << "sum parallelization params: " << std::endl
                << "coeffMatNumCols " << paip.coeffsMatNumCols << " coeffsMatTotalSz " << paip.coeffsMatTotalSz << std::endl;

        double *d_coeffsMatDst;
        cudaMalloc((void**) &d_coeffsMatDst, paip.coeffsMatTotalSz * sizeof (double));
        cudaMemset(d_coeffsMatDst, 0.0, paip.coeffsMatTotalSz * sizeof (double));
        //    for (int i = 0; i < coeffsMatTotalSz; ++i) {
        //        coeffsMatDst1[i] = 0.0;
        //    }

        double* d_partsumsDst;
        cudaMalloc((void**) &d_partsumsDst, paip.blockSz * paip.coeffsMatNumColsPadded * sizeof (double));
        cudaMemset(d_partsumsDst, 0.0, paip.blockSz * paip.coeffsMatNumColsPadded * sizeof (double));



        cudaEventRecord(startDst);
        iigDw << <paip.numBlocks, paip.blockSz >> >(kernelInputDst, tp.arsIsoSigma, tp.arsIsoSigma, currChunkSz, tp.arsIsoOrder, paip.coeffsMatNumColsPadded, tp.arsIsoPnebiMode, d_coeffsMatDst);
        //    sumColumnsNoPadding << <1, sumBlockSz>> >(coeffsMatDst, gridTotalSizeAfterPadding, coeffsMatNumColsPadded, d_coeffsArsDst);
        makePartialSums << < paip.coeffsMatNumColsPadded, paip.blockSz >>> (d_coeffsMatDst, paip.gridTotalSizeAfterPadding, paip.coeffsMatNumColsPadded, d_partsumsDst);
        sumColumnsPartialSums << <paip.coeffsMatNumColsPadded, 1 >> >(d_partsumsDst, paip.blockSz, paip.coeffsMatNumColsPadded, d_coeffsArsDst);
        cudaEventRecord(stopDst);



        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

        //    for (int i = 0; i < coeffsMatNumColsPadded; ++i) {
        //        std::cout << "coeffsArsDst[" << i << "] " << coeffsArsDst[i] << std::endl;
        //    }

        cudaFree(d_partsumsDst);
        cudaFree(d_coeffsMatDst);
        cudaFree(kernelInputDst);

    }

    cudaMemcpy(coeffsArsDst, d_coeffsArsDst, paip.coeffsMatNumColsPadded * sizeof (double), cudaMemcpyDeviceToHost);
    cudaFree(d_coeffsArsDst);

    cudaEventSynchronize(stopDst);
    float millisecondsDst = 0.0f;
    cudaEventElapsedTime(&millisecondsDst, startDst, stopDst);
    std::cout << "\nDST -> insertIsotropicGaussians() " << millisecondsDst << " ms\n\n" << std::endl;
    paip.gpu_dstExecTime = millisecondsDst;


    cudaEventDestroy(startDst);
    cudaEventDestroy(stopDst);
    //END OF ARS DST




    //Computation final computations (correlation, ...) on CPU
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
        cuars::findGlobalMaxBBFourier(coeffsCor, 0.0, M_PI, tp.arsIsoThetaToll, fourierTol, thetaMax, corrMax);
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
