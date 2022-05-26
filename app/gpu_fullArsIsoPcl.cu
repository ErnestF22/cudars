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
#include <cstdlib>
#include <fstream>
#include <string>
#include <ctime>
#include <regex>
#include <algorithm>
#include <unordered_map>

#include <experimental/filesystem>

#include <rofl/common/param_map.h>

#include "cudars/ars2d.cuh"
#include "cudars/utils.h"

#include "cudars/ars2d.h"
#include "cudars/Profiler.h"

#include "cudars/mpeg7_io.h"
#include "cudars/mpeg7RW.h"

namespace expfs = std::experimental::filesystem;


// Functions
void parseResAsSigmaMin(std::string infoFilename, double& sigmaMin);

void readRotsFile(std::string filename, std::vector<double>& v);

void findComparisonPair(const std::vector<std::string>& inputFilenames, std::vector<std::pair<int, int> >& comPairs);

void filterComparisonPair(std::string resumeFilename, std::ostream& outputfile,
        const std::vector<std::string>& inputFilenames, std::vector<std::pair<int, int> >& inputPairs,
        std::vector<std::pair<int, int> >& outputPairs);

std::string getPrefix(std::string filename);

std::string getLeafDirectory(std::string filename);

int main(int argc, char** argv) {

    rofl::ParamMap params;

    std::string configFilename;
    std::string inputGlob;
    std::vector<std::string> inputFilenames;
    //    std::vector<std::pair<int, int> > allPairs;
    std::vector<std::pair<int, int> > outPairs;
    std::string outputFilename;

    TestParams tp;

    double rotTrue, rotArsIso, gpu_rotArsIso;
    int srcNumPts, dstNumPts;

    std::string rotsFilename;
    std::vector<double> rots; //rotations that are read from file

    //    std::string prefixName;

    params.read(argc, argv);

    // If a configuration file is specified, then the parameters are loaded from it.
    // Other command line options may overwrite the parameter values of config file. 
    params.getParam<std::string>("cfg", configFilename, "");
    std::cout << "config filename: " << configFilename << std::endl;
    if (configFilename != "") {
        params.read(configFilename);
    }

    // Re-parse from command line to overwrite config params: precedence is given to command-line!
    params.read(argc, argv);

    // Parameters value
    params.getParam<std::string>("in", inputGlob, expfs::current_path().string() + "/*");
    params.getParam<std::string>("out", outputFilename, mpeg7io::generateStampedString("results_", ".txt"));
    params.getParam<std::string>("rots", rotsFilename, "");


    params.getParam<int>("fileSkipper", tp.fileSkipper, 1);
    params.getParam<bool>("extrainfoEnable", tp.extrainfoEnable, true);


    // ArsIso params
    params.getParam<bool>("arsisoEnable", tp.arsIsoEnable, false);
    params.getParam<bool>("gpu_arsisoEnable", tp.gpu_arsIsoEnable, true);

    params.getParam<int>("arsOrder", tp.aiPms.arsIsoOrder, 20);
    params.getParam<double>("arsSigma", tp.aiPms.arsIsoSigma, 0.05);
    params.getParam<double>("arsTollDeg", tp.aiPms.arsIsoThetaToll, 0.5);
    tp.aiPms.arsIsoThetaToll *= M_PI / 180.0;


    ParlArsIsoParams paip;

    params.getParam<int>("blockSz", paip.blockSz, 256);
    params.getParam<int>("chunkMaxSz", paip.chunkMaxSz, 4096);


    std::cout << "\nParameter values:\n";
    params.write(std::cout);
    std::cout << std::endl;
    //End of Params section



    // Computation
    //ArsImgTests::glob(inputGlob, inputFilenames);
    std::cout << std::endl;
    mpeg7io::getDirectoryFiles(inputGlob, inputFilenames);
    std::cout << "\nFilenames:\n";
    size_t numFiles = 0;
    for (auto& filename : inputFilenames) {
        if (numFiles < 30)
            std::cout << "  " << filename << "\n";
        else if (numFiles == 30)
            std::cout << "..." << std::endl;

        numFiles++;
    }
    std::cout << std::endl;


    if (rotsFilename.empty())
        rotsFilename = expfs::path(inputGlob).parent_path().parent_path().string() + "/rand_rotations.txt";

    readRotsFile(rotsFilename, rots);




    if (!inputFilenames.empty()) {
        std::string leafDir = getLeafDirectory(inputFilenames[0]);
        std::cout << std::endl << "leafDir: \"" << leafDir << "\"" << std::endl;
        std::string methodSuffix;
        if (tp.arsIsoEnable) {
            methodSuffix = methodSuffix + "_arsiso";
        }
        if (tp.gpu_arsIsoEnable) {
            methodSuffix = methodSuffix + "_gpuarsiso";
        }
        if (tp.extrainfoEnable) {
            methodSuffix = methodSuffix + "_extrainfo";
        }
        std::string datafolder = expfs::path(inputGlob).parent_path().parent_path().filename().string();
        outputFilename = mpeg7io::generateStampedString("results_" + datafolder.replace(datafolder.begin(), datafolder.begin() + 5, "") + methodSuffix + "_", ".txt"); //replace() is used in order to remove "data_" from the string
        std::cout << "outputFilename: \"" << outputFilename << "\"" << std::endl;

    }

    // Open output results file
    std::ofstream outfile(outputFilename.c_str());
    if (!outfile) {
        std::cerr << "Cannot open file \"" << outputFilename << "\"" << std::endl;
        return -1;
    }


    for (int i = 0; i < inputFilenames.size() - 1; ++i) {
        std::pair<int, int> pr(i, i + 1);
        outPairs.push_back(pr);
    }


    outfile << "# Parameters:\n";
    params.write(outfile, "#  ");
    outfile << "# \n";
    outfile << "# file1 numpts1 file2 numpts2 rotTrue rotTrue[deg] ";

    if (tp.arsIsoEnable) {
        outfile << "arsIso rotArsIso[deg] ";
    }
    if (tp.gpu_arsIsoEnable) {
        outfile << "gpu_arsIso rotGpuArsIso[deg] ";
    }
    if (tp.extrainfoEnable)
        outfile << "srcNumPts srcExecTime gpu_srcExecTime dstNumPts dstExecTime gpu_dstExecTime";


    outfile << "\n";

    std::cout << std::endl << std::endl;

    int countPairs = 0;
    for (auto& comp : outPairs) {
        if (countPairs % tp.fileSkipper) {
            countPairs++;
            continue;
        }

        //        std::cout << "[" << countPairs + 1 << "/" << outPairs.size() << "]\n" << "  * \"" << inputFilenames[comp.first] << "\"\n    \"" << inputFilenames[comp.second] << "\"" << std::endl;
        std::cout << std::endl << "[" << countPairs + 1 << "/" << outPairs.size() << "]" << std::endl;

        ArsImgTests::PointReaderWriter pointsSrc;
        ArsImgTests::PointReaderWriter pointsDst;

        std::cout << "--- Reading input cloud SRC ---" << std::endl;
        pointsSrc.loadPcdAscii(inputFilenames[comp.first]);
        std::cout << std::endl << "--- Reading input cloud DST ---" << std::endl;
        pointsDst.loadPcdAscii(inputFilenames[comp.second]);
        //apply random rotation to pointsDst
        cudars::Affine2d aff2d(M_PI * rots[countPairs] / 180.0f, 0.0, 0.0); //putting all these translations to 0 for now
        pointsDst.applyTransform(aff2d); //this should also automatically update the internal rotTheta and transl members

        rotTrue = pointsDst.getRotTheta() - pointsSrc.getRotTheta();
        //    if (rotTrue < 0.0) rotTrue += M_PI;
        //    else if (rotTrue > M_PI) rotTrue -= M_PI;

        //        r.setPointsSrc(pointsSrc);
        //        r.setPointsDst(pointsDst);


        std::cout << " angle dst " << (180.0 / M_PI * pointsDst.getRotTheta()) << " [deg], src " << (180.0 / M_PI * pointsSrc.getRotTheta()) << " [deg]" << std::endl;
        std::cout << std::fixed << std::setprecision(2) << std::setw(10)
                << "  rotTrue \t" << (180.0 / M_PI * rotTrue) << " deg\t\t" << (180.0 / M_PI * cudars::mod180(rotTrue)) << " deg [mod 180]\n";

        outfile
                << std::setw(12) << getPrefix(inputFilenames[comp.first]) << " "
                << std::setw(6) << pointsSrc.points().size() << " "
                << std::setw(12) << getPrefix(inputFilenames[comp.second]) << " "
                << std::setw(6) << pointsDst.points().size() << " "
                << "rotTrue" << std::fixed << std::setprecision(2) << std::setw(8) << (180.0 / M_PI * cudars::mod180(rotTrue)) << " ";


        if (tp.arsIsoEnable) {
            //            r.estimateRotationArsIso(rotArsIso);
            std::cout << std::fixed << std::setprecision(2) << std::setw(10)
                    << "  rotArsIso \t" << (180.0 / M_PI * rotArsIso) << " deg\t\t" << (180.0 / M_PI * cudars::mod180(rotArsIso)) << " deg [mod 180]\n";
            outfile << std::setw(6) << "arsIso " << std::fixed << std::setprecision(2) << std::setw(6) << (180.0 / M_PI * cudars::mod180(rotArsIso)) << " ";
        }
        if (tp.gpu_arsIsoEnable) {
            gpu_estimateRotationArsIso(pointsSrc.points(), pointsDst.points(), tp, paip, gpu_rotArsIso);
            std::cout << std::endl << std::fixed << std::setprecision(2) << std::setw(10)
                    << "  gpu_rotArsIso \t" << (180.0 / M_PI * gpu_rotArsIso) << " deg\t\t" << (180.0 / M_PI * cudars::mod180(gpu_rotArsIso)) << " deg [mod 180]\n";
            outfile << std::setw(6) << "gpu_arsIso " << std::fixed << std::setprecision(2) << std::setw(6) << (180.0 / M_PI * cudars::mod180(gpu_rotArsIso)) << " ";
        }

        if (tp.extrainfoEnable) {
            srcNumPts = pointsSrc.points().size();
            dstNumPts = pointsDst.points().size();
            std::cout << std::fixed << std::setprecision(2) << std::setw(10)
                    << "  srcNumPts \t" << srcNumPts << "  srcExecTime \t" << paip.srcExecTime << "  gpu_srcExecTime \t" << paip.gpu_srcExecTime
                    << "  dstNumPts \t" << dstNumPts << "  dstExecTime \t" << paip.dstExecTime << "  gpu_dstExecTime \t" << paip.gpu_dstExecTime
                    << std::endl;

            outfile << std::setw(8) << "sPts " << std::fixed << std::setprecision(2) << std::setw(6) << srcNumPts << " "
                    << std::setw(8) << "sTm " << std::fixed << std::setprecision(2) << std::setw(6) << paip.srcExecTime << " "
                    << std::setw(8) << "gpu_sTm " << std::fixed << std::setprecision(2) << std::setw(12) << paip.gpu_srcExecTime << " "
                    << std::setw(8) << "dPts " << std::fixed << std::setprecision(2) << std::setw(6) << dstNumPts << " "
                    << std::setw(8) << "dTm " << std::fixed << std::setprecision(2) << std::setw(6) << paip.dstExecTime << " "
                    << std::setw(8) << "gpu_dTm " << std::fixed << std::setprecision(2) << std::setw(12) << paip.gpu_dstExecTime << " ";
        }

        outfile << std::endl;


        countPairs++;
        std::cout << "\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n";
    }

    outfile.close();

    return 0;
}

void parseResAsSigmaMin(std::string infoFilename, double& sigmaMin) {
    std::cout << "Reading resolution from file " << infoFilename << " and setting sigmaMin equal to it" << std::endl;
    std::ifstream infoInStream(infoFilename);

    std::string line;
    while (std::getline(infoInStream, line)) {
        std::regex regex("resolution\\s+([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+))");
        if (std::regex_match(line, regex)) {
            std::smatch sm;
            std::regex_search(line, sm, regex);

            double res = boost::lexical_cast<double, std::string>(sm.str(1)); //NOTE: sm[0] is the entire string

            std::cout << "res " << res << std::endl;
            sigmaMin = res;
            break;
        }
    }
    infoInStream.close();
}

void readRotsFile(std::string filename, std::vector<double>& v) {
    std::cout << "Reading rotations from file " << filename << std::endl;
    std::ifstream ifstrm(filename);
    std::string line;
    int rotCtr = 0;
    while (std::getline(ifstrm, line)) {
        double rotVal = boost::lexical_cast<double, std::string>(line);
        v.push_back(rotVal);
        rotCtr++;
    }
    std::cout << "Read " << rotCtr << " rotation values" << std::endl;
    ifstrm.close();
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
        for (int i1 = idx1; i1 < idx2; ++i1) {
            for (int i2 = i1 + 1; i2 < idx2; ++i2) {
                comPairs.push_back(std::make_pair(i1, i2));
            }
        }
        idx1 = idx2;
    }
}

// Reads outputFilename for the list of already processed files

std::string getPrefix(std::string filename) {
    // Strips filename of the path 
    expfs::path filepath(filename);
    std::string name = filepath.filename().string();
    std::string prefix;
    //  std::cout << "  name: \"" << name << "\"\n";

    // Finds the prefix
    size_t pos = name.find_first_of('.');
    //    size_t pos = name.find_first_of('_');
    if (pos != std::string::npos) {
        prefix = name.substr(0, pos);
    } else {
        prefix = name;
    }
    return prefix;
}

std::string getLeafDirectory(std::string filename) {
    expfs::path filenamePath = filename;
    std::string parent = filenamePath.parent_path().string();
    size_t pos = parent.find_last_of('/');
    std::string leafDir = "";
    if (pos != std::string::npos) {
        leafDir = parent.substr(pos + 1, parent.length());
    }
    return leafDir;
}



