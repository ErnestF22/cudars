
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

#include "ars/ars2d.cuh"
#include "ars/utils.h"

#include <ars/ars2d.h>
#include <ars/Profiler.h>

#include <ars/mpeg7_io.h>
#include <ars/mpeg7RW.h>

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

void gpu_estimateRotationArsIso(const ArsImgTests::PointReaderWriter& pointsSrc, const ArsImgTests::PointReaderWriter& pointsDst, TestParams& tp, ParlArsIsoParams& paip, double& rotOut);

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
        cuars::Affine2d aff2d(M_PI * rots[countPairs] / 180.0f, 0.0, 0.0); //putting all these translations to 0 for now
        pointsDst.applyTransform(aff2d); //this should also automatically update the internal rotTheta and transl members

        rotTrue = pointsDst.getRotTheta() - pointsSrc.getRotTheta();
        //    if (rotTrue < 0.0) rotTrue += M_PI;
        //    else if (rotTrue > M_PI) rotTrue -= M_PI;

        //        r.setPointsSrc(pointsSrc);
        //        r.setPointsDst(pointsDst);


        std::cout << " angle dst " << (180.0 / M_PI * pointsDst.getRotTheta()) << " [deg], src " << (180.0 / M_PI * pointsSrc.getRotTheta()) << " [deg]" << std::endl;
        std::cout << std::fixed << std::setprecision(2) << std::setw(10)
                << "  rotTrue \t" << (180.0 / M_PI * rotTrue) << " deg\t\t" << (180.0 / M_PI * cuars::mod180(rotTrue)) << " deg [mod 180]\n";

        outfile
                << std::setw(12) << getPrefix(inputFilenames[comp.first]) << " "
                << std::setw(6) << pointsSrc.points().size() << " "
                << std::setw(12) << getPrefix(inputFilenames[comp.second]) << " "
                << std::setw(6) << pointsDst.points().size() << " "
                << "rotTrue" << std::fixed << std::setprecision(2) << std::setw(8) << (180.0 / M_PI * cuars::mod180(rotTrue)) << " ";


        if (tp.arsIsoEnable) {
            //            r.estimateRotationArsIso(rotArsIso);
            std::cout << std::fixed << std::setprecision(2) << std::setw(10)
                    << "  rotArsIso \t" << (180.0 / M_PI * rotArsIso) << " deg\t\t" << (180.0 / M_PI * cuars::mod180(rotArsIso)) << " deg [mod 180]\n";
            outfile << std::setw(6) << "arsIso " << std::fixed << std::setprecision(2) << std::setw(6) << (180.0 / M_PI * cuars::mod180(rotArsIso)) << " ";
        }
        if (tp.gpu_arsIsoEnable) {
            gpu_estimateRotationArsIso(pointsSrc.points(), pointsDst.points(), tp, paip, gpu_rotArsIso);
            std::cout << std::endl << std::fixed << std::setprecision(2) << std::setw(10)
                    << "  gpu_rotArsIso \t" << (180.0 / M_PI * gpu_rotArsIso) << " deg\t\t" << (180.0 / M_PI * cuars::mod180(gpu_rotArsIso)) << " deg [mod 180]\n";
            outfile << std::setw(6) << "gpu_arsIso " << std::fixed << std::setprecision(2) << std::setw(6) << (180.0 / M_PI * cuars::mod180(gpu_rotArsIso)) << " ";
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

void gpu_estimateRotationArsIso(const ArsImgTests::PointReaderWriter& pointsSrc, const ArsImgTests::PointReaderWriter& pointsDst, TestParams& tp, ParlArsIsoParams& paip, double& rotOut) {
    //ARS SRC -> preparation for kernel calls and kernel calls
    cudaEvent_t startSrc, stopSrc; //timing using CUDA events
    cudaEventCreate(&startSrc);
    cudaEventCreate(&stopSrc);

    const cuars::VecVec2d& inputSrc = pointsSrc.points();
    initParallelizationParams(paip, tp.aiPms.arsIsoOrder, inputSrc.size(), paip.blockSz, paip.chunkMaxSz); //cudarsIso.init()
    double* coeffsArsSrc = new double [paip.coeffsMatNumColsPadded];
    computeArsIsoGpu(paip, tp.aiPms, inputSrc, coeffsArsSrc, startSrc, stopSrc, paip.gpu_srcExecTime); //cudarsIso.compute()

    cudaEventDestroy(startSrc);
    cudaEventDestroy(stopSrc);
    //END OF ARS SRC

    //    std::cout << "\n------\n" << std::endl; //"pause" between ars src and ars dst

    //ARS DST -> preparation for kernel calls and kernel calls
    cudaEvent_t startDst, stopDst; //timing using CUDA events
    cudaEventCreate(&startDst);
    cudaEventCreate(&stopDst);

    const cuars::VecVec2d& inputDst = pointsDst.points();
    initParallelizationParams(paip, tp.aiPms.arsIsoOrder, inputDst.size(), paip.blockSz, paip.chunkMaxSz); //cudarsIso.init()
    double* coeffsArsDst = new double [paip.coeffsMatNumColsPadded];
    computeArsIsoGpu(paip, tp.aiPms, inputDst, coeffsArsDst, startDst, stopDst, paip.gpu_dstExecTime); //cudarsIso.compute()

    cudaEventDestroy(startDst);
    cudaEventDestroy(stopDst);
    //END OF ARS DST

    std::cout << std::endl << "---Computing corelation---" << std::endl;

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

    std::cout << std::endl << "ROT OUT " << rotOut << std::endl;

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

