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
#include <experimental/filesystem>

#include <cudars/utils.h>
#include <cudars/mpeg7RW.h>
#include <cudars/mpeg7_io.h>

#include <cudars/BBTranslation.h>
#include <cudars/ars2d.cuh>

#include <rofl/common/param_map.h>

namespace expfs = std::experimental::filesystem;

int main(int argc, char **argv)
{
    cudars::BBTranslation arsBBTransl;

    rofl::ParamMap params;

    /////////////////////////////////////////////////////////////////////////////////////////////

    cudars::Vec2d translMin, translMax, translGt, p;

    double translRes;

    std::string configFilename;
    std::string inputGlob;
    std::vector<std::string> inputFilenames;
    std::vector<std::pair<int, int>> allPairs;
    std::vector<std::pair<int, int>> outPairs;
    std::string outputFilename;
    std::string resumeFilename;

    double rotTrue, rotArsIso, rotNiArs, rotHS;
    cudars::Vec2d translTrue, translBbTransl;

    CudarsImgTests::PointReaderWriter pointsSrc;
    CudarsImgTests::PointReaderWriter pointsDst;

    std::string prefixName;

    cudars::AngularRadonSpectrum2d arsSrc;
    cudars::AngularRadonSpectrum2d arsDst;
    TestParams tparams;
    ParlArsIsoParams paiParams;

    int srcNumPts, dstNumPts;

    params.read(argc, argv);

    // If a configuration file is specified, then the parameters are loaded from it.
    // Other command line options may overwrite the parameter values of config file.
    params.getParam<std::string>("cfg", configFilename, "");
    std::cout << "config filename: " << configFilename << std::endl;
    if (configFilename != "")
    {
        params.read(configFilename);
    }

    // Re-parse from command line to overwrite config params: precedence is given to command-line!
    params.read(argc, argv);

    bool bbTranslEnable;
    double bbTranslRes, bbTranslEps;
    int bbTranslMaxIterations;
    params.getParam<bool>("bbTranslEnable", bbTranslEnable, true);
    params.getParam<double>("bbTranslRes", bbTranslRes, 0.1);
    params.getParam<double>("bbTranslEps", bbTranslEps, 0.1);
    params.getParam<int>("bbTranslMaxIterations", bbTranslMaxIterations, 1000);

    arsBBTransl.setResolution(bbTranslRes);
    arsBBTransl.setEps(bbTranslEps);
    arsBBTransl.setNumMaxIterations(bbTranslMaxIterations);

    // Enabled methods bool params
    bool rotEvalEnable, arsTecEnable;
    bool arsIcpEnable;
    bool procrUmeyEnable;

    // Parameters value
    params.getParam<std::string>("in", inputGlob, expfs::current_path().string() + "/*");
    params.getParam<std::string>("out", outputFilename, mpeg7io::generateStampedString("results_", ".txt"));
    params.getParam<std::string>("out", outputFilename, mpeg7io::generateStampedString("results_", ".txt"));
    //    params.getParam<std::string>("rots", rotsFilename, "");

    // Enabled methods bool params
    bool arsIsoEnable, niArsEnable, hsEnable;

    // Other params used only in executable and not during computations
    int fileSkipper;
    bool plotGrid, plotOutput, extrainfoEnable;
    // ArsIso params (CPU and GPU)
    params.getParam<bool>("arsisoEnable", tparams.arsIsoEnable, false);
    params.getParam<bool>("gpu_arsisoEnable", tparams.gpu_arsIsoEnable, true);
    params.getParam<int>("arsisoOrder", tparams.aiPms.arsIsoOrder, 20);
    params.getParam<double>("arsisoSigma", tparams.aiPms.arsIsoSigma, 1.0);
    params.getParam<double>("arsisoTollDeg", tparams.aiPms.arsIsoThetaToll, 0.5);
    tparams.aiPms.arsIsoThetaToll *= M_PI / 180.0;
    //    params.getParam<unsigned int>("arsisoPnebiMode", tparams.arsIsoPnebiMode, cudars::ArsKernelIsotropic2d::ComputeMode::PNEBI_DOWNWARD);


    arsSrc.setARSFOrder(tparams.aiPms.arsIsoOrder);
    //    arsSrc.initLUT(0.0001);
    //    arsSrc.setComputeMode(ars::ArsKernelIsotropic2d::ComputeMode::PNEBI_LUT);
    arsSrc.setComputeMode(cudars::ArsKernelIso2dComputeMode::PNEBI_DOWNWARD);
    arsDst.setARSFOrder(tparams.aiPms.arsIsoOrder);
    arsDst.setComputeMode(cudars::ArsKernelIso2dComputeMode::PNEBI_DOWNWARD);


    //parallelization parameters
    params.getParam<int>("blockSz", paiParams.blockSz, 256);
    params.getParam<int>("chunkMaxSz", paiParams.chunkMaxSz, 4096);

    params.getParam<int>("fileSkipper", tparams.fileSkipper, 1);

    std::cout << "Params:\n";
    params.write(std::cout);
    std::cout << std::endl;

    // Computation
    std::cout << std::endl;
    mpeg7io::getDirectoryFiles(inputGlob, inputFilenames);
    std::cout << "\nFilenames:\n";
    size_t numFiles = 0;
    for (auto &filename : inputFilenames)
    {
        if (numFiles < 30)
            std::cout << "  " << filename << "\n";
        else if (numFiles == 30)
            std::cout << "..." << std::endl;

        numFiles++;
    }
    std::cout << std::endl;

    if (!inputFilenames.empty())
    {
        std::string leafDir = mpeg7io::getLeafDirectory(inputFilenames[0]);
        std::cout << std::endl
                  << "leafDir: \"" << leafDir << "\"" << std::endl;
        std::string methodSuffix;
        if (arsTecEnable)
        {
            methodSuffix = methodSuffix + "_arstec";
        }
        if (extrainfoEnable)
        {
            methodSuffix = methodSuffix + "_extrainfo";
        }
        std::string datafolder = expfs::path(inputGlob).parent_path().parent_path().filename().string();
        outputFilename = mpeg7io::generateStampedString("results_arspose_doicp" + boost::lexical_cast<std::string>(arsIcpEnable) + "_" + leafDir + methodSuffix + "_", ".txt");
        std::cout << "outputFilename: \"" << outputFilename << "\"" << std::endl;
    }

    // Open output results file
    std::ofstream outfile(outputFilename.c_str());
    if (!outfile)
    {
        std::cerr << "Cannot open file \"" << outputFilename << "\"" << std::endl;
        return -1;
    }

    mpeg7io::findComparisonPair(inputFilenames, allPairs);
    std::cout << "Processing " << inputFilenames.size() << " files, " << allPairs.size() << " comparisons\n"
              << std::endl;
    mpeg7io::filterComparisonPair(resumeFilename, outfile, inputFilenames, allPairs, outPairs);
    std::cout << "Remaining comparisons " << outPairs.size() << " comparisons\n"
              << std::endl;

    outfile << "# Parameters:\n";
    params.write(outfile, "#  ");
    outfile << "# \n";
    outfile << "# file1 numpts1 file2 numpts2 translTrue translTrue[m] ";

    if (arsTecEnable)
    {
        outfile << "ars translArs[m] ";
    }
    if (extrainfoEnable)
    {
        //        if (arsIsoEnable)
        //            outfile << " srcArsIsoTm dstArsIsoTm ";
        //        if (niArsEnable)
        //            outfile << "srcNumKers srcNiArsExecTime dstNumKers dstNiArsExecTime "; //Kers stands for kernels
        //        if (hsEnable)
        //            outfile << " srcHsTm dstHsTm ";
        if (extrainfoEnable)
            outfile << " srcTranslTm dstTranslTm ";
    }

    outfile << "\n";

    std::cout << std::endl
              << std::endl;

    int countPairs = 0;
    for (auto &comp : outPairs)
    {
        if (countPairs % fileSkipper)
        {
            countPairs++;
            std::cout << "\n\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n";
            continue;
        }

        //        std::cout << "[" << countPairs + 1 << "/" << outPairs.size() << "]\n" << "  * \"" << inputFilenames[comp.first] << "\"\n    \"" << inputFilenames[comp.second] << "\"" << std::endl;
        std::cout << std::endl
                  << "[" << countPairs + 1 << "/" << outPairs.size() << "]" << std::endl;

        pointsSrc.load(inputFilenames[comp.first]);
        pointsDst.load(inputFilenames[comp.second]);
        prefixName = mpeg7io::getPrefix(inputFilenames[comp.first]);

        arsBBTransl.setPtsSrc(pointsSrc.points());
        arsBBTransl.setPtsDst(pointsDst.points());

        //        std::cout << std::endl << "Inserting pair source-destination..." << std::endl;
        //        arsBBTransl.insertPoints();

        double rotTrue = pointsDst.getRotTheta() - pointsSrc.getRotTheta();

        //        std::cout << "pointsSrc transform" << std::endl << pointsSrc.getTransform().matrix() << std::endl;
        //        std::cout << "pointsDst transform" << std::endl << pointsDst.getTransform().matrix() << std::endl;

        cudars::Affine2d transfSrcToDst = cudars::aff2ProdWRV(pointsDst.getTransform(), pointsSrc.getTransform().inverse());
        std::cout << "diff transform" << std::endl
                  << transfSrcToDst << std::endl;
        translTrue = transfSrcToDst.translation();
        translGt = translTrue;

        //        std::cout << "transl src [m]\n" << pointsSrc.getTransl() << "\ntransl dst [m]\n" << pointsDst.getTransl() << std::endl;
        std::cout << std::fixed << std::setprecision(2) << std::setw(10)
                  << "\nrotTrue [deg]\t" << rotTrue * 180.0 / M_PI << "\n";
        std::cout << std::fixed << std::setprecision(2) << std::setw(10) << "\ntranslTrue [m]\t" << translTrue.x << " " << translTrue.y << std::endl;

        cudars::Vec2d translMin, translMax;
        translMin.x = pointsDst.xmin() - pointsSrc.xmax();
        translMin.y = pointsDst.ymin() - pointsSrc.ymax();
        translMax.x = pointsDst.xmax() - pointsSrc.xmin();
        translMax.y = pointsDst.ymax() - pointsSrc.ymin();
        //        std::cout << "tmax [m]\n" << translMax << std::endl;

        arsBBTransl.setTranslMinMax(translMin, translMax);

        std::cout << " angle dst " << (180.0 / M_PI * pointsDst.getRotTheta()) << " [deg], src " << (180.0 / M_PI * pointsSrc.getRotTheta()) << " [deg]" << std::endl;
        std::cout << std::fixed << std::setprecision(2) << std::setw(10)
                  << "  rotTrue \t" << (180.0 / M_PI * rotTrue) << " deg\t\t" << (180.0 / M_PI * cudars::mod180(rotTrue)) << " deg [mod 180]\n";

        outfile
            << std::setw(12) << mpeg7io::getPrefix(inputFilenames[comp.first]) << " "
            << std::setw(6) << pointsSrc.points().size() << " "
            << std::setw(12) << mpeg7io::getPrefix(inputFilenames[comp.second]) << " "
            << std::setw(6) << pointsDst.points().size() << " "
            << "translTrue" << std::fixed << std::setprecision(2) << std::setw(8) << translTrue.x << " " << translTrue.y << "\t";

        double rotArsIso_gpu;
        if (arsIsoEnable)
        {
            gpu_estimateRotationArsIso(pointsSrc.points(), pointsDst.points(), tparams, paiParams, rotArsIso_gpu);
            std::cout << std::fixed << std::setprecision(2) << std::setw(10)
                      << "  rotArsIso \t" << (180.0 / M_PI * rotArsIso) << " deg\t\t" << (180.0 / M_PI * cudars::mod180(rotArsIso)) << " deg [mod 180]\n";
            outfile << std::setw(6) << "arsIso " << std::fixed << std::setprecision(2) << std::setw(6) << (180.0 / M_PI * cudars::mod180(rotArsIso)) << " ";
        }
        // if (niArsEnable)
        // {
        //     r.estimateRotationNiArs(rotNiArs);
        //     std::cout << std::fixed << std::setprecision(2) << std::setw(10)
        //               << "  rotNiArs \t" << (180.0 / M_PI * rotNiArs) << " deg\t\t" << (180.0 / M_PI * cudars::mod180(rotNiArs)) << " deg [mod 180]\n";
        //     outfile << std::setw(6) << "niArs " << std::fixed << std::setprecision(2) << std::setw(6) << (180.0 / M_PI * cudars::mod180(rotNiArs)) << " ";
        // }
        // if (hsEnable)
        // {
        //     r.estimateRotationHS(rotHS);
        //     std::cout << std::fixed << std::setprecision(2) << std::setw(10)
        //               << "  rotHS  \t" << (180.0 / M_PI * rotHS) << " deg\t\t" << (180.0 / M_PI * cudars::mod180(rotHS)) << " deg [mod 180]\n";
        //     outfile << std::setw(6) << "hs " << std::fixed << std::setprecision(2) << std::setw(6) << (180.0 / M_PI * cudars::mod180(rotHS)) << " ";
        // }
        if (bbTranslEnable)
        {
            cudars::Vec2d minA, maxA, minB, maxB;
            cudars::Affine2d transf;

            pointsSrc.applyTransform(0.0, 0.0, rotTrue);

            cudars::findBoundingBox(pointsSrc.points(), minA, maxA);
            cudars::findBoundingBox(pointsDst.points(), minB, maxB);

            // Add bias
            // cudars::Vec2d bias;
            // bias << 20.0, 20.0;
            // std::cout << "ptsA: size " << ptsA.size() << ", min [" << minA.transpose()
            //           << "]  max [" << maxA.transpose() << "]" << std::endl;
            // std::cout << "ptsB: size " << ptsB.size() << ", min [" << minB.transpose()
            //           << "]  max [" << maxB.transpose() << "]" << std::endl;
            // std::cout << "translation interval: min [" << (minB - maxA).transpose()
            //           << "]  max [" << (maxB - minA + bias).transpose() << "]" << std::endl;
            // arsBBTransl.setTranslMinMax(minB - maxA, maxB - minA + bias);

            arsBBTransl.setTranslMinMax(cudars::vec2diffWRV(minB, maxA), cudars::vec2diffWRV(maxB, minA));

            arsBBTransl.setPts(pointsSrc.points(), pointsDst.points()); //?? need to clear before setting the new ones?
            arsBBTransl.compute(translBbTransl);

            std::cout << std::fixed << std::setprecision(2) << std::setw(10)
                      << "  rotTrue  \t" << (180.0 / M_PI * rotTrue) << " deg\t\t" << std::endl
                      << "  translBbTransl  \t" << translBbTransl.x << " " << translBbTransl.y << " \t[m]" << std::endl;
            outfile << std::setw(6) << "  translBbTransl  \t" << translBbTransl.x << " " << translBbTransl.y;

            pointsSrc.applyTransform(0.0, 0.0, -rotTrue);
        }

        if (tparams.extrainfoEnable)
        {
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

            if (arsIsoEnable)
            {
                // std::cout << std::endl
                //           << std::fixed << std::setprecision(2) << std::setw(10) << std::endl
                //           << "  isoSrcExecTime \t" << isoSrcExecTime << std::endl
                //           << "  isoDstExecTime \t" << isoDstExecTime << std::endl;

                // outfile << std::setw(6) << "sIsoTm " << std::fixed << std::setprecision(2) << std::setw(12) << isoSrcExecTime << " "
                //         << std::setw(6) << "dIsoTm " << std::fixed << std::setprecision(2) << std::setw(12) << isoDstExecTime << " ";
            }
            if (bbTranslEnable)
            {
                // std::cout << std::endl
                //           << std::fixed << std::setprecision(2) << std::setw(10) << std::endl
                //           << "  hsSrcExecTime \t" << ei.hsSrcExecTime << std::endl
                //           << "  hsDstExecTime \t" << ei.hsDstExecTime << std::endl;

                // outfile << std::setw(6) << "sHsTm " << std::fixed << std::setprecision(2) << std::setw(12) << ei.hsSrcExecTime << " "
                // << std::setw(6) << "dHsTm " << std::fixed << std::setprecision(2) << std::setw(12) << ei.hsDstExecTime << " ";
            }
        }

        outfile << std::endl;

        countPairs++;
        std::cout << "\n\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n";
    }

    return 0;
}
