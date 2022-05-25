#include <iostream>
#include "ars/utils.h"
#include "ars/ars2d.cuh"
#include <ars/ars2d.h>
#include <ars/ConsensusTranslationEstimator.cuh>
#include <rofl/common/param_map.h>

int main(int argc, char **argv)
{
    // cuars::ConsensusTranslationEstimator2d translEstim;
    cuars::VecVec2d pointsSrc, pointsDst, translCandidates;
    rofl::ParamMap params;
    std::string filenameCfg;

    cuars::ArsTec2dParams translParams;
    // ParlArsIsoParams paiParams;

    // Reads params from command line
    params.read(argc, argv);
    params.getParam("cfg", filenameCfg, std::string(""));
    params.read(filenameCfg);
    params.read(argc, argv);
    params.getParam<double>("translRes", translParams.translRes, 1.0);
    // params.getParamContainer("translMin", translMin.data(), translMin.data() + translMin.size(), "[-10.0,-10.0]", double(0.0), "[,]"); //TODO: adapt ParamContainer to Cuda types
    params.getParam<double>("translMin-x", translParams.translMin.x, -10.0);
    params.getParam<double>("translMin-y", translParams.translMin.y, -10.0);
    // params.getParamContainer("translGt", translGt.data(), translGt.data() + translGt.size(), "[-4.2,5.0]", double(1.0), "[,]");
    params.getParam<double>("translGt-x", translParams.translGt.x, -4.2);
    params.getParam<double>("translGt-y", translParams.translGt.y, 5.0);
    params.getParamContainer("gridSize", translParams.gridSize.data(), translParams.gridSize.data() + translParams.gridSize.size(), "[21,21]", int(0), "[,]");
    params.getParamContainer("gridWin", translParams.gridWin.data(), translParams.gridWin.data() + translParams.gridWin.size(), "[1,1]", int(1), "[,]");
    params.getParam<bool>("adaptive", translParams.adaptiveGrid, false);
    params.getParam<bool>("plot", translParams.plot, false);

    std::cout << "\nParams:" << std::endl;
    params.write(std::cout);
    std::cout << "-------\n"
              << std::endl;

    for (int i = 0; i < 10; ++i)
    {
        cuars::Vec2d p;
        cuars::fillVec2d(p, (1.0 + 0.4 * i), (-2.0 - 0.35 * i));
        pointsSrc.push_back(p);
        pointsDst.push_back(cuars::vec2sumWRV(p, translParams.translGt));
    }
    pointsDst.push_back(make_double2(0.0, 0.0));
    pointsDst.push_back(make_double2(4.0, 4.0));
    pointsSrc.push_back(make_double2(3.2, 6.2));
    pointsSrc.push_back(make_double2(3.5, 2.6));

    std::cout << "Source point set:\n";
    for (auto &pt : pointsSrc)
    {
        std::cout << "  [";
        // cuars::printVec2d(pt);
        std::cout << pt.x << "\t" << pt.y;
        std::cout << "]\n";
    }
    std::cout << "Destination point set:\n";
    for (auto &pt : pointsDst)
    {
        std::cout << "  [";
        // cuars::printVec2d(pt);
        std::cout << pt.x << "\t" << pt.y;
        std::cout << "]\n";
    }

    cuars::computeArsTec2d(translCandidates, pointsSrc, pointsDst, translParams);

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
