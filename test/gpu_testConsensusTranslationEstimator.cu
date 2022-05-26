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
#include "cudars/utils.h"
#include "cudars/ars2d.cuh"
#include "cudars/ars2d.h"
#include "cudars/ConsensusTranslationEstimator.cuh"
#include <rofl/common/param_map.h>
#include "cudars/mpeg7RW.h"


int main(int argc, char **argv)
{
    // cudars::ConsensusTranslationEstimator2d translEstim;
    cudars::VecVec2d pointsSrc, pointsDst, translCandidates;
    // cudars::VecVec2d translCandidates;
    cudars::Vec2d translTrue, translArs;
    rofl::ParamMap params;
    std::string filenameCfg;

    cudars::ArsTec2dParams translParams;
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
        cudars::Vec2d p;
        cudars::fillVec2d(p, (1.0 + 0.4 * i), (-2.0 - 0.35 * i));
        pointsSrc.push_back(p);
        pointsDst.push_back(cudars::vec2sumWRV(p, translParams.translGt));
    }
    pointsDst.push_back(make_double2(0.0, 0.0));
    pointsDst.push_back(make_double2(4.0, 4.0));
    pointsSrc.push_back(make_double2(3.2, 6.2));
    pointsSrc.push_back(make_double2(3.5, 2.6));

    std::cout << "Source point set:\n";
    for (auto &pt : pointsSrc)
    {
        std::cout << "  [";
        // cudars::printVec2d(pt);
        std::cout << pt.x << "\t" << pt.y;
        std::cout << "]\n";
    }
    std::cout << "Destination point set:\n";
    for (auto &pt : pointsDst)
    {
        std::cout << "  [";
        // cudars::printVec2d(pt);
        std::cout << pt.x << "\t" << pt.y;
        std::cout << "]\n";
    }

    const double rotArs = 0.0; //this is a Dummy example

    ArsImgTests::PointReaderWriter src(pointsSrc);
    ArsImgTests::PointReaderWriter dst(pointsDst);

    cudars::computeArsTec2d(translArs, rotArs, src, dst, translParams);

    // std::cout << "Estimated translation values:\n";
    // for (auto &pt : translCandidates)
    // {
    //     std::cout << "  [";
    //     // cudars::printVec2d(pt);
    //     std::cout << pt.x << "\t" << pt.y;
    //     std::cout << "]\n";
    // }

    // std::cout << "translTrue:" << std::endl;
    translTrue = translParams.translGt;
    cudars::printVec2d(translTrue, "translTrue");

    // std::cout << "translArs:" << std::endl;
    cudars::printVec2d(translArs, "translArs");

    return 0;
}
