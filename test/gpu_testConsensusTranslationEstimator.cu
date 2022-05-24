#include <iostream>
#include "ars/utils.h"
#include "ars/ars2d.cuh"
#include <ars/ars2d.h>
#include <ars/ConsensusTranslationEstimator.cuh>
#include <rofl/common/param_map.h>

//!! cuars::ConsensusTranslationEstimator2d::Grid -> come usare gli using definiti nelle classi
void plotGrid(const cuars::ArsTec<cuars::Grid2d, cuars::Indices2d, cuars::PeakFinder2d, 2> & arsTec, const cuars::Vec2d &translMin, double translRes, const std::string &filename, double factor);

void computeArsTec(cuars::VecVec2d &translCandidates, const cuars::VecVec2d &pointsSrc, const cuars::VecVec2d &pointsDst, cuars::ArsTecParams &translParams);

int main(int argc, char **argv)
{
    // cuars::ConsensusTranslationEstimator2d translEstim;
    cuars::VecVec2d pointsSrc, pointsDst, translCandidates;
    cuars::Vec2d p;
    rofl::ParamMap params;
    std::string filenameCfg;

    cuars::ArsTecParams translParams;
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


    computeArsTec(translCandidates, pointsSrc, pointsDst, translParams);

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

void plotGrid(const cuars::ArsTec<cuars::Grid2d, cuars::Indices2d, cuars::PeakFinder2d, 2>& arsTec, const cuars::Vec2d &translMin, double translRes, const std::string &filename, double factor)
{
    cuars::Grid2d grid = arsTec.grid_;
    int dim0 = grid.dimensions()[0];
    int dim1 = grid.dimensions()[1];
    int dim0Reduced = round(dim0 / factor);
    int dim1Reduced = round(dim1 / factor);
    double v0, v1;

    std::ofstream file(filename);
    if (!file)
    {
        std::cerr << "Cannot open \"" << filename << "\"" << std::endl;
        return;
    }

    std::cout << "plotting grid with size " << dim0 << " x " << dim1 << std::endl;

    file << "set grid nopolar\n"
         << "set style data lines\n"
         << "set dgrid3d " << dim0Reduced << "," << dim1Reduced << "\n"
         << "set hidden3d\n";

    file << "splot '-'\n";
    for (int i0 = 0; i0 < dim0; ++i0)
    {
        for (int i1 = 0; i1 < dim1; ++i1)
        {
            // v0 = translMin(0) + translRes * i0;
            v0 = translMin.x + translRes * i0;
            // v1 = translMin(1) + translRes * i1;
            v1 = translMin.y + translRes * i1;

            file << v0 << " " << v1 << " " << grid.value({i0, i1}) << "\n";
        }
    }
    file << "e\n";

    file.close();
}

void computeArsTec(cuars::VecVec2d &translCandidates, const cuars::VecVec2d &pointsSrc, const cuars::VecVec2d &pointsDst, cuars::ArsTecParams &translParams)
{
    // template <typename Grid, typename Indices, typename PeakFinder, size_t Dim, typename Scalar = double>
    cuars::ArsTec<cuars::Grid2d, cuars::Indices2d, cuars::PeakFinder2d, 2> translObj(translParams); // ArsTec 2D object
    // translEstim.init(translMin, translRes, gridSize);
    // translEstim.setNonMaximaWindowDim(gridWin);
    // translObj.init(translParams.);

    std::cout << "Inserting pair source-destination...\n";
    // translEstim.insert(pointsSrc, pointsDst);
    translObj.insert(pointsSrc, pointsDst, translParams.adaptiveGrid); // adaptive = false for the dummy example

    // if (translParams.plot)
    // {
    //     translObj.ConsensusTranslationEstimator2d translEstimPlot(grid, pf, translParams.translMin, translParams.translRes, translParams.gridSize);
    //     plotGrid(translEstimPlot.getGrid(), translParams.translMin, translParams.translRes, "consensus_transl_grid.plot", 1.0);
    // }

    std::cout << "Computing maxima...\n";
    // translEstim.computeMaxima(translCandidates); //TODO: adapt computeMaxima() for CUDA GPU parallelization
    // cuars::computeMaxima<cuars::Grid2d, cuars::Indices2d, cuars::PeakFinder2d, 2>(translCandidates, grid, peakF, translMin, translRes);
    translObj.computeMaxima(translCandidates); // TODO: adapt computeMaxima() for CUDA GPU parallelization
}