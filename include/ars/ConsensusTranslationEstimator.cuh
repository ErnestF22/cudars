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
#ifndef CONSENSUS_TRANSLATION_ESTIMATOR_CUH
#define CONSENSUS_TRANSLATION_ESTIMATOR_CUH

#include <iostream>
#include <fstream>
#include <vector>
#include "ars/definitions.h"
#include "ars/Grid.cuh"
#include <rofl/common/peak_finder_d.h>

namespace cuars
{
    using Point = typename MakePt<Scalar, 2>::type; // for now we work with 2d
    using VectorPoint = thrust::host_vector<Point>;

    using Index = int;
    using Counter = size_t;

    // using Grid2d = rofl::Grid<2, Counter, Index, rofl::detail::RasterIndexer<2, Index>, std::vector, std::allocator>;
    using Grid2d = cuars::Grid;
    using Indices2d = std::array<Index, DIM>;
    using PeakFinder2d = rofl::PeakFinderD<2, Counter, Index, std::greater<Index>>;

    struct ArsTec2dParams
    {
        cuars::Vec2d translMin, translMax, translGt;
        double translRes;
        cuars::Indices2d gridSize, gridWin;
        bool adaptiveGrid;
        bool plot;
    };

        template <typename Grid, typename Indices, typename PeakFinder, size_t Dim, typename Scalar = double>
    struct ArsTec
    {
        /*members*/
        Grid grid_;
        Point translMin_;
        Point translMax_;
        Scalar translRes_;
        PeakFinder peakFinder_;

        /**
         * Default constructor.
         */
        ArsTec() : grid_(), translRes_(1.0), peakFinder_()
        {
            translMin_.x = 0.0;
            translMin_.y = 0.0;
        }

        ArsTec(const Point &translMin, const Scalar &translRes, const Indices &gridSize)
            : grid_(), translMin_(translMin), translRes_(translRes), peakFinder_()
        {
            grid_.initBounds(gridSize);
            peakFinder_.setDomain(gridSize);
        }

        virtual ~ArsTec()
        {
        }

        void setNonMaximaWindowDim(const Indices &dim)
        {
            peakFinder_.setPeakWindow(dim);
        }

        /**
         * @brief Initial init (used in main function() )
         */
        void init(const ArsTec2dParams &translParams)
        {
            grid_.initBounds(translParams.gridSize);

            translMin_ = translParams.translMin;
            translMax_ = translParams.translMax;
            translRes_ = translParams.translRes;

            peakFinder_.setDomain(translParams.gridSize);
            // translEstim.setNonMaximaWindowDim(gridWin);

            setNonMaximaWindowDim(translParams.gridWin);
        }

        /**
         * @brief Re-init (called from insert() when @bool adaptive is true)
         */
        void adaptInit(const Indices &gridSize)
        {
            // grid_.reset();
            grid_.initBounds(gridSize);
            // translMin_ = translMin; //translMin, translRes are used directly in the followings
            // translRes_ = translRes;
            peakFinder_.setDomain(gridSize);
        }

        void potentialKernel(const VectorPoint &pointsSrc, const VectorPoint &pointsDst, Point &transl, Indices &indices)
        {
            for (auto &ps : pointsSrc)
            {
                for (auto &pd : pointsDst)
                {
                    // transl = pd - ps;
                    vec2diff(transl, pd, ps);
                    indices = getIndices(transl);
                    // ARS_VARIABLE4(transl.transpose(),indices[0],indices[1],grid_.inside(indices));
                    if (grid_.insideGrid(indices))
                    {
                        grid_.value(indices)++;
                    }
                }
            }
        }

        /**
         * Insert points into the grid, incrementing counter of to corresponding grid cell
         * Also, calls enableFilterPeakMin() function, needed for the correct functioning of the peak finder
         * If @param adaptive is true, adaptation is performed before inserting points
         */
        void insert(const VectorPoint &pointsSrc, const VectorPoint &pointsDst, bool adaptive = false)
        {
            Point transl, srcMin, srcMax, dstMin, dstMax;
            // Point translMax;
            Indices indices, gridSize;

            if (adaptive)
            {
                // srcMin.fill(std::numeric_limits<Scalar>::max());
                // srcMax.fill(std::numeric_limits<Scalar>::lowest());
                fillVec2d(srcMin, std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::max());
                fillVec2d(srcMax, std::numeric_limits<Scalar>::lowest(), std::numeric_limits<Scalar>::lowest());
                for (auto &p : pointsSrc)
                {
                    for (int d = 0; d < Dim; ++d)
                    {
                        // if (p(d) < srcMin(d))
                        //     srcMin(d) = p(d);
                        // if (p(d) > srcMax(d))
                        //     srcMax(d) = p(d);
                        if (idxGetter(p, d) < idxGetter(p, d))
                            idxSetter(srcMin, d, idxGetter(p, d));
                        if (idxGetter(p, d) > idxGetter(srcMax, d))
                            idxSetter(srcMax, d, idxGetter(p, d));
                    }
                }
                // dstMin.fill(std::numeric_limits<Scalar>::max());
                // dstMax.fill(std::numeric_limits<Scalar>::lowest());
                fillVec2d(dstMin, std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::max());
                fillVec2d(dstMax, std::numeric_limits<Scalar>::lowest(), std::numeric_limits<Scalar>::lowest());
                for (auto &p : pointsDst)
                {
                    for (int d = 0; d < Dim; ++d)
                    {
                        if (idxGetter(p, d) < idxGetter(p, d))
                            idxSetter(dstMin, d, idxGetter(p, d));
                        if (idxGetter(p, d) > idxGetter(srcMax, d))
                            idxSetter(dstMax, d, idxGetter(p, d));
                    }
                }
                // translMin = dstMin - srcMax;
                vec2diff(translMin_, dstMin, srcMax);
                // translMax = dstMax - srcMin;
                vec2diff(translMax_, dstMax, srcMin);

                for (int d = 0; d < Dim; ++d)
                {
                    // gridSize[d] = (Index)ceil((translMax(d) - translMin(d)) / translRes);
                    gridSize[d] = (Index)ceil((idxGetter(translMax_, d) - idxGetter(translMin_, d)) / translRes_);
                }
                //                ARS_VAR5(translMin.transpose(), translMax.transpose(), translRes, gridSize[0], gridSize[1]);

                // init(translMin, translRes, gridSize);
                adaptInit(gridSize);
            }

            potentialKernel(pointsSrc, pointsDst, transl, indices);

            // translEstim.setupPickFilter(pointsSrc, pointsDst);
            Counter thres = std::min(pointsSrc.size(), pointsDst.size()) / 2; // TODO (maybe): move this line and the one below to an "init" function
            peakFinder_.enableFilterPeakMin(true, thres);
        }

        /**
         * @brief Calls peakFinder detection method on @param grid
         * It is called internally from function computeMaxima()
         */
        void computeMaximaInd(std::vector<Indices> &indicesMax)
        {
            auto histoMap = [&](const Indices &indices) -> Counter
            {
                // ARS_VARIABLE3(indices[0], indices[1], grid.inside(indices));
                return grid_.value(indices);
            };
            peakFinder_.detect(histoMap, std::back_inserter(indicesMax));

            //            ARS_PRINT("Maxima:");
            //            for (auto &idx : indicesMax) {
            //                std::cout << "  indices [" << idx[0] << "," << idx[1] << "] value " << histoMap(idx)
            //                        << " grid2.value() " << grid.value(idx) << std::endl;
            //            }
        }

        /**
         * @brief Adaptation outside TEC class of its main computation method
         */
        void computeMaxima(VectorPoint &translMax)
        {
            std::vector<Indices> indicesMax;
            computeMaximaInd(indicesMax);

            translMax.clear();
            translMax.reserve(indicesMax.size());
            // int ctr = 0;
            for (auto idx : indicesMax)
            // for (Counter i = 0; i < indicesMax.size(); ++i) //instead of using foreach, this standard for could be used...
            {
                // Indices idx = indicesMax.at(i); //... together with this additional variable/line

                Point p = getTranslation(idx); // expanding this function below

                //                ARS_VAR4(idx[0], idx[1], grid.value(idx), p.transpose());
                translMax.push_back(p);

                // std::cout << "p " << ctr << " - " << p.x << " " << p.y << std::endl;
                // ctr++;
            }
        }

        /**
         * Return indices corresponding to point @param p, according to grid params @param translMin, @param translRes
         */
        Indices getIndices(const Point &p)
        {
            Indices indices;
            for (int d = 0; d < Dim; ++d)
            {
                // indices[d] = round((p(d) - translMin(d)) / translRes);
                indices[d] = round((idxGetter(p, d) - idxGetter(translMin_, d)) / translRes_);
            }
            return indices;
        }

        /**
         * @brief New version, now outside of TEC class, of method: Point getTranslation(const Indices &indices) const;
         */
        Point getTranslation(const Indices &indices)
        {
            Point transl;

            for (int d = 0; d < Dim; ++d)
            {
                // transl(d) = translRes * indices[d] + translMin(d);
                idxSetter(transl, d, translRes_ * indices[d] + idxGetter(translMin_, d));
            }

            return transl;
        }

        Counter getScore(const Point &p) const
        {
            Indices indices = getIndices(p);
            return getScore(indices);
        }

        Counter getScore(const Indices &indices) const
        {
            return grid_.value(indices);
        }

        const Grid &getGrid() const
        {
            return grid_;
        }

        // template Point getTranslation<Indices2d, 2>(const Indices2d &indices, Point &translMin, Scalar &translRes);
        // template void computeMaximaInd<Grid2d, Indices2d, PeakFinder2d, 2>(std::vector<Indices2d> &indicesMax, Grid2d &grid, PeakFinder2d &peakFinder);
        // template void computeMaxima<Grid2d, Indices2d, PeakFinder2d, 2>(VectorPoint &translMax, Grid2d &grid, PeakFinder2d &peakFinder, Point &translMin, Scalar &translRes); // explicit instantiation for 2d version of computeMaxima()
    };

    /**
     * Wrappers with explicited dimensions, floating-point precision, useful/used when calling functions for main
     * for improved clarity
     */
    // void init2d(Grid2d &grid, PeakFinder2d &peakFinder, const Indices2d &gridSize, const Indices2d &gridWin)
    // {
    //     ArsTec<Grid2d, Indices2d, PeakFinder2d, 2, Scalar>::init(grid, peakFinder, gridSize, gridWin);
    // }

    // void insert2d(const VectorPoint &pointsSrc, const VectorPoint &pointsDst, PeakFinder2d &pf, Grid2d &grid, Point &translMin, Scalar &translRes, bool adaptive = false)
    // {
    //     ArsTec::insert<Grid2d, Indices2d, PeakFinder2d, 2, Scalar>(pointsSrc, pointsDst, pf, grid, translMin, translRes, adaptive);
    // }

    // void computeMaxima2d(VectorPoint &translMax, Grid2d &grid, PeakFinder2d &peakFinder, Point &translMin, Scalar &translRes)
    // {
    //     ArsTec::computeMaxima<Grid2d, Indices2d, PeakFinder2d, 2>(translMax, grid, peakFinder, translMin, translRes);
    // }

    // TODO: make ArsTec2d object specifying some template struct params

    /**
     * @brief Support visual output method (for debugging) used inside computeArsTec     *
     *
     * @param arsTec
     * @param translMin
     * @param translRes
     * @param filename
     * @param factor
     */
    void plotGrid2d(const ArsTec<cuars::Grid2d, Indices2d, PeakFinder2d, 2> &arsTec, const cuars::Vec2d &translMin, double translRes, const std::string &filename, double factor)
    {
        Grid2d grid = arsTec.grid_;
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

    /**
     * @brief Main computation method, ideally the only one that should be "public"
     *
     * @param translCandidates
     * @param pointsSrc
     * @param pointsDst
     * @param translParams
     */
    void computeArsTec(VecVec2d &translCandidates, const VecVec2d &pointsSrc, const VecVec2d &pointsDst, ArsTec2dParams &translParams)
    {
        ArsTec<Grid2d, cuars::Indices2d, cuars::PeakFinder2d, 2> translObj; // ArsTec 2D object

        // translEstim.init(translMin, translRes, gridSize);
        // translEstim.setNonMaximaWindowDim(gridWin);
        translObj.init(translParams); // for now init is included

        std::cout << "Inserting pair source-destination:\n";
        // translEstim.insert(pointsSrc, pointsDst);
        translObj.insert(pointsSrc, pointsDst, translParams.adaptiveGrid); // adaptive = false for the dummy example

        if (translParams.plot)
        {
            //     translObj.ConsensusTranslationEstimator2d translEstimPlot(grid, pf, translParams.translMin, translParams.translRes, translParams.gridSize);
            plotGrid2d(translObj, translParams.translMin, translParams.translRes, "consensus_transl_grid.plot", 1.0);
        }

        std::cout << "Computing maxima:\n";
        // translEstim.computeMaxima(translCandidates); //TODO: adapt computeMaxima() for CUDA GPU parallelization
        // cuars::computeMaxima<cuars::Grid2d, cuars::Indices2d, cuars::PeakFinder2d, 2>(translCandidates, grid, peakF, translMin, translRes);
        translObj.computeMaxima(translCandidates); // TODO: adapt computeMaxima() for CUDA GPU parallelization
    }

} // end of namespace

#endif /* CONSENSUS_TRANSLATION_ESTIMATOR_CUH */
