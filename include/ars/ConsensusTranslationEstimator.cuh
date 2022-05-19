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
#include <vector>
#include <ars/definitions.h>
#include <rofl/common/grid.h>
#include <rofl/common/peak_finder_d.h>

namespace cuars
{

    using Point = typename MakePt<Scalar, 2>::type; // for now we work with 2d
    using VectorPoint = thrust::host_vector<Point>;

    /**
     * @brief Initial init (used in main function() )
     */
    template <typename Grid, typename Indices, typename PeakFinder, size_t Dim, typename Scalar = double>
    void init(Grid &grid, PeakFinder &peakFinder, const Indices &gridSize, const Indices &gridWin)
    {
        grid.initBounds(gridSize);
        // translMin_ = translMin; //translMin, translRes are used directly in the followings
        // translRes_ = translRes;
        peakFinder.setDomain(gridSize);

        // translEstim.setNonMaximaWindowDim(gridWin);
        peakFinder.setPeakWindow(gridWin);
    }

    /**
     * @brief Re-init (used inside insert() when adaptive is true)
     *
     * @tparam Grid
     * @tparam Indices
     * @tparam PeakFinder
     * @tparam Dim
     * @tparam Scalar
     * @param grid
     * @param peakFinder
     * @param gridSize
     */
    template <typename Grid, typename Indices, typename PeakFinder, size_t Dim, typename Scalar = double>
    void adaptInit(Grid &grid, PeakFinder &peakFinder, const Indices &gridSize)
    {
        grid.initBounds(gridSize);
        // translMin_ = translMin; //translMin, translRes are used directly in the followings
        // translRes_ = translRes;
        peakFinder.setDomain(gridSize);
    }

    /**
     * Return indices corresponding to point @param p, according to grid params @param translMin, @param translRes
     */
    template <typename Indices, size_t Dim>
    Indices getIndices(const Point &p, const Point &translMin, const Scalar &translRes)
    {
        Indices indices;
        for (int d = 0; d < Dim; ++d)
        {
            // indices[d] = round((p(d) - translMin(d)) / translRes);
            indices[d] = round((idxGetter(p, d) - idxGetter(translMin, d)) / translRes);
        }
        return indices;
    }

    /**
     * Insert points into the grid, incrementing counter of to corresponding grid cell
     * Also, calls enableFilterPeakMin() function, needed for the correct functioning of the peak finder
     * If @param adaptive is true, adaptation is performed before inserting points
     */
    template <typename Grid, typename Indices, typename PeakFinder, size_t Dim, typename Scalar = double>
    void insert(const VectorPoint &pointsSrc, const VectorPoint &pointsDst, PeakFinder &pf, Grid &grid, Point &translMin, Scalar &translRes, bool adaptive = false)
    {
        Point transl, translMax, srcMin, srcMax, dstMin, dstMax;
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
            vec2diff(translMin, dstMin, srcMax);
            // translMax = dstMax - srcMin;
            vec2diff(translMax, dstMax, srcMin);

            for (int d = 0; d < Dim; ++d)
            {
                // gridSize[d] = (Index)ceil((translMax(d) - translMin(d)) / translRes);
                gridSize[d] = (Index)ceil((idxGetter(translMax, d) - idxGetter(translMin, d)) / translRes);
            }
            //                ARS_VAR5(translMin.transpose(), translMax.transpose(), translRes, gridSize[0], gridSize[1]);

            // init(translMin, translRes, gridSize);
            adaptInit<Grid, Indices, PeakFinder, Dim, Scalar>(grid, pf, gridSize);
        }

        // translEstim.setupPickFilter(pointsSrc, pointsDst);
        Counter thres = std::min(pointsSrc.size(), pointsDst.size()) / 2; // TODO (maybe): move this line and the one below to an "init" function
        pf.enableFilterPeakMin(true, thres);

        for (auto &ps : pointsSrc)
        {
            for (auto &pd : pointsDst)
            {
                // transl = pd - ps;
                vec2diff(transl, pd, ps);
                indices = getIndices<Indices, Dim>(transl, translMin, translRes);
                // ARS_VARIABLE4(transl.transpose(),indices[0],indices[1],grid_.inside(indices));
                if (grid.inside(indices))
                {
                    grid.value(indices)++;
                }
            }
        }
    }

    /**
     * @brief New version, now outside of TEC class, of method: Point getTranslation(const Indices &indices) const;
     */
    template <typename Indices, size_t Dim, typename Scalar = double>
    Point getTranslation(const Indices &indices, Point &translMin, Scalar &translRes)
    {
        Point transl;

        for (int d = 0; d < Dim; ++d)
        {
            // transl(d) = translRes * indices[d] + translMin(d);
            idxSetter(transl, d, translRes * indices[d] + idxGetter(translMin, d));
        }

        return transl;
    }

    /**
     * @brief Calls peakFinder detection method on @param grid
     * It is called internally from function computeMaxima()
     */
    template <typename Grid, typename Indices, typename PeakFinder, size_t Dim, typename Scalar = double>
    void computeMaximaInd(std::vector<Indices> &indicesMax, Grid &grid, PeakFinder &peakFinder)
    {
        auto histoMap = [&](const Indices &indices) -> Counter
        {
            // ARS_VARIABLE3(indices[0], indices[1], grid.inside(indices));
            return grid.value(indices);
        };
        peakFinder.detect(histoMap, std::back_inserter(indicesMax));

        //            ARS_PRINT("Maxima:");
        //            for (auto &idx : indicesMax) {
        //                std::cout << "  indices [" << idx[0] << "," << idx[1] << "] value " << histoMap(idx)
        //                        << " grid2.value() " << grid.value(idx) << std::endl;
        //            }
    }

    /**
     * @brief Adaptation outside TEC class of its main computation method
     */
    template <typename Grid, typename Indices, typename PeakFinder, size_t Dim, typename Scalar = double>
    void computeMaxima(VectorPoint &translMax, Grid &grid, PeakFinder &peakFinder, Point &translMin, Scalar &translRes)
    {
        std::vector<Indices> indicesMax;
        computeMaximaInd<Grid, Indices, PeakFinder, Dim>(indicesMax, grid, peakFinder);

        translMax.clear();
        translMax.reserve(indicesMax.size());
        for (auto idx : indicesMax)
        // for (Counter i = 0; i < indicesMax.size(); ++i) //instead of using foreach, this standard for could be used...
        {
            // Indices idx = indicesMax.at(i); //... together with this additional variable/line

            Point p = getTranslation<Indices, Dim>(idx, translMin, translRes); // expanding this function below

            //                ARS_VAR4(idx[0], idx[1], grid.value(idx), p.transpose());
            translMax.push_back(p);
        }
    }

    // template Point getTranslation<Indices2d, 2>(const Indices2d &indices, Point &translMin, Scalar &translRes);
    // template void computeMaximaInd<Grid2d, Indices2d, PeakFinder2d, 2>(std::vector<Indices2d> &indicesMax, Grid2d &grid, PeakFinder2d &peakFinder);
    // template void computeMaxima<Grid2d, Indices2d, PeakFinder2d, 2>(VectorPoint &translMax, Grid2d &grid, PeakFinder2d &peakFinder, Point &translMin, Scalar &translRes); // explicit instantiation for 2d version of computeMaxima()

    /**
     * Wrappers with explicited dimensions, floating-point precision, useful/used when calling functions for main
     * for improved clarity
     */
    void init2d(Grid2d &grid, PeakFinder2d &peakFinder, const Indices2d &gridSize, const Indices2d &gridWin)
    {
        init<Grid2d, Indices2d, PeakFinder2d, 2, Scalar>(grid, peakFinder, gridSize, gridWin);
    }

    void insert2d(const VectorPoint &pointsSrc, const VectorPoint &pointsDst, PeakFinder2d &pf, Grid2d &grid, Point &translMin, Scalar &translRes, bool adaptive = false)
    {
        insert<Grid2d, Indices2d, PeakFinder2d, 2, Scalar>(pointsSrc, pointsDst, pf, grid, translMin, translRes, adaptive);
    }

    void computeMaxima2d(VectorPoint &translMax, Grid2d &grid, PeakFinder2d &peakFinder, Point &translMin, Scalar &translRes)
    {
        computeMaxima<Grid2d, Indices2d, PeakFinder2d, 2>(translMax, grid, peakFinder, translMin, translRes);
    }

} // end of namespace

#endif /* CONSENSUS_TRANSLATION_ESTIMATOR_CUH */
