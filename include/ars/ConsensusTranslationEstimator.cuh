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

    // template <typename Grid, typename Indices, typename PeakFinder, size_t Dim, typename Scalar = double>
    // void computeMaximaInd(std::vector<Indices> &indicesMax, Grid &grid_, PeakFinder &peakFinder_)
    // {
    //     auto histoMap = [&](const Indices &indices) -> Counter
    //     {
    //         // ARS_VARIABLE3(indices[0], indices[1], grid_.inside(indices));
    //         return grid_.value(indices);
    //     };
    //     peakFinder_.detect(histoMap, std::back_inserter(indicesMax));

    //     //            ARS_PRINT("Maxima:");
    //     //            for (auto &idx : indicesMax) {
    //     //                std::cout << "  indices [" << idx[0] << "," << idx[1] << "] value " << histoMap(idx)
    //     //                        << " grid2.value() " << grid_.value(idx) << std::endl;
    //     //            }
    // } // TODO: declare the real function

    template <typename Grid, typename Indices, typename PeakFinder, size_t Dim, typename Scalar = double>
    void computeMaxima(VectorPoint &translMax, Grid &grid, PeakFinder &peakFinder, Point &translMin_, Scalar &translRes_)
    {
        std::vector<Indices> indicesMax;
        // computeMaximaInd(indicesMax, grid, peakFinder); //expanding this function below
        auto histoMap = [&](const Indices &indices) -> Counter
        {
            // ARS_VARIABLE3(indices[0], indices[1], grid_.inside(indices));
            return grid.value(indices);
        };
        peakFinder.detect(histoMap, std::back_inserter(indicesMax));
        //end of expanded function

        translMax.clear();
        translMax.reserve(indicesMax.size());
        for (auto idx : indicesMax)
        {
            // Point p = getTranslation(idx); //expanding this function below
            Point p;
            for (int d = 0; d < Dim; ++d)
            {
                // transl(d) = translRes_ * indices[d] + translMin_(d);
                idxSetter(p, d, translRes_ * idx[d] + idxGetter(translMin_, d));
            }
            //end of expanded function

            //                ARS_VAR4(idx[0], idx[1], grid_.value(idx), p.transpose());
            translMax.push_back(p);
        }
    }

    // Point getTranslation(const Indices &indices) const
    // template <typename Indices, size_t Dim, typename Scalar = double>
    // Point getTranslation(const Indices &indices, Point &translMin_, Scalar &translRes_)
    // {
    //     Point transl;

    //     return transl;
    // } // TODO: declare the real function

    template void computeMaxima<Grid2d, Indices2d, PeakFinder2d, 2>(VectorPoint &translMax, Grid2d &grid, PeakFinder2d &peakFinder, Point &translMin_, Scalar &translRes_); // explicit instantiation for computeMaxima2d
    // template void computeMaximaInd<Grid2d, Indices2d, PeakFinder2d, 2>(std::vector<Indices2d> &indicesMax, Grid2d &grid_, PeakFinder2d &peakFinder_);
    // template Point getTranslation<Indices2d, 2>(const Indices2d &indices, Point &translMin_, Scalar &translRes_);

    // using computeMaxima2d = ;

} // end of namespace

#endif /* CONSENSUS_TRANSLATION_ESTIMATOR_CUH */
