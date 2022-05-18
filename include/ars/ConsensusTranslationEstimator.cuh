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
    using Index = int;
    using Counter = size_t;

    typedef typename MakePt<Scalar, 2>::type Point;
    using VectorPoint = thrust::host_vector<Point>;

    template <size_t Dim, typename Scalar = double>
    void computeMaxima(VectorPoint &translMax)
    {
        using Grid = rofl::Grid<Dim, Counter, Index, rofl::detail::RasterIndexer<2, Index>, std::vector, std::allocator>;
        using Indices = typename Grid::Indices;
        using PeakFinder = rofl::PeakFinderD<Dim, Counter, Index, std::greater<Index>>;

        std::vector<Indices> indicesMax;
        computeMaxima(indicesMax);
        translMax.clear();
        translMax.reserve(indicesMax.size());
        for (auto idx : indicesMax)
        {
            Point p = getTranslation(idx);
            //                ARS_VAR4(idx[0], idx[1], grid_.value(idx), p.transpose());
            translMax.push_back(p);
        }
    }

    // Point getTranslation(const Indices &indices) const
    template <typename Indices, size_t Dim, typename Scalar = double>
    Point getTranslation(const Indices &indices, Point& translMin_, Scalar& translRes_)
    {
        Point transl;
        for (int d = 0; d < Dim; ++d)
        {
            // transl(d) = translRes_ * indices[d] + translMin_(d);
            idxSetter(transl, d, translRes_ * indices[d] + idxGetter(translMin_, d));
        }
        return transl;
    }

    // using Grid = rofl::Grid<Dim, Counter, Index, rofl::detail::RasterIndexer<2, Index>, std::vector, std::allocator>;
    // using Indices = typename Grid::Indices;
    // using PeakFinder = rofl::PeakFinderD<Dim, Counter, Index, std::greater<Index>>;

} // end of namespace

#endif /* CONSENSUS_TRANSLATION_ESTIMATOR_CUH */
