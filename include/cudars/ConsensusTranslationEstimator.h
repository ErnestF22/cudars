// /**
//  * ARS - Angular Radon Spectrum
//  * Copyright (C) 2017 Dario Lodi Rizzini.
//  *
//  * ARS is free software: you can redistribute it and/or modify
//  * it under the terms of the GNU Lesser General Public License as published by
//  * the Free Software Foundation, either version 3 of the License, or
//  * (at your option) any later version.
//  *
//  * ARS is distributed in the hope that it will be useful,
//  * but WITHOUT ANY WARRANTY; without even the implied warranty of
//  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  * GNU Lesser General Public License for more details.
//  *
//  * You should have received a copy of the GNU Lesser General Public License
//  * along with ARS.  If not, see <http://www.gnu.org/licenses/>.
//  */
// #ifndef CONSENSUS_TRANSLATION_ESTIMATOR_H
// #define CONSENSUS_TRANSLATION_ESTIMATOR_H

// #include <iostream>
// #include <vector>
// #include "cudars/definitions.h"
// #include <rofl/common/grid.h>
// #include <rofl/common/peak_finder_d.h>

// namespace cudars
// {

//     /**
//      * Class ConsensusTranslationEstimator provides a simple solution for
//      * computing the translation between two point sets, which we assume to
//      * have the same orientation.
//      * The principle is simple: given the source points pointsSrc[i],
//      * the destination points pointsDst[i], with perfect correspondence between them,
//      * they are related by translation t
//      *
//      *  pointsDst[i] = pointsSrc[i] + t  -> t = pointsDst[i] - pointsSrc[i]
//      *
//      * Since we do not have such clear correspondence (some outlier may not have
//      * association), what we compute all the vectors
//      *
//      *   t[i][j] = pointsDst[j] - pointsSrc[i]
//      *
//      * When the pair (i, j) are right correspondences, t[i][j] is (close to)
//      * the translation value. We assume that this is the majority of cases.
//      * Thus, ConsensusTranslationEstimator computes the consensus using a
//      * classic voting grid.
//      * Translation vector candidates correspond to maxima in grid.
//      */

//     template <size_t Dim, typename Scalar = double>
//     class ConsensusTranslationEstimator
//     {
//     public:
//         using Index = int;
//         using Counter = size_t;
//         using Grid = rofl::Grid<Dim, Counter, Index, rofl::detail::RasterIndexer<2, Index>, std::vector, std::allocator>;
//         using Indices = typename Grid::Indices;
//         using PeakFinder = rofl::PeakFinderD<Dim, Counter, Index, std::greater<Index>>;

//         // using Point =  Eigen::Matrix<Scalar, Dim, 1>;
//         typedef typename MakePt<Scalar, 2>::type Point;
//         using VectorPoint = thrust::host_vector<Point>;

//         /**
//          * Default constructor.
//          */
//         ConsensusTranslationEstimator() : grid_(), translRes_(1.0), peakFinder_()
//         {
//             translMin_ = make_double2(0.0, 0.0);
//         }

//         ConsensusTranslationEstimator(const Point &translMin, const Scalar &translRes, const Indices &gridSize)
//             : grid_(), translMin_(translMin), translRes_(translRes), peakFinder_()
//         {
//             grid_.initBounds(gridSize);
//             peakFinder_.setDomain(gridSize);
//         }

//         ConsensusTranslationEstimator(const Grid &grid, const PeakFinder &peakF, const Point &translMin, const Scalar &translRes, const Indices &gridSize)
//             : grid_(), translMin_(translMin), translRes_(translRes), peakFinder_()
//         {
//             grid_.initBounds(gridSize);
//             peakFinder_.setDomain(gridSize);

//             grid_ = grid;
//             peakFinder_ = peakF;
//         }

//         virtual ~ConsensusTranslationEstimator()
//         {
//         }

//         void init(const Point &translMin, const Scalar &translRes, const Indices &gridSize)
//         {
//             grid_.initBounds(gridSize);
//             translMin_ = translMin;
//             translRes_ = translRes;
//             peakFinder_.setDomain(gridSize);
//         }

//         void reset()
//         {
//             grid_.fill(0);
//         }

//         void setNonMaximaWindowDim(const Indices &dim)
//         {
//             peakFinder_.setPeakWindow(dim);
//         }

//         void setupPickFilter(const VecVec2d &pointsSrc, const VecVec2d &pointsDst)
//         {
//             Counter thres = std::min(pointsSrc.size(), pointsDst.size()) / 2; // TODO: re-add these 2 lines
//             peakFinder_.enableFilterPeakMin(true, thres);
//         }

//         void insert(const VectorPoint &pointsSrc, const VectorPoint &pointsDst, bool adaptive = false)
//         {
//             Point transl, translMax, srcMin, srcMax, dstMin, dstMax;
//             Indices indices, gridSize;

//             if (adaptive)
//             {
//                 // srcMin.fill(std::numeric_limits<Scalar>::max());
//                 // srcMax.fill(std::numeric_limits<Scalar>::lowest());
//                 fillVec2d(srcMin, std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::max());
//                 fillVec2d(srcMax, std::numeric_limits<Scalar>::lowest(), std::numeric_limits<Scalar>::lowest());
//                 for (auto &p : pointsSrc)
//                 {
//                     for (int d = 0; d < Dim; ++d)
//                     {
//                         // if (p(d) < srcMin(d))
//                         //     srcMin(d) = p(d);
//                         // if (p(d) > srcMax(d))
//                         //     srcMax(d) = p(d);
//                         if (idxGetter(p, d) < idxGetter(p, d))
//                             idxSetter(srcMin, d, idxGetter(p, d));
//                         if (idxGetter(p, d) > idxGetter(srcMax, d))
//                             idxSetter(srcMax, d, idxGetter(p, d));
//                     }
//                 }
//                 // dstMin.fill(std::numeric_limits<Scalar>::max());
//                 // dstMax.fill(std::numeric_limits<Scalar>::lowest());
//                 fillVec2d(dstMin, std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::max());
//                 fillVec2d(dstMax, std::numeric_limits<Scalar>::lowest(), std::numeric_limits<Scalar>::lowest());
//                 for (auto &p : pointsDst)
//                 {
//                     for (int d = 0; d < Dim; ++d)
//                     {
//                         if (idxGetter(p, d) < idxGetter(p, d))
//                             idxSetter(dstMin, d, idxGetter(p, d));
//                         if (idxGetter(p, d) > idxGetter(srcMax, d))
//                             idxSetter(dstMax, d, idxGetter(p, d));
//                     }
//                 }
//                 // translMin_ = dstMin - srcMax;
//                 vec2diff(translMin_, dstMin, srcMax);
//                 // translMax = dstMax - srcMin;
//                 vec2diff(translMax, dstMax, srcMin);

//                 for (int d = 0; d < Dim; ++d)
//                 {
//                     // gridSize[d] = (Index)ceil((translMax(d) - translMin_(d)) / translRes_);
//                     gridSize[d] = (Index)ceil((idxGetter(translMax, d) - idxGetter(translMin_, d)) / translRes_);
//                 }
//                 //                ARS_VAR5(translMin_.transpose(), translMax.transpose(), translRes_, gridSize[0], gridSize[1]);
//                 init(translMin_, translRes_, gridSize);
//             }

//             for (auto &ps : pointsSrc)
//             {
//                 for (auto &pd : pointsDst)
//                 {
//                     // transl = pd - ps;
//                     vec2diff(transl, pd, ps);
//                     indices = getIndices(transl);
//                     // ARS_VARIABLE4(transl.transpose(),indices[0],indices[1],grid_.inside(indices));
//                     if (grid_.inside(indices))
//                     {
//                         grid_.value(indices)++;
//                     }
//                 }
//             }
//             Counter thres = std::min(pointsSrc.size(), pointsDst.size()) / 2;
//             peakFinder_.enableFilterPeakMin(true, thres);
//         }

//         void computeMaximaInd(std::vector<Indices> &indicesMax)
//         {
//             auto histoMap = [&](const Indices &indices) -> Counter
//             {
//                 // ARS_VARIABLE3(indices[0], indices[1], grid_.inside(indices));
//                 return grid_.value(indices);
//             };
//             peakFinder_.detect(histoMap, std::back_inserter(indicesMax));
//             //            ARS_PRINT("Maxima:");
//             //            for (auto &idx : indicesMax) {
//             //                std::cout << "  indices [" << idx[0] << "," << idx[1] << "] value " << histoMap(idx)
//             //                        << " grid2.value() " << grid_.value(idx) << std::endl;
//             //            }
//         }

//         void computeMaxima(VectorPoint &translMax)
//         {
//             std::vector<Indices> indicesMax;
//             computeMaximaInd(indicesMax);
//             translMax.clear();
//             translMax.reserve(indicesMax.size());
//             for (auto idx : indicesMax)
//             {
//                 Point p = getTranslation(idx);
//                 //                ARS_VAR4(idx[0], idx[1], grid_.value(idx), p.transpose());
//                 translMax.push_back(p);
//             }
//         }

//         Indices getIndices(const Point &p) const
//         {
//             Indices indices;
//             for (int d = 0; d < Dim; ++d)
//             {
//                 // indices[d] = round((p(d) - translMin_(d)) / translRes_);
//                 indices[d] = round((idxGetter(p, d) - idxGetter(translMin_, d)) / translRes_);
//             }
//             return indices;
//         }

//         Point getTranslation(const Indices &indices) const
//         {
//             Point transl;
//             for (int d = 0; d < Dim; ++d)
//             {
//                 // transl(d) = translRes_ * indices[d] + translMin_(d);
//                 idxSetter(transl, d, translRes_ * indices[d] + idxGetter(translMin_, d));
//             }
//             return transl;
//         }

//         Counter getScore(const Point &p) const
//         {
//             Indices indices = getIndices(p);
//             return getScore(indices);
//         }

//         Counter getScore(const Indices &indices) const
//         {
//             return grid_.value(indices);
//         }

//         const Grid &getGrid() const
//         {
//             return grid_;
//         }

//         const PeakFinder &getPeakFinder() const
//         {
//             return peakFinder_;
//         }

//         const Point &getTranslMin() const
//         {
//             return translMin_;
//         }

//         const Scalar &getTranslRes() const
//         {
//             return translRes_;
//         }

//     private:
//         Grid grid_;
//         Point translMin_;
//         Scalar translRes_;
//         PeakFinder peakFinder_;
//     };

//     using ConsensusTranslationEstimator2f = ConsensusTranslationEstimator<2, float>;
//     using ConsensusTranslationEstimator2d = ConsensusTranslationEstimator<2, double>;
//     using ConsensusTranslationEstimator3f = ConsensusTranslationEstimator<3, float>;
//     using ConsensusTranslationEstimator3d = ConsensusTranslationEstimator<3, double>;

// } // end of namespace

// #endif /* CONSENSUS_TRANSLATION_ESTIMATOR_H */