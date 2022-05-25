#ifndef GRID_CUH
#define GRID_CUH

#include <array>
#include <vector>

#include "IntervalIndices.cuh"

namespace cuars
{
    // TODO (maybe): switch these declarations to template-approach
    using Index = int;
    static const int DIM = 2;
    using Indices = std::array<Index, DIM>;

    using IntervalType = IntervalIndices<DIM, Index>;

    using Indexer = detail::RasterIndexer<DIM, Index>;
    using IndexerType = Indexer; // TODO later: remove this line appropriately

    using Value = size_t;

    struct Grid // cuars::Grid is already specialized as Grid2d
    {
        IntervalType domain_;
        std::vector<size_t, std::allocator<size_t>> data_;

        Grid() : data_(), domain_()
        {
        }

        Grid(const Indices &min, const Indices &dimensions) : data_(), domain_(min, dimensions)
        {
            data_.resize(domain_.size());
        }

        Grid(const Indices &dimensions) : data_(), domain_()
        {
            Indices zeros;
            zeros.fill(0);
            domain_.initBounds(zeros, dimensions);
            data_.resize(domain_.size());
        }

        virtual ~Grid()
        {
        }

        void initBounds(const Indices &dimensions)
        {
            Indices zeros;
            zeros.fill(0);
            domain_.initBounds(zeros, dimensions);
            data_.resize(domain_.size());
        }

        void initBounds(const Indices &min, const Indices &dimensions)
        {
            domain_.initBounds(min, dimensions);
            data_.resize(domain_.size());
        }

        void initCenter(const Indices &icenter, const Indices &iwin)
        {
            domain_.initCentered(icenter, iwin);
            data_.resize(domain_.size());
        }

        void initMinMax(const Indices &imin, const Indices &imax)
        {
            domain_.initMinMax(imin, imax);
            data_.resize(domain_.size());
        }

        size_t size() const
        {
            return data_.size();
        }

        const Indices &dimensions() const
        {
            return domain_.dimensions();
        }

        /**
         * Says if the given indices are inside the index domain.
         * @param indices the input indices
         * @return true if the given indices are inside the domain
         */
        bool insideGrid(const Indices &indices) const
        {
            return domain_.insideDomain(indices);
        }

        /**
         * Warning: the user is responsible about the correct access to the buffer.
         * This function is defined for more efficient access to data by avoiding Indices
         */
        const Value &value(const Index *indices) const
        {
            return data_[detail::StaticRasterIndexer<DIM, Index>::getPos(domain_.min().data(), domain_.dimensions().data(), indices)];
        }

        /**
         * Warning: the user is responsible about the correct access to the buffer.
         * This function is defined for more efficient access to data.
         */
        Value &value(const Index *indices)
        {
            return data_[detail::StaticRasterIndexer<DIM, Index>::getPos(domain_.min().data(), domain_.dimensions().data(), indices)];
        }

        const Value &value(const Indices &indices) const
        {
            //			Index pos = getPos(indices);
            //			return data_.at(pos);
            return data_[detail::StaticRasterIndexer<DIM, Index>::getPos(domain_.min().data(), domain_.dimensions().data(), indices.data())];
        }

        Value &value(const Indices &indices)
        {
            // Index pos = getPos(indices);
            // return data_.at(pos);
            return data_[detail::StaticRasterIndexer<DIM, Index>::getPos(domain_.min().data(), domain_.dimensions().data(), indices.data())];
        }

        const Value &value(const Index &pos) const
        {
            return data_.at(pos);
        }

        Value &value(const Index &pos)
        {
            return data_.at(pos);
        }

        void fill(const Value &value)
        {
            std::fill(std::begin(data_), std::end(data_), value);
        }

        Index getPos(const Indices &indices) const
        {
            return IndexerType::getPos(domain_.min(), domain_.dimensions(), indices);
        }

        Indices getIndices(const Index &index) const
        {
            return IndexerType::getIndices(domain_.min(), domain_.dimensions(), index);
        }
    };

}

#endif /*GRID_CUH*/