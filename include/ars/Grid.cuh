#include <array>
#include <vector>

namespace cuars
{
    // TODO (maybe): switch these declarations to template-approach
    using Index = int;
    static const int DIM = 2;
    using Indices = std::array<Index, DIM>;

    using Value = size_t;

    // template <size_t Dim, typename Index>
    struct RasterIndexer
    {

        static Indices getIndices(const Indices &minval, const Indices &dimensions, Index pos)
        {
            Indices indices;
            size_t res = pos;
            for (Index d = 0; d < DIM; ++d)
            {
                indices[d] = minval[d] + res % dimensions[d];
                res = res / dimensions[d];
            }
            return indices;
        }

        static Index getPos(const Indices &minval, const Indices &dimensions, const Indices &indices)
        {
            Index index = 0;
            for (int d = DIM - 1; d >= 0; --d)
            {
                index = index * dimensions[d] + indices[d] - minval[d];
            }
            return index;
        }
    };

    //!! DOMAIN = INTERVALTYPE = INTERVALINDICES
    struct Domain
    {
        Indices min_;
        Indices dimensions_;

        /**
         * Default constructor.
         * It creates an empty interval.
         */
        Domain() : min_(), dimensions_()
        {
            min_.fill(0);
            dimensions_.fill(0);
        }

        /**
         * Constructor with given min indices and dimensions.
         * Example: Dim =2, min=[4,1], dimensions=[3,2] creates an interval spanning
         *   on indices[0] from 4 to 6 (end value 4+3 = 7 not included)
         *   on indices[1] from 1 to 2 (end value 1+2 = 3 not included)
         * @param min min values of indices in each coordinate
         * @param dimensions dimension sizes of indices in each coordinate
         */
        Domain(const Indices &min, const Indices &dimensions) : min_(min), dimensions_(dimensions)
        {
        }

        /**
         * Destructor.
         */
        virtual ~Domain()
        {
        }

        /**
         * Returns the number of indices items included in the interval.
         */
        size_t size()
        {
            size_t s = 1;
            for (int d = 0; d < DIM; ++d)
            {
                s *= dimensions_[d];
            }
            return s;
        }

        // TODO: c'è un'altra initBounds() per le expansion...
        /**
         * Initializes the interval with given min indices and dimensions.
         * Example: Dim =2, min=[4,1], dimensions=[3,2] creates an interval spanning
         *   on indices[0] from 4 to 6 (end value 4+3 = 7 not included)
         *   on indices[1] from 1 to 2 (end value 1+2 = 3 not included)
         * @param min min values of indices in each coordinate
         * @param dimensions dimension sizes of indices in each coordinate
         */
        void initBounds(const Indices &min, const Indices &dimensions)
        {
            min_ = min;
            dimensions_ = dimensions;
        }

        
        /**
         * Returns the dimension sizes of interval.
         */
        const Indices &dimensions() const
        {
            return dimensions_;
        }

        /**
         * Says if the given indices array is included in interval.
         */
        bool insideDomain(const Indices &indices)
        {
            for (int d = 0; d < DIM; ++d)
            {
                if (indices[d] < min_[d] || indices[d] >= min_[d] + dimensions_[d])
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * Returns the min values of indices in each coordinate.
         */
        const Indices &min() const
        {
            return min_;
        }

        /**
         * Says if the interval is empty.
         */
        bool empty() const
        {
            for (int d = 0; d < DIM; ++d)
            {
                if (dimensions_[d] <= 0)
                {
                    return true;
                }
            }
            return false;
        }

        /**
         * Returns the number of indices items included in the interval.
         */
        size_t size() const
        {
            size_t s = 1;
            for (int d = 0; d < DIM; ++d)
            {
                s *= dimensions_[d];
            }
            return s;
        }

        /**
         * Says if the given indices array is included in interval.
         */
        bool inside(const Indices &indices) const
        {
            for (int d = 0; d < DIM; ++d)
            {
                if (indices[d] < min_[d] || indices[d] >= min_[d] + dimensions_[d])
                {
                    return false;
                }
            }
            return true;
        }
    };

    struct Grid
    {
        Domain domain_;
        std::vector<size_t> data_;

        /**
         * @brief Default constructor for a new Grid object
         */
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

        /**
         * Says if the given indices are inside the index domain.
         * @param indices the input indices
         * @return true if the given indices are inside the domain
         */
        bool insideGrid(const Indices &indices)
        {
            // return domain_.inside(indices);
            return domain_.insideDomain(indices);
        }

        Index getPos(const Indices &indices) const
        {
            // return StaticRasterIndexer.getPos(domain_.min(), domain_.dimensions(), indices);
            return RasterIndexer::getPos(domain_.min(), domain_.dimensions_, indices);
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
        bool inside(const Indices &indices) const
        {
            return domain_.inside(indices);
        }

        void initBounds(const Indices& dimensions) {
            Indices zeros;
            zeros.fill(0);
            domain_.initBounds(zeros, dimensions);
            data_.resize(domain_.size());
        }

        void initBounds(const Indices& min, const Indices& dimensions) {
            domain_.initBounds(min, dimensions);
            data_.resize(domain_.size());
        }


        /**
         * Warning: the user is responsible about the correct access to the buffer.
         * This function is defined for more efficient access to data.
         */
        // Value &value(const Index *indices)
        // {
        //     // return data_[detail::StaticRasterIndexer<Dim, Index>::getPos(domain_.min().data(), domain_.dimensions().data(), indices)];
        //     return data_[RasterIndexer::getPos(domain_.min(), domain_.dimensions(), indices)];
        // }

        const Value &value(const Indices &indices) const
        {
            //			Index pos = getPos(indices);
            //			return data_.at(pos);
            return data_[RasterIndexer::getPos(domain_.min(), domain_.dimensions(), indices)];
        }

        Value &value(const Indices &indices)
        {
            // Index pos = getPos(indices);
            // return data_.at(pos);
            return data_[RasterIndexer::getPos(domain_.min(), domain_.dimensions(), indices)];
        }

        const Value &value(const Index &pos) const
        {
            return data_.at(pos);
        }

        Value &value(const Index &pos)
        {
            return data_.at(pos);
        }
    };

}
