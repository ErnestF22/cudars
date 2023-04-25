#ifndef CUDARS_BBTRANSLATION_H_
#define CUDARS_BBTRANSLATION_H_

#include <cudars/definitions.h>

#include <cudars/utils.h>

#include <queue>

// #include <Eigen/Dense>
// #include <Eigen/Core>

namespace cudars
{

    double distancePointBox(const Vec2d &p,
                            const Vec2d &boxMin,
                            const Vec2d &boxMax);

    struct Box
    {
        Vec2d min_;
        Vec2d max_;
        double lower_;
        double upper_;
        double eps_;

        Box(const Vec2d &min, const Vec2d &max, const double eps);

        Box(const Vec2d &min,
            const Vec2d &max,
            const VecVec2d &ptsSrc,
            const VecVec2d &ptsDst,
            const double eps);

        virtual ~Box();

        void computeBoundsNaive(const VecVec2d &ptsSrc,
                                const VecVec2d &ptsDst);

        void computeBoundsInlier(const VecVec2d &ptsSrc,
                                 const VecVec2d &ptsDst);
    };

    std::ostream &operator<<(std::ostream &out, const Box &box);

    class BBTranslation
    {
    public:
        static constexpr int DIM = 2;
        static constexpr int SPLIT_NUM = (1 << DIM);

        /**
         * @brief Default constructor for a new BBTranslation object
         */
        BBTranslation();

        /**
         * @brief Default destructor for BBTranslation objects
         */
        virtual ~BBTranslation();

        /**
         * @brief Main method
         */
        void compute(Vec2d &translOpt);

        /**
         * @brief Set the interval search of translation
         */
        void setTranslMinMax(const Vec2d &translMin,
                             const Vec2d &translMax);

        /**
         * @brief Set minimum box size
         */
        void setResolution(const double r);

        /**
         * @brief Set points src
         */
        void setPtsSrc(const VecVec2d &pts);

        /**
         * @brief Set points dst
         */
        void setPtsDst(const VecVec2d &pts);

        /**
         * @brief Set pts src and dst
         */
        void setPts(const VecVec2d &ptsS, const VecVec2d &ptsD);

        /**
         * @brief Set epsilon param for lower bound computation
         */
        void setEps(const double eps);

        /**
         * @brief Set max number of iteration before B&B alg stops 
         */
        void setNumMaxIterations(const int nmi);

    private:
        // ars::Vec2d translMin_;
        Vec2d translMin_;
        // ars::Vec2d translMax_;
        Vec2d translMax_;

        // ars::VecVec2d ptsSrc_;
        VecVec2d ptsSrc_;
        // ars::VecVec2d ptsDst_;
        VecVec2d ptsDst_;

        double res_;
        double eps_;

        int numMaxIter_;
    };
} // namespace cudars

#endif /*CUDARS_BBTRANSLATION_H_*/
