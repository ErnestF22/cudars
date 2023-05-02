#ifndef CUDARS_BOX_CUH_
#define CUDARS_BOX_CUH_

#include <cudars/definitions.h>
#include <cudars/utils.h>


namespace cudars
{
    // void initCuBox(CuBox &box, const Vec2d &min, const Vec2d &max, const double eps);

    // void initCuBox(CuBox &box, const Vec2d &min,
    //                const Vec2d &max,
    //                const VecVec2d &ptsSrc,
    //                const VecVec2d &ptsDst,
    //                const double eps);

    // // virtual ~Box();

    // void computeBoundsNaive(const VecVec2d &ptsSrc,
    //                         const VecVec2d &ptsDst);

    

    // void computeBoundsInlier(Vec2d &min_, Vec2d &max_, double &lower_, double &upper_, double eps_,
    //                          const VecVec2d &ptsSrc, const VecVec2d &ptsDst);

    // std::ostream &operator<<(std::ostream &out, const cudars::Box &box);

} // end of namespace cudars

#endif /*CUDARS_BOX_CUH_*/