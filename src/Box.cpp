#include <cudars/Box.h>

namespace cudars
{

    Box::Box(const Vec2d &min, const Vec2d &max, const double eps)
        : min_(min), max_(max), lower_(0.0), upper_(0.0), eps_(eps) {}

    Box::Box(const Vec2d &min,
             const Vec2d &max,
             const VecVec2d &ptsSrc,
             const VecVec2d &ptsDst,
             const double eps)
    {
        double dist, distMin, distUpper, distUpperMin;
        Vec2d boxMin, boxMax, boxMid;
        min_ = min;
        max_ = max;
        eps_ = eps;
        // computeBoundsNaive(ptsSrc, ptsDst);
        computeBoundsInlier(ptsSrc, ptsDst);
        
        printf("Box constructor after computeBoundsInlier()\n");
        std::cout << "New Box: " << "min_ " << min_.x << " " << min_.y 
                << " max_ " << max_.x << " " << max_.y
                << " lower_ " << lower_ << " upper_ " << upper_ << std::endl;
    }

    Box::~Box() {}

    void Box::computeBoundsNaive(const VecVec2d &ptsSrc,
                                 const VecVec2d &ptsDst)
    {
        double distLower, distLowerMin, distUpper, distUpperMin;
        Vec2d boxMin, boxMax, boxMid;
        lower_ = 0.0;
        upper_ = 0.0;
        for (int is = 0; is < ptsSrc.size(); ++is)
        {
            distLowerMin = 1e+6;
            distUpperMin = 1e+6;
            // boxMin = ptsSrc[is] + min_;
            // boxMax = ptsSrc[is] + max_;
            // boxMid = ptsSrc[is] + 0.5 * (max_ + min_);
            vec2sum(boxMin, ptsSrc[is], min_);
            vec2sum(boxMax, ptsSrc[is], max_);
            vec2sum(boxMid, ptsSrc[is], scalarMulWRV(vec2sumWRV(max_, min_), 0.5));
            for (int id = 0; id < ptsDst.size(); ++id)
            {
                distLower = distancePointBox(ptsDst[id], boxMin, boxMax);
                if (distLower < distLowerMin)
                {
                    distLowerMin = distLower;
                }
                // distUpper = (ptsDst[id] - (ptsSrc[is] + boxMid)).squaredNorm();
                distUpper = vec2squarednorm(vec2diffWRV(ptsDst[id], vec2sumWRV(ptsSrc[is], boxMid)));
                // ARS_VARIABLE5(boxMid.transpose(), ptsSrc[is].transpose(),
                //          ptsDst[id].transpose(), distUpper, distUpperMin);
                if (distUpper < distUpperMin)
                {
                    distUpperMin = distUpper;
                }
            }
            lower_ += distLowerMin;
            upper_ += distUpperMin;
            // ARS_VARIABLE4(distLowerMin, distUpperMin, lower_, upper_);
        }
    }

    void Box::computeBoundsInlier(const VecVec2d &ptsSrc,
                                  const VecVec2d &ptsDst)
    {
        // Vec2d mid = 0.5 * (min_ + max_);
        Vec2d mid = vec2sumWRV(min_, max_);
        scalarMul(mid, 0.5);
        Vec2d srcTransl;
        double dist, len;
        bool inlierFoundUpper, inlierFoundLower;

        // len = 0.5 * (max_ - min_).maxCoeff(); // Half of Infinity norm
        len = 0.5 * maxCoeffWRV(vec2diffWRV(max_, min_)); // Half of Infinity norm
        lower_ = (double)ptsSrc.size();
        upper_ = (double)ptsSrc.size();
        // ARS_VARIABLE4(lower_, upper_, len, mid);
        for (int is = 0; is < ptsSrc.size(); ++is)
        {
            // srcTransl = ptsSrc[is] + mid;
            vec2sum(srcTransl, ptsSrc[is], mid);
            inlierFoundLower = false;
            inlierFoundUpper = false;
            // ARS_VAR1(srcTransl.transpose());
            for (int id = 0; id < ptsDst.size() && !(inlierFoundLower && inlierFoundUpper); ++id)
            {
                // ARS_VARIABLE3(id, lower_, upper_);
                // dist = (ptsDst[id] - srcTransl).norm();
                // dist = (ptsDst[id] - srcTransl).cwiseAbs().maxCoeff(); // Infinity norm
                dist = maxCoeffWRV(cwiseAbsWRV(vec2diffWRV(ptsDst[id], srcTransl))); // Infinity norm
                // ARS_VARIABLE4(ptsDst[id].transpose(), dist, dist < eps_, dist < eps_ + len);
                if (dist < eps_)
                {
                    inlierFoundUpper = true;
                }
                if (dist < eps_ + len)
                {
                    inlierFoundLower = true;
                }
            }
            if (inlierFoundLower)
                lower_ -= 1.0;
            if (inlierFoundUpper)
                upper_ -= 1.0;
        }
    }

    std::ostream &operator<<(std::ostream &out, const cudars::Box &box)
    {
        // out << "min [" << box.min_.transpose() << "] max [" << box.max_.transpose()
        //     << "] lower " << box.lower_ << " upper " << box.upper_;
        out << "min [" << box.min_.x << " " << box.min_.y << "] max [" << box.max_.x << " " << box.max_.y
            << "] lower " << box.lower_ << " upper " << box.upper_;
        return out;
    }
}
