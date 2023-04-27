#include <cudars/BBTranslation.h>

namespace cudars
{

    double distancePointBox(const Vec2d &p,
                            const Vec2d &boxMin,
                            const Vec2d &boxMax)
    {
        double dist = 0.0;
        double len;
        for (int d = 0; d < 2; ++d)
        {
            // if (boxMin(d) <= p(d) && p(d) <= boxMax(d))
            if (idxGetter(boxMin,d) <= idxGetter(p,d) && idxGetter(p,d) <= idxGetter(boxMax,d))
            {
                len = 0.0;
            }
            // else if (p(d) < boxMin(d))
            else if (idxGetter(p,d) < idxGetter(boxMin,d))
            {
                len = idxGetter(boxMin,d) - idxGetter(p,d);
            }
            else
            {
                len = idxGetter(p,d) - idxGetter(boxMax,d);
            }
            dist += len * len;
        }
        return dist;
    }

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
                // dist = (ptsDst[id] - srcTransl).norm();
                dist = (ptsDst[id] - srcTransl).cwiseAbs().maxCoeff(); // Infinity norm
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

    std::ostream &operator<<(std::ostream &out, const Box &box)
    {
        out << "min [" << box.min_.transpose() << "] max [" << box.max_.transpose()
            << "] lower " << box.lower_ << " upper " << box.upper_;
        return out;
    }

    BBTranslation::BBTranslation() : res_(0.1), eps_(0.1), numMaxIter_(1000) {}

    BBTranslation::~BBTranslation() {}

    void BBTranslation::compute(Vec2d &translOpt)
    {
        Vec2d boxSplitMin, boxSplitMax;

        double scoreOpt, scoreTol;
        int iterNum;

        auto cmp = [](const Box &box1, const Box &box2)
        {
            return box1.lower_ > box2.lower_;
        };
        std::priority_queue<Box, std::vector<Box>, decltype(cmp)> prioqueue(cmp);

        scoreTol = 0.05; // TODO: allow setting the value of scoreTol
        Box boxCur(translMin_, translMax_, ptsSrc_, ptsDst_, eps_);
        prioqueue.push(boxCur);
        scoreOpt = prioqueue.top().upper_;
        // translOpt = 0.5 * (boxCur.min_ + boxCur.max_);
        vec2sum(translOpt, boxCur.min_, boxCur.max_);
        scalarMul(translOpt, 0.5);
        ARS_VARIABLE2(boxCur, scoreOpt);
        iterNum = 0;
        while (!prioqueue.empty() && iterNum < numMaxIter_)
        {
            boxCur = prioqueue.top();
            prioqueue.pop();

            std::cout << "\n---\niteration " << iterNum << " queue size "
                      << prioqueue.size() << std::endl;
            ARS_PRINT("boxCur " << boxCur << ", score optimum " << scoreOpt);
            // ARS_VARIABLE4(boxCur.upper_, boxCur.lower_, scoreTol * scoreOpt,
            //          boxCur.upper_ - boxCur.lower_ <= scoreTol * scoreOpt);
            if (scoreOpt - boxCur.lower_ <= scoreTol * scoreOpt)
            {
                ARS_PRINT("STOP");
                break;
            }

            // Splits the current box into 2^DIM parts
            // if ((boxCur.max_ - boxCur.min_).maxCoeff() > res_)
            if (maxCoeffWRV(vec2diffWRV(boxCur.max_, boxCur.min_)) > res_)
            {
                for (int j = 0; j < SPLIT_NUM; ++j)
                {
                    for (int d = 0; d < DIM; ++d)
                    {
                        // ARS_VARIABLE4(j, d, (1 << d), j & (1 << d));
                        if (j & (1 << d))
                        {
                            // boxSplitMin(d) = 0.5 * (boxCur.min_(d) + boxCur.max_(d));
                            idxSetter(boxSplitMin, d, 0.5 * (idxGetter(boxCur.min_,d) + idxGetter(boxCur.max_,d)));
                            // boxSplitMax(d) = boxCur.max_(d);
                            idxSetter(boxSplitMax, d, idxGetter(boxCur.max_, d));
                        }
                        else
                        {
                            // boxSplitMin(d) = boxCur.min_(d);
                            idxSetter(boxSplitMin, d, idxGetter(boxCur.min_, d));
                            // boxSplitMax(d) = 0.5 * (boxCur.min_(d) + boxCur.max_(d));
                            idxSetter(boxSplitMax, d, 0.5 * (idxGetter(boxCur.min_,d) + idxGetter(boxCur.max_,d)));
                        }
                        // ARS_VARIABLE2(boxSplitMin(d), boxSplitMax(d));
                    }
                    Box boxNew(boxSplitMin, boxSplitMax, ptsSrc_, ptsDst_, eps_);
                    ARS_VARIABLE(boxNew);

                    if (boxNew.upper_ < scoreOpt)
                    {
                        scoreOpt = boxNew.upper_;
                        // translOpt = 0.5 * (boxNew.min_ + boxNew.max_);
                        translOpt = scalarMulWRV(vec2sumWRV(boxNew.min_, boxNew.max_), 0.5);
                        // ARS_PRINT("UPDATE optimum " << scoreOpt << " in "
                        //                             << translOpt.transpose());
                    }

                    if (boxNew.lower_ < scoreOpt)
                    {
                        prioqueue.push(boxNew);
                    }
                }
            }
            iterNum++;
        }
        // ARS_PRINT("OPTIMUM " << scoreOpt << " in " << translOpt.transpose());
    }

    void BBTranslation::setTranslMinMax(const cudars::Vec2d &translMin,
                                        const cudars::Vec2d &translMax)
    {
        translMin_ = translMin;
        translMax_ = translMax;
        // ARS_VARIABLE2(translMin_.transpose(), translMax_.transpose())
    }

    void BBTranslation::setResolution(const double r)
    {
        res_ = r;
    }

    void BBTranslation::setPtsSrc(const cudars::VecVec2d &pts)
    {
        ptsSrc_ = pts;
    }

    void BBTranslation::setPtsDst(const cudars::VecVec2d &pts)
    {
        ptsDst_ = pts;
    }

    void BBTranslation::setPts(const cudars::VecVec2d &ptsS,
                               const cudars::VecVec2d &ptsD)
    {
        // ptsSrc_ = ptsS;
        ptsSrc_.clear();
        for (int i=0; i<ptsS.size(); ++i)
            ptsSrc_.push_back(ptsS[i]);
        // ptsDst_ = ptsD;
        ptsDst_.clear();
        for (int i=0; i<ptsD.size(); ++i)
            ptsDst_.push_back(ptsD[i]);
    }

    void BBTranslation::setEps(const double eps) {
        eps_ = eps;
    }

    void BBTranslation::setNumMaxIterations(const int nmi) {
        numMaxIter_ = nmi;
    }

} // namespace cudars