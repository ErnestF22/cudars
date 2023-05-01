#include <cudars/BBTranslation.h>

namespace cudars
{
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
                            idxSetter(boxSplitMin, d, 0.5 * (idxGetter(boxCur.min_, d) + idxGetter(boxCur.max_, d)));
                            // boxSplitMax(d) = boxCur.max_(d);
                            idxSetter(boxSplitMax, d, idxGetter(boxCur.max_, d));
                        }
                        else
                        {
                            // boxSplitMin(d) = boxCur.min_(d);
                            idxSetter(boxSplitMin, d, idxGetter(boxCur.min_, d));
                            // boxSplitMax(d) = 0.5 * (boxCur.min_(d) + boxCur.max_(d));
                            idxSetter(boxSplitMax, d, 0.5 * (idxGetter(boxCur.min_, d) + idxGetter(boxCur.max_, d)));
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
        for (int i = 0; i < ptsS.size(); ++i)
            ptsSrc_.push_back(ptsS[i]);
        // ptsDst_ = ptsD;
        ptsDst_.clear();
        for (int i = 0; i < ptsD.size(); ++i)
            ptsDst_.push_back(ptsD[i]);
    }

    void BBTranslation::setEps(const double eps)
    {
        eps_ = eps;
    }

    void BBTranslation::setNumMaxIterations(const int nmi)
    {
        numMaxIter_ = nmi;
    }

} // namespace cudars