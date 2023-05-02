#include <cudars/BBTranslation.cuh>

__global__ void computeBBTransl_kernel(cudars::VecVec2d &ptsSrc_, cudars::VecVec2d &ptsDst_,
                                       cudars::Vec2d &translOpt, cudars::Vec2d &translMin_, cudars::Vec2d &translMax_,
                                       double eps_, int numMaxIter_, double res_)
{
    cudars::Vec2d boxSplitMin, boxSplitMax;

    double scoreOpt, scoreTol;
    int iterNum;

    // auto cmp = [](const cudars::CuBox &box1, const cudars::CuBox &box2)
    // {
    //     return box1.lower_ > box2.lower_;
    // };
    // std::priority_queue<cudars::CuBox, std::vector<cudars::CuBox>, decltype(cmp)> prioqueue(cmp);
    NodeBox* prioqueue;

    scoreTol = 0.05; // TODO: allow setting the value of scoreTol

    // cudars::CuBox boxCur(translMin_, translMax_, ptsSrc_, ptsDst_, eps_);
    cudars::CuBox boxCur;
    initCuBox(boxCur, translMin_, translMax_, ptsSrc_, ptsDst_, eps_);
    // prioqueue.push(boxCur);
    pushBox(&prioqueue, boxCur);
    // scoreOpt = prioqueue.top().upper_;
    scoreOpt = peekBox(&prioqueue).upper_;
    // translOpt = 0.5 * (boxCur.min_ + boxCur.max_);
    cudars::vec2sum(translOpt, boxCur.min_, boxCur.max_);
    cudars::scalarMul(translOpt, 0.5);
    // ARS_VARIABLE2(boxCur, scoreOpt);
    iterNum = 0;
    // while (!prioqueue.empty() && iterNum < numMaxIter_)
    while (!isEmptyBox(&prioqueue) && iterNum < numMaxIter_)
    {
        // boxCur = prioqueue.top();
        boxCur = peekBox(&prioqueue);
        // prioqueue.pop();
        popBox(&prioqueue);

        // std::cout << "\n---\niteration " << iterNum << " queue size "
        //             << prioqueue.size() << std::endl;
        // ARS_PRINT("boxCur " << boxCur << ", score optimum " << scoreOpt);
        // ARS_VARIABLE4(boxCur.upper_, boxCur.lower_, scoreTol * scoreOpt,
        //          boxCur.upper_ - boxCur.lower_ <= scoreTol * scoreOpt);
        if (scoreOpt - boxCur.lower_ <= scoreTol * scoreOpt)
        {
            // ARS_PRINT("STOP");
            break;
        }

        // Splits the current box into 2^DIM parts
        // if ((boxCur.max_ - boxCur.min_).maxCoeff() > res_)
        if (cudars::maxCoeffWRV(cudars::vec2diffWRV(boxCur.max_, boxCur.min_)) > res_)
        {
            for (int j = 0; j < SPLIT_NUM; ++j)
            {
                for (int d = 0; d < DIM; ++d)
                {
                    // ARS_VARIABLE4(j, d, (1 << d), j & (1 << d));
                    if (j & (1 << d))
                    {
                        // boxSplitMin(d) = 0.5 * (boxCur.min_(d) + boxCur.max_(d));
                        cudars::idxSetter(boxSplitMin, d, 0.5 * (cudars::idxGetter(boxCur.min_, d) + cudars::idxGetter(boxCur.max_, d)));
                        // boxSplitMax(d) = boxCur.max_(d);
                        cudars::idxSetter(boxSplitMax, d, cudars::idxGetter(boxCur.max_, d));
                    }
                    else
                    {
                        // boxSplitMin(d) = boxCur.min_(d);
                        cudars::idxSetter(boxSplitMin, d, cudars::idxGetter(boxCur.min_, d));
                        // boxSplitMax(d) = 0.5 * (boxCur.min_(d) + boxCur.max_(d));
                        cudars::idxSetter(boxSplitMax, d, 0.5 * (cudars::idxGetter(boxCur.min_, d) + cudars::idxGetter(boxCur.max_, d)));
                    }
                    // ARS_VARIABLE2(boxSplitMin(d), boxSplitMax(d));
                }
                // cudars::CuBox boxNew(boxSplitMin, boxSplitMax, ptsSrc_, ptsDst_, eps_);
                cudars::CuBox boxNew;
                initCuBox(boxNew, boxSplitMin, boxSplitMax, ptsSrc_, ptsDst_, eps_);
                // ARS_VARIABLE(boxNew);

                if (boxNew.upper_ < scoreOpt)
                {
                    scoreOpt = boxNew.upper_;
                    // translOpt = 0.5 * (boxNew.min_ + boxNew.max_);
                    translOpt = cudars::scalarMulWRV(cudars::vec2sumWRV(boxNew.min_, boxNew.max_), 0.5);
                    // ARS_PRINT("UPDATE optimum " << scoreOpt << " in "
                    //                             << translOpt.transpose());
                }

                if (boxNew.lower_ < scoreOpt)
                {
                    // prioqueue.push(boxNew);
                    pushBox(&prioqueue, boxNew);
                }
            }
        }
        iterNum++;
    }
    // ARS_PRINT("OPTIMUM " << scoreOpt << " in " << translOpt.transpose());
}