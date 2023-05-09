#include <cudars/BBTranslation.cuh>

__global__ void computeBBTransl_kernel(cudars::Vec2d *ptsSrc_, cudars::Vec2d *ptsDst_,
                                       cudars::Vec2d &translOpt, cudars::Vec2d &translMin_, cudars::Vec2d &translMax_,
                                       double eps_, int numMaxIter_, double res_, int ptsSrcSize, int ptsDstSize)
{
    printf("Running computeBBTransl_kernel\n"); 

    cudars::Vec2d boxSplitMin, boxSplitMax;

    double scoreOpt, scoreTol;
    int iterNum;

    // auto cmp = [](const cudars::CuBox &box1, const cudars::CuBox &box2)
    // {
    //     return box1.lower_ > box2.lower_;
    // };
    // std::priority_queue<cudars::CuBox, std::vector<cudars::CuBox>, decltype(cmp)> prioqueue(cmp);
    NodeBox *prioqueue;

    scoreTol = 0.05; // TODO: allow setting the value of scoreTol

    // cudars::CuBox boxCur(translMin_, translMax_, ptsSrc_, ptsDst_, eps_);
    cudars::CuBox boxCur;
    printf("Initializing first box...\n");
    initCuBox(boxCur, translMin_, translMax_, ptsSrc_, ptsDst_, eps_, ptsSrcSize, ptsDstSize);
    printf("First box has been initialized\n");
    // prioqueue.push(boxCur);
    pushBox(&prioqueue, boxCur);
    // scoreOpt = prioqueue.top().upper_;
    scoreOpt = peekBox(&prioqueue).upper_;
    // translOpt = 0.5 * (boxCur.min_ + boxCur.max_);
    translOpt.x = 0.5 * (boxCur.min_.x + boxCur.max_.x);
    translOpt.y = 0.5 * (boxCur.min_.y + boxCur.max_.y);
    // ARS_VARIABLE2(boxCur, scoreOpt);
    iterNum = 0;
    // while (!prioqueue.empty() && iterNum < numMaxIter_)
    while (!isEmptyBox(&prioqueue) && iterNum < numMaxIter_)
    {
        printf("iterNum %d\n", iterNum);

        // boxCur = prioqueue.top();
        boxCur = peekBox(&prioqueue);
        // prioqueue.pop();
        popBox(&prioqueue);

        printf("\n---\niteration %d queue size %d\n", iterNum, getSizeBox(&prioqueue));
        // ARS_PRINT("boxCur " << boxCur << ", score optimum " << scoreOpt);
        printf("boxCur min [%f %f] max [%d %d] lower %d upper %d score optimum %d\n",
                        boxCur.min_.x, boxCur.min_.y, boxCur.max_.x, boxCur.max_.y, boxCur.lower_, boxCur.upper_, scoreOpt);
        // ARS_VARIABLE4(boxCur.upper_, boxCur.lower_, scoreTol * scoreOpt,
        //          boxCur.upper_ - boxCur.lower_ <= scoreTol * scoreOpt);
        if (scoreOpt - boxCur.lower_ <= scoreTol * scoreOpt)
        {
            // ARS_PRINT("STOP");
            break;
        }

        // Splits the current box into 2^DIM parts
        // if ((boxCur.max_ - boxCur.min_).maxCoeff() > res_)
        bool tmp = max(boxCur.max_.x - boxCur.min_.x, boxCur.max_.y - boxCur.min_.y) > res_;
        if (tmp)
        {
            for (int j = 0; j < SPLIT_NUM; ++j)
            {
                for (int d = 0; d < DIM; ++d)
                {
                    // ARS_VARIABLE4(j, d, (1 << d), j & (1 << d));
                    if (d == 0)
                    {
                        if (j & (1 << d))
                        {
                            boxSplitMin.x = 0.5 * (boxCur.min_.x + boxCur.max_.x);
                            boxSplitMax.x = boxCur.max_.x;
                        }
                        else
                        {
                            boxSplitMin.x = boxCur.min_.x;
                            boxSplitMax.x = 0.5 * (boxCur.min_.x + boxCur.max_.x);
                        }
                    }
                    if (d == 0)
                    {
                        if (j & (1 << d))
                        {
                            boxSplitMin.y = 0.5 * (boxCur.min_.y + boxCur.max_.y);
                            boxSplitMax.y = boxCur.max_.y;
                        }
                        else
                        {
                            boxSplitMin.y = boxCur.min_.y;
                            boxSplitMax.y = 0.5 * (boxCur.min_.y + boxCur.max_.y);
                        }
                    }
                    // ARS_VARIABLE2(boxSplitMin(d), boxSplitMax(d));
                }
                // cudars::CuBox boxNew(boxSplitMin, boxSplitMax, ptsSrc_, ptsDst_, eps_);
                cudars::CuBox boxNew;
                initCuBox(boxNew, boxSplitMin, boxSplitMax, ptsSrc_, ptsDst_, eps_, ptsSrcSize, ptsDstSize);
                // ARS_VARIABLE(boxNew);

                if (boxNew.upper_ < scoreOpt)
                {
                    scoreOpt = boxNew.upper_;
                    // translOpt = 0.5 * (boxNew.min_ + boxNew.max_);
                    translOpt.x = 0.5 * (boxNew.min_.x + boxNew.max_.x);
                    translOpt.y = 0.5 * (boxNew.min_.y + boxNew.max_.y);
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