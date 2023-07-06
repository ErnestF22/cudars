#include <iostream>

#include <cudars/BBTranslation.h>



int main(int argc, char **argv)
{
    cudars::VecVec2d ptsA, ptsB;
    cudars::Vec2d minA, maxA, minB, maxB;
    // int dim = 2;
    cudars::BBTranslation translEstim;

    cudars::Vec2d aA(make_double2(1, 1));
    cudars::Vec2d bA(make_double2(3, 4));
    cudars::Vec2d cA(make_double2(10, 8));
    cudars::Vec2d dA(make_double2(4, 7));
    cudars::Vec2d eA(make_double2(7, 2));
    cudars::Vec2d fA(make_double2(-14, -5));
    ptsA.push_back(aA);
    ptsA.push_back(bA);
    ptsA.push_back(cA);
    ptsA.push_back(dA);
    ptsA.push_back(eA);
    ptsA.push_back(fA);

    double rotTrue = 0.0;
    cudars::Vec2d translTrue = make_double2(3, -3);
    cudars::Affine2d transfTrue(rotTrue, translTrue.x, translTrue.y);
    std::cout << "Applying transformation transfTrue:" << std::endl
              << transfTrue << std::endl;
    for (int i = 0; i < 6; ++i) // 6 = number of points
    {
        cudars::Vec2d ptsAiTransf;
        cudars::vec2sum(ptsAiTransf, ptsA[i], transfTrue.translation());
        ptsB.push_back(cudars::Vec2d(ptsAiTransf));
    }

    translEstim.setResolution(0.1);
    translEstim.setEps(0.1);
    translEstim.setNumMaxIterations(1000);

    cudars::findBoundingBox(ptsA, minA, maxA);
    cudars::findBoundingBox(ptsB, minB, maxB);
    cudars::Vec2d bias = make_double2(20.0, 20.0);
    std::cout << "ptsA: size " << ptsA.size() << ", min [";
    cudars::printVec2d(minA);
    std::cout << "]  max [";
    cudars::printVec2d(maxA);
    std::cout << "]" << std::endl;
    std::cout << "ptsB: size " << ptsB.size() << ", min [";
    cudars::printVec2d(minB);
    std::cout << "]  max [";
    cudars::printVec2d(maxB);
    std::cout << "]" << std::endl;
    std::cout << "translation interval: min [";
    cudars::printVec2d(cudars::vec2diffWRV(minB, maxA));
    std::cout << "]  max [";
    cudars::printVec2d(cudars::vec2sumWRV(cudars::vec2diffWRV(maxB, minA), bias));
    std::cout << "]" << std::endl;

    translEstim.setTranslMinMax(cudars::vec2diffWRV(minB, maxA), cudars::vec2sumWRV(cudars::vec2diffWRV(maxB, minA), bias));
    translEstim.setPts(ptsA, ptsB);
    cudars::Vec2d translOut;
    translEstim.compute(translOut);

    std::cout << std::endl
              << "translOut";
    cudars::printVec2d(translOut);
    std::cout << std::endl
              << "translTrue";
    cudars::printVec2d(transfTrue.translation());

    return 0;
}

