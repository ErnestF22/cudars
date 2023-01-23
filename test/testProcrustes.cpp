#include <iostream>

#include "cudars/procrustes_umeyama.h"


// % 3D setup
// a = [1 1 1; 3 4 5; 10 8 6; 2 4 7; 7 1 2; -14 -5 16]';
// b = zeros(size(a));
// rotz_true = 30; % deg
// transl_true = [-1; -1; -1];
// for ii = 1:size(a, 2)
//     b(:, ii) = rotz(rotz_true) * a(:, ii) + transl_true;
// end
// disp(b);
// d = 3;

// %umeyama paper setup
// % a = [0 0; 1 0 ; 0 2]';
// % b = [0 0; -1 0; 0 2]';
// % d = 2;

// rigid_out = procrustes_umeyama(a,b,d);
// disp("transf_out");
// disp(rigid_out.A)


int main (int argc, char **argv) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr clA(new pcl::PointCloud<pcl::PointXYZ>), clB(new pcl::PointCloud<pcl::PointXYZ>);
    int dim = 3;
    Eigen::Affine3f transf;

    pcl::PointXYZ aS(pcl::PointXYZ(0,0,0));
    clA->push_back(aS);
    pcl::PointXYZ bS(pcl::PointXYZ(1,0,0));
    clA->push_back(bS);
    pcl::PointXYZ cS(pcl::PointXYZ(0,2,0));
    clA->push_back(cS);

    pcl::PointXYZ aD(pcl::PointXYZ(0,0,0));
    clA->push_back(aD);
    pcl::PointXYZ bD(pcl::PointXYZ(-1,0,0));
    clA->push_back(bD);
    pcl::PointXYZ cD(pcl::PointXYZ(0,2,0));
    clA->push_back(cD);

    procrustes_umeyama(transf, clA, clB, dim);

    std::cout << std::endl << transf.matrix() << std::endl;

    return 0;
}