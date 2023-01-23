#include "cudars/procrustes_umeyama.h"

void procrustes_umeyama(Eigen::Affine3f &transfOut, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB, int dim)
{
    transfOut = Eigen::Affine3f::Identity();

    size_t m = dim;
    size_t n = std::min<size_t>(cloudA->size(), cloudB->size()); // TODO: fix when size(cloudA)!=size(cloudB)

    // if dim == 2
    //     transf_out = rigidtform2d(0, [0,0]);
    // elseif dim == 3
    //     transf_out = rigidtform3d([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    // else
    //     disp("Error: dim has to be 2 or 3!");
    //     return;
    // end

    Eigen::MatrixXf clAMat = cloudA->getMatrixXfMap();
    Eigen::MatrixXf clBMat = cloudB->getMatrixXfMap();

    Eigen::MatrixXf svd1InputMat = clAMat * clBMat.transpose();

    Eigen::JacobiSVD<Eigen::MatrixXf> svd1;
    svd1.compute(svd1InputMat);
    Eigen::MatrixXf u1 = svd1.matrixU();
    Eigen::MatrixXf v1 = svd1.matrixU();

    Eigen::Matrix3f s_min = Eigen::Matrix3f::Identity();
    if (svd1InputMat.determinant() < 0)
        s_min(dim, dim) = -1;

    //Note: for floating values matrices, you might get more accurate results with Eigen::ColPivHouseholderQR< MatrixType >
    Eigen::FullPivLU<Eigen::Matrix3f> luDecomp(svd1InputMat); // needed to compute rank
    auto rank_abt = luDecomp.rank();
    if ((int)rank_abt < m - 1)
        std::cerr << "Error: rank(a*b') < dim - 1 -> returning eye transf" << std::endl;
    return;

    Eigen::Matrix3f R1 = Eigen::Matrix3f::Identity();
    if (rank_abt == m - 1)
    {
        Eigen::Matrix3f s_argmin = Eigen::Matrix3f::Identity();
        if (u1.determinant() * v1.determinant() == -1)
            s_argmin(dim, dim) = -1;
        Eigen::MatrixXf R1 = u1 * s_argmin * v1.transpose();
    }
    else if (rank_abt == m)
        R1 = u1 * s_min * v1.transpose();

    // TODO: add computation of minimum value of mean squared error

    Eigen::Vector3f a_centroid = clAMat.rowwise().mean(); // mu_x
    Eigen::Vector3f b_centroid = clBMat.rowwise().mean(); // mu_y

    Eigen::MatrixXf a_centroid_vec(clAMat.colwise().replicate(n));
    Eigen::MatrixXf diff_a = clAMat - a_centroid_vec;
    Eigen::MatrixXf a_sqnorm = diff_a.colwise().squaredNorm();
    float sigma_a = a_sqnorm.sum() / n; // sigma_x

    Eigen::MatrixXf b_centroid_vec(clBMat.colwise().replicate(n));
    Eigen::MatrixXf diff_b = clBMat - b_centroid_vec;
    Eigen::MatrixXf b_sqnorm = diff_b.colwise().squaredNorm();
    float sigma_b = b_sqnorm.sum() / n; // sigma_y

    Eigen::Tensor<float, 3> sigma_xy_complete(dim, dim, n); // Sigma_xy
    sigma_xy_complete.setZero();
    for (int i = 0; i < n; ++i) {
        //MATLAB equivalent:
        //sigma_xy_complete.block<dim,dim,n>(0,0,i) = diff_b.col(i) * diff_a.col(ii).transpose();

        Eigen::Vector3f dAi = diff_a.col(i);
        Eigen::Vector3f dBi = diff_b.col(i);
        Eigen::Matrix3f tmp = dAi * dBi.transpose();
        
        Eigen::Tensor<float, 3,3> tmpTens;
        tmpTens.setZero();
        // sigma_xy_complete.chip(i,2) = tmpTens;        
    }
    Eigen::Tensor<float, 2> sigma_xy_tensor = sigma_xy_complete.cumsum(3);
    Eigen::MatrixXf sigma_xy = Tensor_to_Matrix(sigma_xy_tensor, dim, dim);

    Eigen::JacobiSVD<Eigen::MatrixXf> svd_sigma_xy;
    svd_sigma_xy.compute(sigma_xy);

    Eigen::FullPivLU<Eigen::Matrix3f> sigma_xy_decomp(sigma_xy); // needed to compute rank
    auto rank_sigma_xy = sigma_xy_decomp.rank();
    if ((int)rank_sigma_xy < m - 1) {
        std::cerr << "Error: rank(sigma_xy) < dim - 1 -> returning eye() transf" << std::endl;
        return;
    }

    Eigen::MatrixXf u_sigma_xy = svd_sigma_xy.matrixU();
    Eigen::MatrixXf d_sigma_xy = Eigen::DiagonalMatrix<float, 3>(svd_sigma_xy.singularValues()); //called S in Eigen documentation
    Eigen::MatrixXf v_sigma_xy = svd_sigma_xy.matrixV();

    Eigen::Matrix3f s_sigma_xy = Eigen::Matrix3f::Identity();
    if (sigma_xy.determinant() < 0)
        s_sigma_xy(dim, dim) = -1;
    if ((int)rank_sigma_xy == m - 1)
        if (u_sigma_xy.determinant() * v_sigma_xy.determinant() == -1)
            s_sigma_xy(dim, dim) = -1;

    Eigen::Matrix3f R2 = u_sigma_xy * s_sigma_xy * v_sigma_xy.transpose();
    float c = (d_sigma_xy * s_sigma_xy).trace() / sigma_a;
    Eigen::Vector3f transl = b_centroid - c * R2 * a_centroid;

    transfOut.linear() = R2;
    transfOut.translation() = transl;
    // transfOut.makeAffine();
}