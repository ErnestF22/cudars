#include "cudars/procrustes_umeyama.h"

void procrustes_umeyama3f(Eigen::Affine3f &transfOut, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB)
{
    transfOut = Eigen::Affine3f::Identity();

    assert(cloudA->size() == cloudB->size());

    // Input clouds/matrices are supposed to have size m x n
    int m = THREE_;
    int n = std::min<int>(cloudA->size(), cloudB->size()); // TODO: fix when size(cloudA)!=size(cloudB)

    // // cloudA->points.erase(cloudA->points.end() - 1);
    // // cloudB->points.erase(cloudB->points.end() - 1);
    // Eigen::MatrixXf clAMat = cloudA->getMatrixXfMap();
    // Eigen::MatrixXf clBMat = cloudB->getMatrixXfMap();
    // clAMat = clAMat.block<3,3>(0,0);
    // clBMat = clBMat.block<3,3>(0,0);
    Eigen::MatrixXf clAMat(m, n);
    Eigen::MatrixXf clBMat(m, n);

    for (int i = 0; i < n; ++i)
    {
        Eigen::Vector3f ptA;
        ptA << cloudA->at(i).x, cloudA->at(i).y, cloudA->at(i).z;
        Eigen::Vector3f ptB;
        ptB << cloudB->at(i).x, cloudB->at(i).y, cloudB->at(i).z;
        clAMat.col(i) = ptA;
        clBMat.col(i) = ptB;
    }

    std::cout << "m " << m << " n " << n << std::endl;
    std::cout << "Mat a rows " << clAMat.rows() << " cols " << clAMat.cols() << std::endl;
    std::cout << "clAMat" << std::endl
              << clAMat << std::endl;
    std::cout << "Mat b rows " << clBMat.rows() << " cols " << clBMat.cols() << std::endl;
    std::cout << "clBMat" << std::endl
              << clBMat << std::endl;

    Eigen::MatrixXf svd1InputMat = clAMat * clBMat.transpose();

    Eigen::JacobiSVD<Eigen::MatrixXf> svd1(svd1InputMat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXf u1 = svd1.matrixU();
    Eigen::MatrixXf v1 = svd1.matrixV();

    Eigen::Matrix3f s_min = Eigen::Matrix3f::Identity();
    if (svd1InputMat.determinant() < 0)
        s_min(2, 2) = -1;

    // Note: for floating values matrices, you might get more accurate results with Eigen::ColPivHouseholderQR< MatrixType >
    Eigen::ColPivHouseholderQR<Eigen::MatrixXf> luDecomp(svd1InputMat); // needed to compute rank
    auto rank_abt = luDecomp.rank();
    if ((int)rank_abt < m - 1)
    {
        std::cerr << "Error: rank(a*b') < dim - 1 -> returning eye transf" << std::endl;
        return;
    }

    Eigen::Matrix3f R1 = Eigen::Matrix3f::Identity();
    if (rank_abt == m - 1)
    {
        Eigen::Matrix3f s_argmin = Eigen::Matrix3f::Identity();
        if (u1.determinant() * v1.determinant() == -1)
            s_argmin(2, 2) = -1;
        R1 = u1 * s_argmin * v1.transpose();
    }
    else if (rank_abt == m)
        R1 = u1 * s_min * v1.transpose();

    // TODO: add computation of minimum value of mean squared error

    Eigen::Vector3f a_centroid = clAMat.rowwise().mean(); // mu_x
    Eigen::Vector3f b_centroid = clBMat.rowwise().mean(); // mu_y

    Eigen::MatrixXf a_centroid_repmat(a_centroid.rowwise().replicate(n));
    Eigen::MatrixXf diff_a = clAMat - a_centroid_repmat;
    Eigen::MatrixXf a_sqnorm = diff_a.colwise().squaredNorm();
    float sigma_a = a_sqnorm.sum() / n; // sigma_x

    Eigen::MatrixXf b_centroid_repmat(b_centroid.rowwise().replicate(n));
    Eigen::MatrixXf diff_b = clBMat - b_centroid_repmat;
    Eigen::MatrixXf b_sqnorm = diff_b.colwise().squaredNorm();
    float sigma_b = b_sqnorm.sum() / n; // sigma_y

    Eigen::Tensor<float, THREE_> sigma_xy_complete(m, m, n); // Sigma_xy
    sigma_xy_complete.setZero();
    for (int i = 0; i < n; ++i)
    {
        // MATLAB equivalent:
        // sigma_xy_complete.block<dim,dim,n>(0,0,i) = diff_b.col(i) * diff_a.col(ii).transpose();

        Eigen::Vector3f dAi = diff_a.col(i);
        Eigen::Vector3f dBi = diff_b.col(i);
        Eigen::MatrixXf tmp = dBi * dAi.transpose(); // this is actually a Matrix2f, but conversion function takes MatrixXf as input

        Eigen::Tensor<float, TWO_> tmpTens = Matrix_to_Tensor(tmp, THREE_, THREE_);
        sigma_xy_complete.chip(i, 2) = tmpTens;

        std::cout << "i " << i << std::endl;
        std::cout << "tmp " << std::endl
                  << tmp << std::endl;
        std::cout << "sigma_xy_complete.chip(i, 2) " << std::endl
                  << sigma_xy_complete.chip(i, 2) << std::endl;
        std::cout << "tmpTens " << std::endl
                  << tmpTens << std::endl;
        std::cout << std::endl;
    }
    Eigen::Tensor<float, 2> sigma_xy_tensor(m, m);
    // built-in version
    // std::cout << std::endl << "sigma_xy_tensor" << sigma_xy_tensor << std::endl;
    // std::array<int, 3> two_dims{{3,3}};
    // sigma_xy_tensor = sigma_xy_complete.sum(two_dims);
    // hand-made version
    for (int i = 0; i < n; ++i)
    {
        sigma_xy_tensor += sigma_xy_complete.chip(i, 2);
    }
    Eigen::MatrixXf sigma_xy = Tensor_to_Matrix(sigma_xy_tensor, m, m);

    Eigen::ColPivHouseholderQR<Eigen::Matrix3f> sigma_xy_decomp(sigma_xy); // needed to compute rank
    auto rank_sigma_xy = sigma_xy_decomp.rank();
    if ((int)rank_sigma_xy < m - 1)
    {
        std::cerr << "Error: rank(sigma_xy) < dim - 1 -> returning eye() transf" << std::endl;
        return;
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd_sigma_xy(sigma_xy, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::MatrixXf u_sigma_xy = svd_sigma_xy.matrixU();
    //svd.singularValues().asDiagonal()
    Eigen::MatrixXf d_sigma_xy = Eigen::DiagonalMatrix<float, THREE_>(svd_sigma_xy.singularValues()); // called S in Eigen documentation
    Eigen::MatrixXf v_sigma_xy = svd_sigma_xy.matrixV();

    Eigen::Matrix3f s_sigma_xy = Eigen::Matrix3f::Identity();
    if (sigma_xy.determinant() < 0)
        s_sigma_xy(2, 2) = -1;
    if ((int)rank_sigma_xy == m - 1)
        if (u_sigma_xy.determinant() * v_sigma_xy.determinant() == -1)
            s_sigma_xy(2, 2) = -1;

    Eigen::Matrix3f R2 = u_sigma_xy * s_sigma_xy * v_sigma_xy.transpose();
    float c = (d_sigma_xy * s_sigma_xy).trace() / sigma_a;
    Eigen::Vector3f transl = b_centroid - c * R2 * a_centroid;

    transfOut.linear() = R2;
    transfOut.translation() = transl;
    // transfOut.makeAffine();
}

void procrustes_umeyama2f(Eigen::Affine2d &transfOut, const pcl::PointCloud<pcl::PointXY>::Ptr cloudA, const pcl::PointCloud<pcl::PointXY>::Ptr cloudB)
{
    transfOut = Eigen::Affine2d::Identity();

    assert(cloudA->size() == cloudB->size());

    // Input clouds/matrices are supposed to have size m x n
    int m = TWO_;
    int n = std::min<int>(cloudA->size(), cloudB->size()); // TODO: fix when size(cloudA)!=size(cloudB)

    Eigen::MatrixXd clAMat(m, n);
    Eigen::MatrixXd clBMat(m, n);

    for (int i = 0; i < n; ++i)
    {
        Eigen::Vector2d ptA;
        ptA << cloudA->at(i).x, cloudA->at(i).y;
        Eigen::Vector2d ptB;
        ptB << cloudB->at(i).x, cloudB->at(i).y;
        clAMat.col(i) = ptA;
        clBMat.col(i) = ptB;
    }

    std::cout << "m " << m << " n " << n << std::endl;
    std::cout << "Mat a rows " << clAMat.rows() << " cols " << clAMat.cols() << std::endl;
    std::cout << "clAMat" << std::endl
              << clAMat << std::endl;
    std::cout << "Mat b rows " << clBMat.rows() << " cols " << clBMat.cols() << std::endl;
    std::cout << "clBMat" << std::endl
              << clBMat << std::endl;

    Eigen::MatrixXd svd1InputMat = clAMat * clBMat.transpose();

    Eigen::JacobiSVD<Eigen::MatrixXd> svd1(svd1InputMat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd u1 = svd1.matrixU();
    Eigen::MatrixXd v1 = svd1.matrixV();

    Eigen::Matrix2d s_min = Eigen::Matrix2d::Identity();
    if (svd1InputMat.determinant() < 0)
        s_min(1, 1) = -1;

    // Note: for floating values matrices, you might get more accurate results with Eigen::ColPivHouseholderQR< MatrixType >
    Eigen::FullPivLU<Eigen::MatrixXd> luDecomp(svd1InputMat); // needed to compute rank
    auto rank_abt = luDecomp.rank();
    if ((int)rank_abt < m - 1)
    {
        std::cerr << "Error: rank(a*b') < dim - 1 -> returning eye transf" << std::endl;
        return;
    }

    Eigen::Matrix2d R1 = Eigen::Matrix2d::Identity();
    if (rank_abt == m - 1)
    {
        Eigen::Matrix2d s_argmin = Eigen::Matrix2d::Identity();
        if (u1.determinant() * v1.determinant() == -1)
            s_argmin(1, 1) = -1;
        R1 = u1 * s_argmin * v1.transpose();
    }
    else if (rank_abt == m)
        R1 = u1 * s_min * v1.transpose();

    // TODO: add computation of minimum value of mean squared error

    std::cout << "m " << m << " n " << n << std::endl;
    std::cout << "Mat a rows " << clAMat.rows() << " cols " << clAMat.cols() << std::endl;

    Eigen::Vector2d a_centroid = clAMat.rowwise().mean(); // mu_x
    Eigen::Vector2d b_centroid = clBMat.rowwise().mean(); // mu_y

    Eigen::MatrixXd a_centroid_repmat(a_centroid.rowwise().replicate(n));
    std::cout << "a_centroid_repmat rows " << a_centroid_repmat.rows() << " cols " << a_centroid_repmat.cols() << std::endl;
    Eigen::MatrixXd diff_a = clAMat - a_centroid_repmat;
    Eigen::MatrixXd a_sqnorm = diff_a.colwise().squaredNorm();
    double sigma_a = a_sqnorm.sum() / n; // sigma_x

    Eigen::MatrixXd b_centroid_repmat(b_centroid.rowwise().replicate(n));
    Eigen::MatrixXd diff_b = clBMat - b_centroid_repmat;
    Eigen::MatrixXd b_sqnorm = diff_b.colwise().squaredNorm();
    double sigma_b = b_sqnorm.sum() / n; // sigma_y

    Eigen::Tensor<double, THREE_> sigma_xy_complete(m, m, n); // Sigma_xy
    sigma_xy_complete.setZero();
    for (int i = 0; i < n; ++i)
    {
        // MATLAB equivalent:
        // sigma_xy_complete.block<dim,dim,n>(0,0,i) = diff_b.col(i) * diff_a.col(ii).transpose();

        Eigen::Vector2d dAi = diff_a.col(i);
        Eigen::Vector2d dBi = diff_b.col(i);
        Eigen::MatrixXd tmp = dBi * dAi.transpose(); // this is actually a Matrix2d, but conversion function takes MatrixXd as input

        Eigen::Tensor<double, TWO_> tmpTens = Matrix_to_Tensor(tmp, TWO_, TWO_);
        sigma_xy_complete.chip(i, 2) = tmpTens;

        std::cout << "i " << i << std::endl;
        std::cout << "tmp " << std::endl
                  << tmp << std::endl;
        std::cout << "sigma_xy_complete.chip(i, 2) " << std::endl
                  << sigma_xy_complete.chip(i, 2) << std::endl;
        std::cout << "tmpTens " << std::endl
                  << tmpTens << std::endl;
        std::cout << std::endl;
    }
    Eigen::Tensor<double, 2> sigma_xy_tensor(2, 2);
    // built-in version
    // std::cout << std::endl << "sigma_xy_tensor" << sigma_xy_tensor << std::endl;
    // std::array<int, 3> two_dims{{2,2}};
    // sigma_xy_tensor = sigma_xy_complete.sum(two_dims);
    // hand-made version
    for (int i = 0; i < n; ++i)
    {
        sigma_xy_tensor += sigma_xy_complete.chip(i, 2);
    }
    Eigen::MatrixXd sigma_xy = Tensor_to_Matrix(sigma_xy_tensor, m, m);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd_sigma_xy(sigma_xy, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::FullPivLU<Eigen::Matrix2d> sigma_xy_decomp(sigma_xy); // needed to compute rank
    auto rank_sigma_xy = sigma_xy_decomp.rank();
    if ((int)rank_sigma_xy < m - 1)
    {
        std::cerr << "Error: rank(sigma_xy) < dim - 1 -> returning eye() transf" << std::endl;
        return;
    }

    Eigen::MatrixXd u_sigma_xy = svd_sigma_xy.matrixU();
    //svd.singularValues().asDiagonal()
    Eigen::MatrixXd d_sigma_xy = Eigen::DiagonalMatrix<double, TWO_>(svd_sigma_xy.singularValues()); // called S in Eigen documentation
    Eigen::MatrixXd v_sigma_xy = svd_sigma_xy.matrixV();

    Eigen::Matrix2d s_sigma_xy = Eigen::Matrix2d::Identity();
    if (sigma_xy.determinant() < 0)
        s_sigma_xy(1, 1) = -1;
    if ((int)rank_sigma_xy == m - 1)
        if (u_sigma_xy.determinant() * v_sigma_xy.determinant() == -1)
            s_sigma_xy(1, 1) = -1;

    Eigen::Matrix2d R2 = u_sigma_xy * s_sigma_xy * v_sigma_xy.transpose();
    double c = (d_sigma_xy * s_sigma_xy).trace() / sigma_a;
    Eigen::Vector2d transl = b_centroid - c * R2 * a_centroid;

    transfOut.linear() = R2;
    transfOut.translation() = transl;
    transfOut.makeAffine();
}