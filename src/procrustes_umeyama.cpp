#include "cudars/procrustes_umeyama.h"

void procrustes_umeyama(Eigen::Affine3d& transfOut, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB, int dim) {
    transfOut = Eigen::Affine3d::Identity();

    size_t m = dim; 
    size_t n = std::min<size_t>(cloudA->size(), cloudB->size()); //TODO: fix when size(cloudA)!=size(cloudB)

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
    Eigen::MatrixXf u1=svd1.matrixU();
    Eigen::MatrixXf v1=svd1.matrixU();

    Eigen::Matrix3f s_min = Eigen::Matrix3f::Identity();
    if (svd1InputMat.determinant() < 0) 
        s_min(dim,dim) = -1;

         
    Eigen::FullPivLU<Eigen::Matrix3f> luDecomp(svd1InputMat);
    auto rank_abt = luDecomp.rank();
    if ((int) rank_abt < m - 1)
        std::cerr << "Error: rank(a*b') < dim - 1 -> returning eye transf" << std::endl;
    return;
    
    Eigen::Matrix3f R1 = Eigen::Matrix3f::Identity();
    if (rank_abt == m-1){
        Eigen::Matrix3f s_argmin = Eigen::Matrix3f::Identity();
        if (u1.determinant() * v1.determinant() == -1)
            s_argmin(dim,dim) = -1;
        Eigen::MatrixXf R1 = u1 * s_argmin * v1.transpose();
    }
    else if (rank_abt == m)
        R1 = u1 * s_min * v1.transpose();

    // TODO: add computation of minimum value of mean squared error

    Eigen::Vector3f a_centroid = clAMat.rowwise().mean(); //mu_x
    Eigen::Vector3f b_centroid = clBMat.rowwise().mean(); //mu_y

    
//     a_centroid_vec = repmat(a_centroid,1,n);
//     diff_a = a - a_centroid_vec;
//     a_sqnorm = (vecnorm(diff_a,2,1)).^2;    
//     sigma_a = sum(a_sqnorm) / n; %sigma_x

//     b_centroid_vec = repmat(b_centroid,1,n);
//     diff_b = b - b_centroid_vec;
//     b_sqnorm = (vecnorm(diff_b,2,1)).^2;    
//     sigma_b = sum(b_sqnorm) / n; %sigma_y

//     sigma_xy_complete = zeros(dim,dim,n); %Sigma_xy
//     for ii = 1:n
//         sigma_xy_complete(:,:,ii) = diff_b(:,ii) * diff_a(:,ii)';
//     end
//     sigma_xy = sum(sigma_xy_complete, 3) ./ n;

//     [u_sigma_xy,d_sigma_xy,v_sigma_xy] = svd(sigma_xy);

//     rank_sigma_xy = rank(sigma_xy)
//     if rank_sigma_xy < m - 1
//         disp("Error: rank(sigma_xy) < dim - 1 -> returning eye() transf");
//         return;
//     end

//     s_sigma_xy = eye(dim);
//     if det(sigma_xy) < 0
//         s_sigma_xy (dim,dim) = -1;
//     end    
//     if rank_sigma_xy == m-1
//         if det(u_sigma_xy) * det(v_sigma_xy) == -1 
//             s_sigma_xy(dim,dim) = -1;
//         end
//     end
    

//     R2 = u_sigma_xy * s_sigma_xy * v_sigma_xy'
//     c = trace(d_sigma_xy * s_sigma_xy) / sigma_a
//     transl = b_centroid - c * R2 * a_centroid
    
//     transf_out.R = R2;
//     transf_out.Translation = transl;


}