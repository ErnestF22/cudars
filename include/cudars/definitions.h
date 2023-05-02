/**
 * CudARS: Angular Radon Spectrum - CUDA version
 * Copyright (C) 2017-2020 Dario Lodi Rizzini.
 * Copyright (C) 2021- Dario Lodi Rizzini, Ernesto Fontana.
 *
 * CudARS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CudARS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with CudARS.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <iostream>
#include <limits>
#include <thrust/host_vector.h>
#include <cmath>
#include <vector_functions.h>

#include <rofl/common/peak_finder_d.h>

#define ARS_PRINT(MSG) std::cout << __FILE__ << "," << __LINE__ << ": " << MSG << std::endl;

#define ARS_ERROR(MSG) std::cerr << __FILE__ << "," << __LINE__ << ": " << MSG << std::endl;

#define ARS_VARIABLE(X1) std::cout << __FILE__ << "," << __LINE__ << ": " << (#X1) << " " << (X1) << std::endl;

#define ARS_VARIABLE2(X1, X2) std::cout << __FILE__ << "," << __LINE__ << ": " << (#X1) << " " << (X1) \
                                        << ", " << (#X2) << " " << (X2) << std::endl;

#define ARS_VARIABLE3(X1, X2, X3) std::cout << __FILE__ << "," << __LINE__ << ": " << (#X1) << " " << (X1) \
                                            << ", " << (#X2) << " " << (X2) << ", " << (#X3) << " " << (X3) << std::endl;

#define ARS_VARIABLE4(X1, X2, X3, X4) std::cout << __FILE__ << "," << __LINE__ << ": " << (#X1) << " " << (X1) \
                                                << ", " << (#X2) << " " << (X2) << ", " << (#X3) << " " << (X3) << ", " << (#X4) << " " << (X4) << std::endl;

#define ARS_VARIABLE5(X1, X2, X3, X4, X5) std::cout << __FILE__ << "," << __LINE__ << ": " << (#X1) << " " << (X1)                                  \
                                                    << ", " << (#X2) << " " << (X2) << ", " << (#X3) << " " << (X3) << ", " << (#X4) << " " << (X4) \
                                                    << ", " << (#X5) << " " << (X5) << std::endl;

#define ARS_VARIABLE6(X1, X2, X3, X4, X5, X6) std::cout << __FILE__ << "," << __LINE__ << ": " << (#X1) << " " << (X1)                                  \
                                                        << ", " << (#X2) << " " << (X2) << ", " << (#X3) << " " << (X3) << ", " << (#X4) << " " << (X4) \
                                                        << ", " << (#X5) << " " << (X5) << ", " << (#X6) << " " << (X6) << std::endl;

#define ARS_VARIABLE7(X1, X2, X3, X4, X5, X6, X7) std::cout << __FILE__ << "," << __LINE__ << ": " << (#X1) << " " << (X1)                                  \
                                                            << ", " << (#X2) << " " << (X2) << ", " << (#X3) << " " << (X3) << ", " << (#X4) << " " << (X4) \
                                                            << ", " << (#X5) << " " << (X5) << ", " << (#X6) << " " << (X6) << ", " << (#X7) << " " << (X7) << std::endl;

#define ARS_ASSERT(COND)                                                                          \
   if (!(COND))                                                                                   \
   {                                                                                              \
      std::cerr << __FILE__ << "," << __LINE__ << ": assertion failed on " << #COND << std::endl; \
      exit(-1);                                                                                   \
   }

namespace cudars
{

   static const size_t Two = 2;   // useful for expanding (i,j) indexing into  i*Two+j
   static const size_t Three = 3; // useful for expanding (i,j) indexing into  i*Three+j
   static const size_t Nine = 9;  // useful for expanding (i,j) indexing into  i*NUM_COLS+j

   using Scalar = double;

   using Quaterniond = double4;
   using Quaternionf = float4;

   struct EulerAngles
   {
      double roll, pitch, yaw;
   };

   //    using Vector2 = Eigen::Vector2d;

   using Vec2i = int2;

   using Vec2d = double2;
   using VecVec2d = thrust::host_vector<Vec2d>;

   using Mat2i = int4; // w x \n y z (row-major)

   using Mat2d = double4; // w x \n y z (row-major)
   using VecMat2d = thrust::host_vector<Mat2d>;

   class Affine2d
   {
   public: // TODO: improve separation between public and private
      double data_[Nine];

      double rot_; // TODO: update these rot, transl after making products, transformations, ...
      double translX_;
      double translY_;

      Affine2d();

      Affine2d(double rot, double tx, double ty);

      virtual ~Affine2d();

      void initdata(double r, double tx, double ty);

      double determinant() const;

      void invert();

      Affine2d inverse();

      bool isLastRowOK() const;

      bool isScale1();

      double at(int r, int c) const;

      Vec2d translation() const;

      friend std::ostream &operator<<(std::ostream &os, cudars::Affine2d const &m)
      {
         return os << m.data_[0 * cudars::Three + 0] << " \t" << m.data_[0 * cudars::Three + 1] << " \t" << m.data_[0 * cudars::Three + 2] << " \n"
                   << m.data_[1 * cudars::Three + 0] << " \t" << m.data_[1 * cudars::Three + 1] << " \t" << m.data_[1 * cudars::Three + 2] << " \n"
                   << m.data_[2 * cudars::Three + 0] << " \t" << m.data_[2 * cudars::Three + 1] << " \t" << m.data_[2 * cudars::Three + 2] << " \n";
      }
   };

   enum class ArsKernelIso2dComputeMode : unsigned int
   {
      PNEBI_DOWNWARD = 0,
      PNEBI_LUT = 1
   };

   // translation definitions

   template <typename T, int cn>
   struct MakePt;
   template <>
   struct MakePt<float, 3>
   {
      using type = float3;
   };
   template <>
   struct MakePt<double, 3>
   {
      using type = double3;
   };
   template <>
   struct MakePt<double, 2>
   {
      using type = double2;
   };

   // Prioqueue, Box definitions
   // Node
   typedef struct node
   {
      int data;

      // Lower values indicate
      // higher priority
      int priority;

      struct node *next;

   } Node;

   /** @brief Box struct used in BBTransl
    */
   struct CuBox
   {
      Vec2d min_;
      Vec2d max_;
      double lower_;
      double upper_;
      double eps_;
   };

} // end of namespace

// std::ostream &operator<<(std::ostream &os, cudars::Vec2d const &v) {
//     return os << v.x << " \t" << v.y << "\n";
// }

#endif /* DEFINITIONS_H */
