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

#include "cudars/utils.h"
#include "cudars/definitions.h"

namespace cudars
{

    double mod180(double angle)
    {
        return (angle - floor(angle / M_PI) * M_PI);
    }

    void diagonalize(const Mat2d &m, double &lmin, double &lmax, double &theta)
    {
        double a, b, c, s;

        // Diagonalizes sigma12
        a = 0.5 * (m.z - m.w);
        b = 0.5 * (m.x + m.y);
        // ARS_VARIABLE2(a, b);

        theta = 0.5 * atan2(-b, a);

        c = cos(theta);
        s = sin(theta);
        lmax = m.w * c * c + m.z * s * s + (m.x + m.y) * c * s;
        lmin = m.w * s * s + m.z * c * c - (m.x + m.y) * c * s;
        // ARS_VARIABLE3(theta, lmax, lmin);

        if (lmax < lmin)
        {
            theta += 0.5 * M_PI;
            std::swap(lmax, lmin);
            // ARS_PRINT("after swap: lmin " << lmin << " < lmax " << lmax << ", theta + PI/2: " << theta);
        }
    }

    void diagonalize(const Mat2d &m, Mat2d &l, Mat2d &v)
    {
        double lmin, lmax, theta;

        diagonalize(m, lmin, lmax, theta);
        cudars::resetToZero(l);
        cudars::setDiagonal(l, lmax, lmin);
        cudars::make2dRotMat(v, theta);
    }

    void saturateEigenvalues(Mat2d &covar, double sigmaMinSquare)
    {
        Mat2d v;
        double lmin, lmax, theta;

        diagonalize(covar, lmin, lmax, theta);
        if (lmin < sigmaMinSquare)
        {
            lmin = sigmaMinSquare;
        }
        if (lmax < sigmaMinSquare)
        {
            lmax = sigmaMinSquare;
        }
        fillRowMajor(covar, lmax, 0.0, 0.0, lmin);
        make2dRotMat(v, theta);
        //                covar = v * covar * v.transpose();
        Mat2d tmpProdResult;
        mat2dProd(tmpProdResult, v, covar);
        transpose(v);
        mat2dProd(covar, tmpProdResult, v);
        transpose(v); // transpose back after using it for the product
    }

    int ceilPow2(int n)
    {
        ARS_ASSERT(n > 0);

        int exponent = ceil(log2(n));

        int nPadded = std::pow<int>(2, exponent);
        std::cout << "ceilPow2(" << n << ") = " << nPadded << std::endl;

        return nPadded;
    }

    int sumNaturalsUpToN(int n)
    {
        ARS_ASSERT(n > 0);

        int result = 0.5 * n * (n + 1);

        std::cout << "sumNaturals(" << n << ") = " << result << std::endl;

        return result;
    }

    double distancePointBox(const Vec2d &p,
                            const Vec2d &boxMin,
                            const Vec2d &boxMax)
    {
        double dist = 0.0;
        double len;
        for (int d = 0; d < 2; ++d)
        {
            // if (boxMin(d) <= p(d) && p(d) <= boxMax(d))
            if (idxGetter(boxMin, d) <= idxGetter(p, d) && idxGetter(p, d) <= idxGetter(boxMax, d))
            {
                len = 0.0;
            }
            // else if (p(d) < boxMin(d))
            else if (idxGetter(p, d) < idxGetter(boxMin, d))
            {
                len = idxGetter(boxMin, d) - idxGetter(p, d);
            }
            else
            {
                len = idxGetter(p, d) - idxGetter(boxMax, d);
            }
            dist += len * len;
        }
        return dist;
    }

    void findBoundingBox(const cudars::VecVec2d &pts,
                         cudars::Vec2d &ptMin,
                         cudars::Vec2d &ptMax)
    {
        for (int i = 0; i < pts.size(); ++i)
        {
            for (int d = 0; d < 2; ++d)
            {
                // if (i == 0 || pts[i](d) < ptMin(d))
                // {
                //     ptMin(d) = pts[i](d);
                // }
                // if (i == 0 || pts[i](d) > ptMax(d))
                // {
                //     ptMax(d) = pts[i](d);
                // }
                if (i == 0 || cudars::idxGetter(pts[i], d) < cudars::idxGetter(ptMin, d))
                {
                    cudars::idxSetter(ptMin, d, cudars::idxGetter(pts[i], d));
                }
                if (i == 0 || cudars::idxGetter(pts[i], d) > cudars::idxGetter(ptMax, d))
                {
                    cudars::idxSetter(ptMax, d, cudars::idxGetter(pts[i], d));
                }
            }
        }
    }

    // --------------------------------------------------------
    // Below: Vec2d and Mat2d util functions (simpler reimplementation of basic Eigen functions)
    // --------------------------------------------------------

    __host__ void printVec2d(const Vec2d &vec)
    {
        std::cout << std::endl
                  << vec.x << "\t" << vec.y << std::endl; // print mat2d
    }

    __host__ void printVec2d(const Vec2d &vec, const std::string &name)
    {
        std::cout << name << std::endl
                  << vec.x << "\t" << vec.y << std::endl; // print mat2d
    }

    __device__ void printfVec2d(const Vec2d &vec, const char *name)
    {
        printf("%s\n%f\t%f\n", name, vec.x, vec.y); // printf mat2d
    }

    __host__ __device__ double idxGetter(const Vec2d &vec, int idx)
    {
        double retval = 0.0;
        switch (idx)
        {
        case 0:
            retval = vec.x;
            break;
        case 1:
            retval = vec.y;
            break;
        default:
            printf("Bad idx getter (vec)!\n");
        }
        return retval; // TODO: maybe try to return directly from switch cases
    }

    __host__ __device__ double idxGetter(const Mat2d &vec, int idx)
    {
        double retval = 0.0;
        switch (idx)
        {
        case 0:
            retval = vec.w;
            break;
        case 1:
            retval = vec.x;
            break;
        case 2:
            retval = vec.y;
            break;
        case 3:
            retval = vec.z;
            break;
        default:
            printf("Bad idx getter (mat)!\n");
        }
        return retval;
    }

    __host__ __device__ void idxSetter(Vec2d &vec, int idx, double val)
    {
        switch (idx)
        {
        case 0:
            vec.x = val;
            break;
        case 1:
            vec.y = val;
            break;
        default:
            printf("Bad idx setter (vec)!\n");
        }
    }

    __host__ __device__ void idxSetter(Mat2d &vec, int idx, double val)
    {
        switch (idx)
        {
        case 0:
            vec.w = val;
            break;
        case 1:
            vec.x = val;
            break;
        case 2:
            vec.y = val;
            break;
        case 3:
            vec.z = val;
            break;
        default:
            printf("Bad idx setter (mat)!\n");
        }
    }

    __host__ __device__ void resetToZero(Vec2d &vec)
    {
        vec.x = 0.0;
        vec.y = 0.0;
    }

    __host__ __device__ void resetToZero(Mat2d &mtx)
    {
        mtx.w = 0.0;
        mtx.x = 0.0;
        mtx.y = 0.0;
        mtx.z = 0.0;
    }

    __host__ __device__ void setToIdentity(Mat2d &mtx)
    {
        //        data_[0 * Two + 0] = 1.0; // = data[0]
        //        data_[0 * Two + 1] = 0.0; // = data[1]
        //        data_[1 * Two + 0] = 0.0; // = data[2]
        //        data_[1 * Two + 1] = 1.0; // = data[3]
        mtx.w = 1.0;
        mtx.x = 0.0;
        mtx.y = 0.0;
        mtx.z = 1.0;
    }

    __host__ __device__ void setDiagonal(Mat2d &mtx, double a11, double a22)
    {
        mtx.w = a11;
        mtx.x = 0.0;
        mtx.y = 0.0;
        mtx.z = a22;
    }

    __host__ __device__ void make2dRotMat(Mat2d &mtx, double theta)
    {
        double cth = cos(theta); // avoiding useless function calling
        double sth = sin(theta);
        mtx.w = cth;
        mtx.x = -sth;
        mtx.y = sth;
        mtx.z = cth;
    }

    __host__ __device__ void fillVec2d(Vec2d &vec, double x, double y)
    {
        vec.x = x;
        vec.y = y;
    }

    __host__ __device__ void fillRowMajor(Mat2d &mtx, double a, double b, double c, double d)
    {
        mtx.w = a;
        mtx.x = b;
        mtx.y = c;
        mtx.z = d;
    }

    __host__ __device__ void scalarMul(Vec2d &vec, double d)
    {
        vec.x *= d;
        vec.y *= d;
    }

    __host__ __device__ Vec2d scalarMulWRV(const Vec2d &vec, double d)
    {
        Vec2d res;

        res.x = vec.x * d;
        res.y = vec.y * d;

        return res;
    }

    __host__ __device__ void scalarMul(Mat2d &mtx, double d)
    {
        mtx.w *= d;
        mtx.x *= d;
        mtx.y *= d;
        mtx.z *= d;
    }

    __host__ __device__ Mat2d scalarMulWRV(const Mat2d &mtx, double d)
    {
        Mat2d res;

        res.w = mtx.w * d;
        res.x = mtx.x * d;
        res.y = mtx.y * d;
        res.z = mtx.z * d;

        return res;
    }

    __host__ __device__ void scalarDiv(Vec2d &vec, double d)
    {
        if (d == 0)
            assert(false);

        vec.x /= d;
        vec.y /= d;
    }

    __host__ __device__ Vec2d scalarDivWRV(const Vec2d &vec, double d)
    {
        if (d == 0)
            assert(false);

        Vec2d res;

        res.x = vec.x / d;
        res.y = vec.y / d;

        return res;
    }

    __host__ void printMat2d(const Mat2d &mtx, const std::string &name)
    {
        std::cout << name << std::endl
                  << mtx.w << "\t" << mtx.x << std::endl
                  << mtx.y << "\t" << mtx.z << std::endl; // print mat2d
    }

    __device__ void printfMat2d(const Mat2d &mtx, const char *name)
    {
        printf("%s\n%f\t%f\n%f\t%f\n", name, mtx.w, mtx.x, mtx.y, mtx.z); // printf mat2d
    }

    __host__ __device__ void scalarDiv(Mat2d &mtx, double d)
    {
        if (d == 0)
            assert(false);

        mtx.w /= d;
        mtx.x /= d;
        mtx.y /= d;
        mtx.z /= d;
    }

    __host__ __device__ Mat2d scalarDivWRV(const Mat2d &mtx, double d)
    {
        if (d == 0)
            assert(false);

        Mat2d res;

        res.w = mtx.w / d;
        res.x = mtx.x / d;
        res.y = mtx.y / d;
        res.z = mtx.z / d;

        return res;
    }

    __host__ __device__ void transpose(Mat2d &mtx)
    {
        double tmp = mtx.x;
        mtx.x = mtx.y;
        mtx.y = tmp;
    }

    __host__ __device__ Mat2d transposeWRV(const Mat2d &mtx)
    {
        Mat2d res;
        res.w = mtx.w;
        res.x = mtx.y;
        res.y = mtx.x;
        res.z = mtx.z;
        return res;
    }

    __host__ __device__ double mat2dDeterminant(const Mat2d &mtx)
    {
        //        return data_[0 * Two + 0] * data_[1 * Two + 1] - data_[0 * Two + 1] * data_[1 * Two + 0];
        return mtx.w * mtx.z - mtx.x * mtx.y;
    }

    __host__ __device__ double mat2dTrace(const Mat2d &mtx)
    {
        //        return data_[0 * Two + 0] + data_[1 * Two + 1];
        return mtx.w + mtx.z;
    }

    __host__ __device__ void mat2dInvert(Mat2d &mtx)
    {
        double detInv = 1.0 / mat2dDeterminant(mtx);

        double aOrig = mtx.w;
        double bOrig = mtx.x;
        double cOrig = mtx.y;
        double dOrig = mtx.z;

        mtx.w = dOrig * detInv;
        mtx.x = -bOrig * detInv;
        mtx.y = -cOrig * detInv;
        mtx.z = aOrig * detInv;
    }

    __host__ __device__ Mat2d mat2dInverse(const Mat2d &mtx)
    {
        double detInv = 1.0 / mat2dDeterminant(mtx);

        double aOrig = mtx.w;
        double bOrig = mtx.x;
        double cOrig = mtx.y;
        double dOrig = mtx.z;

        Mat2d r;

        r.w = dOrig * detInv;
        r.x = -bOrig * detInv;
        r.y = -cOrig * detInv;
        r.z = aOrig * detInv;

        return r;
    }

    __host__ __device__ void mat2dSum(Mat2d &resultMtx, const Mat2d &aMtx, const Mat2d &bMtx)
    {
        resultMtx.w = aMtx.w + bMtx.w;
        resultMtx.x = aMtx.x + bMtx.x;
        resultMtx.y = aMtx.y + bMtx.y;
        resultMtx.z = aMtx.z * bMtx.z;
    }

    __host__ __device__ Mat2d mat2dSumWRV(const Mat2d &aMtx, const Mat2d &bMtx)
    {
        Mat2d resultMtx;
        resultMtx.w = aMtx.w + bMtx.w;
        resultMtx.x = aMtx.x + bMtx.x;
        resultMtx.y = aMtx.y + bMtx.y;
        resultMtx.z = aMtx.z + bMtx.z;
        return resultMtx;
    }

    __host__ __device__ void mat2dDiff(Mat2d &resultMtx, const Mat2d &aMtx, const Mat2d &bMtx)
    {
        resultMtx.w = aMtx.w - bMtx.w;
        resultMtx.x = aMtx.x - bMtx.x;
        resultMtx.y = aMtx.y - bMtx.y;
        resultMtx.z = aMtx.z - bMtx.z;
    }

    __host__ __device__ Mat2d mat2dDiffWRV(const Mat2d &aMtx, const Mat2d &bMtx)
    {
        Mat2d resultMtx;
        resultMtx.w = aMtx.w - bMtx.w;
        resultMtx.x = aMtx.x - bMtx.x;
        resultMtx.y = aMtx.y - bMtx.y;
        resultMtx.z = aMtx.z - bMtx.z;
        return resultMtx;
    }

    __host__ __device__ void mat2dPlusEq(Mat2d &resultMtx, const Mat2d &aMtx)
    {
        resultMtx.w += aMtx.w;
        resultMtx.x += aMtx.x;
        resultMtx.y += aMtx.y;
        resultMtx.z += aMtx.z;
    }

    __host__ __device__ void mat2dProd(Mat2d &resultMtx, const Mat2d &aMtx, const Mat2d &bMtx)
    {
        resultMtx.w = aMtx.w * bMtx.w + aMtx.x * bMtx.y;
        resultMtx.x = aMtx.w * bMtx.x + aMtx.x * bMtx.z;
        resultMtx.y = aMtx.y * bMtx.w + aMtx.z * bMtx.y;
        resultMtx.z = aMtx.y * bMtx.x + aMtx.z * bMtx.z;
    }

    __host__ __device__ Mat2d mat2dProdWRV(const Mat2d &aMtx, const Mat2d &bMtx)
    {
        Mat2d resultMtx;
        resultMtx.w = aMtx.w * bMtx.w + aMtx.x * bMtx.y;
        resultMtx.x = aMtx.w * bMtx.x + aMtx.x * bMtx.z;
        resultMtx.y = aMtx.y * bMtx.w + aMtx.z * bMtx.y;
        resultMtx.z = aMtx.y * bMtx.x + aMtx.z * bMtx.z;
        return resultMtx;
    }

    __host__ __device__ void threeMats2dProd(Mat2d &resultMtx, const Mat2d &aMtx, const Mat2d &bMtx, const Mat2d &cMtx)
    {
        Mat2d tmp;

        //        mat2dProd(tmp, aMtx, bMtx);
        //        mat2dProd(resultMtx, tmp, cMtx);

        Mat2d aCopy = aMtx;
        Mat2d bCopy = bMtx;
        Mat2d cCopy = cMtx;

        mat2dProd(tmp, aCopy, bCopy);
        mat2dProd(resultMtx, tmp, cCopy);
    }

    __host__ __device__ double vec2norm(const Vec2d &v)
    {
        return sqrt(v.x * v.x + v.y * v.y);
    }

    __host__ __device__ double vec2squarednorm(const Vec2d &v)
    {
        return (v.x * v.x + v.y * v.y);
    }

    __host__ __device__ void vec2sum(Vec2d &result, const Vec2d &a, const Vec2d &b)
    {
        result.x = a.x + b.x;
        result.y = a.y + b.y;
    }

    __host__ __device__ Vec2d vec2sumWRV(const Vec2d &a, const Vec2d &b)
    {
        Vec2d result;
        result.x = a.x + b.x;
        result.y = a.y + b.y;
        return result;
    }

    __host__ __device__ void vec2dPlusEq(Vec2d &result, const Vec2d &v)
    {
        result.x += v.x;
        result.y += v.y;
    }

    __host__ __device__ void vec2diff(Vec2d &result, const Vec2d &a, const Vec2d &b)
    {
        result.x = a.x - b.x;
        result.y = a.y - b.y;
    }

    __host__ __device__ Vec2d vec2diffWRV(const Vec2d &a, const Vec2d &b)
    {
        Vec2d result;
        result.x = a.x - b.x;
        result.y = a.y - b.y;
        return result;
    }

    __host__ __device__ double vec2dotProduct(const Vec2d &a, const Vec2d &b)
    {
        return a.x * b.x + a.y * b.y;
    }

    __host__ __device__ void vec2outerProduct(Mat2d &result, const Vec2d &a, const Vec2d &b)
    {
        result.w = a.x * b.x;
        result.x = a.y * b.x;
        result.y = a.x * b.y;
        result.z = a.y * b.y;
    }

    __host__ __device__ Mat2d vec2outerProductWRV(const Vec2d &a, const Vec2d &b)
    {
        Mat2d result;

        result.w = a.x * b.x;
        result.x = a.y * b.x;
        result.y = a.x * b.y;
        result.z = a.y * b.y;

        return result;
    }

    __host__ __device__ Vec2d row2VecTimesMat2WRV(const Vec2d &v, const Mat2d &m)
    {
        Vec2d result;
        result.x = (v.x * m.w) + (v.y * m.y);
        result.y = (v.x * m.x) + (v.y * m.z);

        //        result.isCol_ = false;

        return result;
    }

    __host__ __device__ Vec2d mat2dTimesVec2dWRV(const Mat2d &m, const Vec2d &v)
    {
        Vec2d result;
        result.x = (m.w * v.x) + (m.x * v.y);
        result.y = (m.y * v.x) + (m.z * v.y);
        return result;
    }

    __host__ __device__ void cwiseAbsWRV(Vec2d &vOut, const Vec2d &vIn)
    {
        vOut.x = fabs(vIn.x);
        vOut.y = fabs(vIn.y);
    }

    __host__ __device__ void cwiseAbsWRV(Vec2d &v)
    {
        v.x = fabs(v.x);
        v.y = fabs(v.y);
    }

    __host__ __device__ Vec2d cwiseAbsWRV(const Vec2d &v)
    {
        return make_double2(fabs(v.x), fabs(v.y));
    }

    __host__ __device__ void cwiseAbsWRV(Mat2d &vOut, const Mat2d &vIn)
    {
        vOut.x = fabs(vIn.x);
        vOut.y = fabs(vIn.y);
        vOut.z = fabs(vIn.z);
        vOut.w = fabs(vIn.w);
    }

    __host__ __device__ void cwiseAbsWRV(Mat2d &v)
    {
        v.x = fabs(v.x);
        v.y = fabs(v.y);
        v.z = fabs(v.z);
        v.w = fabs(v.w);
    }

    __host__ __device__ Mat2d cwiseAbsWRV(const Mat2d &v)
    {
        return make_double4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
    }

    __host__ __device__ void maxCoeff(double &maxVal, const Vec2d &v)
    {
        if (v.x >= v.y)
            maxVal = v.x;
        else
            maxVal = v.y;
    }

    __host__ __device__ double maxCoeffWRV(const Vec2d &v)
    {
        if (v.x >= v.y)
            return v.x;
        else
            return v.y;
    }

    __host__ __device__ void maxCoeff(double &maxVal, const Mat2d &v)
    {
        if (v.x >= v.y && v.x >= v.z && v.x >= v.w)
            maxVal = v.x;
        else if (v.y >= v.x && v.y >= v.z && v.y >= v.w)
            maxVal = v.y;
        else if (v.z >= v.x && v.z >= v.y && v.z >= v.w)
            maxVal = v.z;
        else
            maxVal = v.w;
    }

    __host__ __device__ double maxCoeffWRV(const Mat2d &v)
    {
        if (v.x >= v.y && v.x >= v.z && v.x >= v.w)
            return v.x;
        else if (v.y >= v.x && v.y >= v.z && v.y >= v.w)
            return v.y;
        else if (v.z >= v.x && v.z >= v.y && v.z >= v.w)
            return v.z;
        else
            return v.w;
    }

    // affine matrices related

    void preTransfVec2(Vec2d &p, const Affine2d &t)
    {
        if (t.isLastRowOK())
        {
            double px = p.x;
            double py = p.y;
            p.x = (px * t.data_[0 * Three + 0]) + (py * t.data_[0 * Three + 1]) + (t.data_[0 * Three + 2]);
            p.y = (px * t.data_[1 * Three + 0]) + (py * t.data_[1 * Three + 1]) + (t.data_[1 * Three + 2]);
            // p.z = 1.0;
        }
        else
        {
            printf("ERROR: Transf Matrix affine scale != 1\n");
        }
    }

    void preRotateAff2(Affine2d &t, double angle)
    {
        if (t.isLastRowOK())
        {
            double cth = cos(angle);
            double sth = sin(angle);
            Affine2d tTmpCopy = t;
            // first row
            t.data_[0 * cudars::Three + 0] = (tTmpCopy.data_[0 * cudars::Three + 0] * cth) - (tTmpCopy.data_[1 * cudars::Three + 0] * sth);
            t.data_[0 * cudars::Three + 1] = (tTmpCopy.data_[0 * cudars::Three + 1] * cth) - (tTmpCopy.data_[1 * cudars::Three + 1] * sth);
            t.data_[0 * cudars::Three + 2] = (tTmpCopy.data_[0 * cudars::Three + 2] * cth) - (tTmpCopy.data_[1 * cudars::Three + 2] * sth);
            // second row
            t.data_[1 * cudars::Three + 0] = (tTmpCopy.data_[0 * cudars::Three + 0] * sth) + (tTmpCopy.data_[1 * cudars::Three + 0] * cth);
            t.data_[1 * cudars::Three + 1] = (tTmpCopy.data_[0 * cudars::Three + 1] * sth) + (tTmpCopy.data_[1 * cudars::Three + 1] * cth);
            t.data_[1 * cudars::Three + 2] = (tTmpCopy.data_[0 * cudars::Three + 2] * sth) + (tTmpCopy.data_[1 * cudars::Three + 2] * cth);

            // third (last) row should already be ok
            //            t.data_[2 * cudars::Three + 0] = 0.0;
            //            t.data_[2 * cudars::Three + 1] = 0.0;
            //            t.data_[2 * cudars::Three + 2] = 1.0;

            //            std::cout << "t after prerotation" << std::endl;
            //            std::cout << t;
        }
        else
        {
            printf("ERROR: Transf Matrix affine scale != 1\n");
        }
    }

    void preTranslateAff2(Affine2d &t, double x, double y)
    {
        if (t.isLastRowOK())
        {
            // just last column: the other two remain untouched
            t.data_[0 * cudars::Three + 2] = t.data_[0 * cudars::Three + 2] + x; // += x
            t.data_[1 * cudars::Three + 2] = t.data_[1 * cudars::Three + 2] + y; // += y
            //            t.data_[2 * cudars::Three + 2] = 1.0;
        }
        else
        {
            printf("ERROR: Transf Matrix affine scale != 1\n");
        }
    }

    void preTranslateAff2(Affine2d &t, const Vec2d &p)
    {
        if (t.isLastRowOK())
        {
            // just last column: the other two remain untouched
            t.data_[0 * cudars::Three + 2] = t.data_[0 * cudars::Three + 2] + p.x; // += x
            t.data_[1 * cudars::Three + 2] = t.data_[1 * cudars::Three + 2] + p.y; // += y
            //            t.data_[2 * cudars::Three + 2] = 1.0;
        }
        else
        {
            printf("ERROR: Transf Matrix affine scale != 1\n");
        }
    }

    void aff2Prod(Affine2d &out, const Affine2d &a, const Affine2d &b)
    {
        if (a.isLastRowOK() && b.isLastRowOK())
        {
            // elements not mentioned are implicitly ok (or invariant if they are sum terms, because last rows are [0  0  1])

            // first column
            out.data_[0 * cudars::Three + 0] = (a.at(0, 0) * b.at(0, 0)) + (a.at(0, 1) * b.at(1, 0)); // + a.at(0,2) * b.at(2,0)
            out.data_[1 * cudars::Three + 0] = (a.at(1, 0) * b.at(0, 0)) + (a.at(1, 1) * b.at(1, 0)); // + a.at(1,2) * b.at(2,0)
            //            out.data_[2 * cudars::Three + 0] = (a.at(2, 0) * b.at(0, 0)) + (a.at(2, 1) * b.at(1, 0)) + (a.at(2,2) * b.at(2,0));
            out.data_[2 * cudars::Three + 0] = 0.0;

            // second column
            out.data_[0 * cudars::Three + 1] = (a.at(0, 0) * b.at(0, 1)) + (a.at(0, 1) * b.at(1, 1)); // + a.at(0,2) * b.at(2,1)
            out.data_[1 * cudars::Three + 1] = (a.at(1, 0) * b.at(0, 1)) + (a.at(1, 1) * b.at(1, 1)); // + a.at(1,2) * b.at(2,1)
            //            out.data_[2 * cudars::Three + 1] = (a.at(2, 0) * b.at(0, 1)) + (a.at(2, 1) * b.at(1, 1)) + (a.at(2,2) * b.at(2,1));
            out.data_[2 * cudars::Three + 1] = 0.0;

            // third column
            out.data_[0 * cudars::Three + 2] = (a.at(0, 0) * b.at(0, 2)) + (a.at(0, 1) * b.at(1, 2)) + a.at(0, 2) * b.at(2, 2);
            out.data_[1 * cudars::Three + 2] = (a.at(1, 0) * b.at(0, 2)) + (a.at(1, 1) * b.at(1, 2)) + a.at(1, 2) * b.at(2, 2);
            //            out.data_[2 * cudars::Three + 2] = (a.at(2, 0) * b.at(0, 2)) + (a.at(2, 1) * b.at(1, 2)) + (a.at(2, 2) + b.at(2, 2));
            out.data_[2 * cudars::Three + 2] = 1.0;
        }
        else
        {
            printf("ERROR: Transf Matrix last row != 0  0  1\n");
        }
    }

    Affine2d aff2ProdWRV(const Affine2d &a, const Affine2d &b)
    {
        Affine2d out;
        if (a.isLastRowOK() && b.isLastRowOK())
        {
            // elements not mentioned are implicitly ok (or invariant if they are sum terms, because last rows are [0  0  1])

            // first column
            out.data_[0 * cudars::Three + 0] = (a.at(0, 0) * b.at(0, 0)) + (a.at(0, 1) * b.at(1, 0)); // + a.at(0,2) * b.at(2,0)
            out.data_[1 * cudars::Three + 0] = (a.at(1, 0) * b.at(0, 0)) + (a.at(1, 1) * b.at(1, 0)); // + a.at(1,2) * b.at(2,0)
            //            out.data_[2 * cudars::Three + 0] = (a.at(2, 0) * b.at(0, 0)) + (a.at(2, 1) * b.at(1, 0)) + (a.at(2,2) * b.at(2,0));
            out.data_[2 * cudars::Three + 0] = 0.0;

            // second column
            out.data_[0 * cudars::Three + 1] = (a.at(0, 0) * b.at(0, 1)) + (a.at(0, 1) * b.at(1, 1)); // + a.at(0,2) * b.at(2,1)
            out.data_[1 * cudars::Three + 1] = (a.at(1, 0) * b.at(0, 1)) + (a.at(1, 1) * b.at(1, 1)); // + a.at(1,2) * b.at(2,1)
            //            out.data_[2 * cudars::Three + 1] = (a.at(2, 0) * b.at(0, 1)) + (a.at(2, 1) * b.at(1, 1)) + (a.at(2,2) * b.at(2,1));
            out.data_[2 * cudars::Three + 1] = 0.0;

            // third column
            out.data_[0 * cudars::Three + 2] = (a.at(0, 0) * b.at(0, 2)) + (a.at(0, 1) * b.at(1, 2)) + a.at(0, 2); // + a.at(0,2) * b.at(2,2) = a.at(0,2) because b.at(2,2) = 1.0
            out.data_[1 * cudars::Three + 2] = (a.at(1, 0) * b.at(0, 2)) + (a.at(1, 1) * b.at(1, 2)) + a.at(1, 2); // + a.at(1,2) * b.at(2,2) = a.at(1,2) because b.at(2,2) = 1.0
            //            out.data_[2 * cudars::Three + 2] = (a.at(2, 0) * b.at(0, 2)) + (a.at(2, 1) * b.at(1, 2)) + (a.at(2, 2) + b.at(2, 2));
            out.data_[2 * cudars::Three + 2] = 1.0;
        }
        else
        {
            printf("ERROR: Transf Matrix last row != 0  0  1\n");
        }
        return out;
    }

    Vec2d aff2TimesVec2WRV(const Affine2d &mAff, const Vec2d &p)
    {
        Vec2d result;
        if (mAff.isLastRowOK())
        {
            result.x = (mAff.at(0, 0) * p.x) + (mAff.at(0, 1) * p.y) + (mAff.at(0, 2));
            result.y = (mAff.at(1, 0) * p.x) + (mAff.at(1, 1) * p.y) + (mAff.at(1, 2));
            // result scale factor = 1
        }
        else
        {
            printf("ERROR: Transf Matrix last row != 0  0  1\n");
        }
        return result;
    }

    Affine3d aff3Inverse(const Affine2d &a)
    {
        double s0 = a.at(0, 0) * a.at(1, 1) - a.at(1, 0) * a.at(0, 1);
        double s1 = a.at(0, 0) * a.at(1, 2) - a.at(1, 0) * a.at(0, 2);
        double s2 = a.at(0, 0) * a.at(1, 3) - a.at(1, 0) * a.at(0, 3);
        double s3 = a.at(0, 1) * a.at(1, 2) - a.at(1, 1) * a.at(0, 2);
        double s4 = a.at(0, 1) * a.at(1, 3) - a.at(1, 1) * a.at(0, 3);
        double s5 = a.at(0, 2) * a.at(1, 3) - a.at(1, 2) * a.at(0, 3);

        double c5 = a.at(2, 2) * a.at(3, 3) - a.at(3, 2) * a.at(2, 3);
        double c4 = a.at(2, 1) * a.at(3, 3) - a.at(3, 1) * a.at(2, 3);
        double c3 = a.at(2, 1) * a.at(3, 2) - a.at(3, 1) * a.at(2, 2);
        double c2 = a.at(2, 0) * a.at(3, 3) - a.at(3, 0) * a.at(2, 3);
        double c1 = a.at(2, 0) * a.at(3, 2) - a.at(3, 0) * a.at(2, 2);
        double c0 = a.at(2, 0) * a.at(3, 1) - a.at(3, 0) * a.at(2, 1);

        // Should check for 0 determinant
        double det = (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);
        if (abs(det) < EPS_)
        {
            std::cerr << "Affine matrix determinant = 0 -> cannot invert -> returning Affine Identity" << std::endl;
            return Affine3d();
        }
        double invdet = 1.0 / det;

        Affine3d b; // TODO: real init of b after Affine3d class is done properly

        b.data_[0 * cudars::Four + 0] = (a.at(1, 1) * c5 - a.at(1, 2) * c4 + a.at(1, 3) * c3) * invdet;
        b.data_[0 * cudars::Four + 1] = (-a.at(0, 1) * c5 + a.at(0, 2) * c4 - a.at(0, 3) * c3) * invdet;
        b.data_[0 * cudars::Four + 2] = (a.at(3, 1) * s5 - a.at(3, 2) * s4 + a.at(3, 3) * s3) * invdet;
        b.data_[0 * cudars::Four + 3] = (-a.at(2, 1) * s5 + a.at(2, 2) * s4 - a.at(2, 3) * s3) * invdet;

        b.data_[1 * cudars::Four + 0] = (-a.at(1, 0) * c5 + a.at(1, 2) * c2 - a.at(1, 3) * c1) * invdet;
        b.data_[1 * cudars::Four + 1] = (a.at(0, 0) * c5 - a.at(0, 2) * c2 + a.at(0, 3) * c1) * invdet;
        b.data_[1 * cudars::Four + 2] = (-a.at(3, 0) * s5 + a.at(3, 2) * s2 - a.at(3, 3) * s1) * invdet;
        b.data_[1 * cudars::Four + 3] = (a.at(2, 0) * s5 - a.at(2, 2) * s2 + a.at(2, 3) * s1) * invdet;

        b.data_[2 * cudars::Four + 0] = (a.at(1, 0) * c4 - a.at(1, 1) * c2 + a.at(1, 3) * c0) * invdet;
        b.data_[2 * cudars::Four + 1] = (-a.at(0, 0) * c4 + a.at(0, 1) * c2 - a.at(0, 3) * c0) * invdet;
        b.data_[2 * cudars::Four + 2] = (a.at(3, 0) * s4 - a.at(3, 1) * s2 + a.at(3, 3) * s0) * invdet;
        b.data_[2 * cudars::Four + 3] = (-a.at(2, 0) * s4 + a.at(2, 1) * s2 - a.at(2, 3) * s0) * invdet;

        b.data_[3 * cudars::Four + 0] = (-a.at(1, 0) * c3 + a.at(1, 1) * c1 - a.at(1, 2) * c0) * invdet;
        b.data_[3 * cudars::Four + 1] = (a.at(0, 0) * c3 - a.at(0, 1) * c1 + a.at(0, 2) * c0) * invdet;
        b.data_[3 * cudars::Four + 2] = (-a.at(3, 0) * s3 + a.at(3, 1) * s1 - a.at(3, 2) * s0) * invdet;
        b.data_[3 * cudars::Four + 3] = (a.at(2, 0) * s3 - a.at(2, 1) * s1 + a.at(2, 2) * s0) * invdet;

        return b;
    }

    // Quaternions, Euler Angles related

    cudars::EulerAngles quatTo2dAngle(const double4 &q)
    {
        cudars::EulerAngles angles;

        // roll (x-axis rotation)
        double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
        double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
        angles.roll = std::atan2(sinr_cosp, cosr_cosp);

        // pitch (y-axis rotation)
        double sinp = 2 * (q.w * q.y - q.z * q.x);
        if (std::abs(sinp) >= 1)
            angles.pitch = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
        else
            angles.pitch = std::asin(sinp);

        // yaw (z-axis rotation)
        double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
        double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
        angles.yaw = std::atan2(siny_cosp, cosy_cosp);

        std::cout << "quat " << q.w << " " << q.x << " " << q.y << " " << q.z
                  << " -> "
                  << "roll " << angles.roll << " pitch " << angles.pitch << " yaw " << angles.yaw << std::endl;

        return angles;
    }

} // end of namespace
