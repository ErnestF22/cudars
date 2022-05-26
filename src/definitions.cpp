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

#include "cudars/definitions.h"

namespace cudars
{

    Affine2d::Affine2d()
    {
        rot_ = 0.0;
        translX_ = 0.0;
        translY_ = 0.0;

        initdata(rot_, translX_, translY_);
    }

    Affine2d::Affine2d(double rot, double tx, double ty)
    {
        rot_ = rot;
        translX_ = tx;
        translY_ = ty;

        initdata(rot, tx, ty);
    }

    Affine2d::~Affine2d()
    {
    }

    void Affine2d::initdata(double r, double tx, double ty)
    {
        data_[0 * Three + 0] = cos(r);
        data_[0 * Three + 1] = -sin(r);
        data_[1 * Three + 0] = -data_[0 * Three + 1];
        data_[1 * Three + 1] = data_[0 * Three + 0];

        data_[0 * Three + 2] = tx;
        data_[1 * Three + 2] = ty;

        data_[2 * Three + 2] = 1.0;

        data_[2 * Three + 0] = 0.0;
        data_[2 * Three + 1] = 0.0;
    }

    // assumes matrix indices start from 0 (0,1 and 2)
    double Affine2d::determinant() const
    {
        // computed like it is a normal 3d matrix
        const double m00 = at(0, 0);
        const double m01 = at(0, 1);
        const double m02 = at(0, 2);
        const double m10 = at(1, 0);
        const double m11 = at(1, 1);
        const double m12 = at(1, 2);
        const double m20 = at(2, 0);
        const double m21 = at(2, 1);
        const double m22 = at(2, 2);

        // double det = (m00 * m11 * m22) - (m00 * m12 * m21) + (m01 * m12 * m20) - (m01 * m10 * m22) + (m02 * m10 * m21) - (m02 * m11 * m20); //Laplace (more general) rule
        double det = m00 * m11 * m22 + m01 * m12 * m20 + m02 * m10 * m21 - m00 * m12 * m21 - m01 * m10 * m22 - m02 * m11 * m20; // Sarrus rule
        return det;
    }

    void Affine2d::invert()
    {
        double det = determinant();

        const double m00 = at(0, 0);
        const double m01 = at(0, 1);
        const double m02 = at(0, 2);
        const double m10 = at(1, 0);
        const double m11 = at(1, 1);
        const double m12 = at(1, 2);
        const double m20 = at(2, 0);
        const double m21 = at(2, 1);
        const double m22 = at(2, 2);

        data_[0 * Three + 0] = (m11 * m22 - m12 * m21) / det;
        data_[0 * Three + 1] = (m02 * m21 - m01 * m22) / det;
        data_[0 * Three + 2] = (m01 * m12 - m02 * m11) / det;
        data_[1 * Three + 0] = (m12 * m20 - m10 * m22) / det;
        data_[1 * Three + 1] = (m00 * m22 - m02 * m20) / det;
        data_[1 * Three + 2] = (m02 * m10 - m00 * m12) / det;
        data_[2 * Three + 0] = (m10 * m21 - m11 * m20) / det;
        data_[2 * Three + 1] = (m01 * m20 - m00 * m21) / det;
        data_[2 * Three + 2] = (m00 * m11 - m01 * m10) / det;
    }

    Affine2d Affine2d::inverse()
    {
        double det = determinant();

        const double m00 = at(0, 0);
        const double m01 = at(0, 1);
        const double m02 = at(0, 2);
        const double m10 = at(1, 0);
        const double m11 = at(1, 1);
        const double m12 = at(1, 2);
        const double m20 = at(2, 0);
        const double m21 = at(2, 1);
        const double m22 = at(2, 2);

        Affine2d out;

        out.data_[0 * Three + 0] = (m11 * m22 - m12 * m21) / det;
        out.data_[0 * Three + 1] = (m02 * m21 - m01 * m22) / det;
        out.data_[0 * Three + 2] = (m01 * m12 - m02 * m11) / det;
        out.data_[1 * Three + 0] = (m12 * m20 - m10 * m22) / det;
        out.data_[1 * Three + 1] = (m00 * m22 - m02 * m20) / det;
        out.data_[1 * Three + 2] = (m02 * m10 - m00 * m12) / det;
        out.data_[2 * Three + 0] = (m10 * m21 - m11 * m20) / det;
        out.data_[2 * Three + 1] = (m01 * m20 - m00 * m21) / det;
        out.data_[2 * Three + 2] = (m00 * m11 - m01 * m10) / det;

        return out;
    }

    bool Affine2d::isLastRowOK() const
    {
        double a20 = data_[2 * Three + 0];
        double a21 = data_[2 * Three + 1];
        double a22 = data_[2 * Three + 2];

        if (a20 == 0 && a21 == 0 && a22 == 1)
            return true;

        printf("BAD LAST ROW\n");
        return false;
    }

    bool Affine2d::isScale1()
    {
        double a22 = data_[2 * Three + 2];

        if (a22 == 1)
            return true;

        printf("BAD SCALE\n");
        return false;
    }

    double Affine2d::at(int r, int c) const
    {
        if (r >= 0 && r < Three && c >= 0 && c < Three)
            return data_[r * Three + c];
        else
        {
            printf("ERROR accessing matrix with .at() method!\n");
            return 1000000;
        }
    }

    Vec2d Affine2d::translation() const
    {
        const double a22 = data_[2 * Three + 2];

        return make_double2(data_[0 * Three + 2] / a22, data_[1 * Three + 2] / a22);
    }

}