/**
 * ARS - Angular Radon Spectrum 
 * Copyright (C) 2017-2020 Dario Lodi Rizzini.
 *
 * ARS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * ARS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with ARS.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef ARS2D_H
#define ARS2D_H

#include <iostream>

#include "cudars/definitions.h"
#include "cudars/functions.h"
#include "cudars/BBOptimizer1d.h"
#include "cudars/ArsKernelIsotropic2d.h"
//#include "cudars/ArsKernelAnisotropic2d.h"


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>



namespace cudars {

    /** Computes coefficients of Fourier series Correlation. 
     * Given two series Fsrc(t) and Fdst(t) with coefficients:
     *   Fsrc(t) = \sum_{i=0}^{n} ( fourierSrc[2*i] * cos(2*i*t) + fourierSrc[2*i+1] * sin(2*i*t) )
     *   Fdst(t) = \sum_{i=0}^{n} ( fourierDst[2*i] * cos(2*i*t) + fourierDst[2*i+1] * sin(2*i*t) )
     * the correlation of the two function is the integral over period T:
     *   Fcor(h) = \int_{0}^{T} Fsrc(t+h) Fdst(t) dt
     * where Fcor(h) is still represented as a Fourier serie with coeffients fourierCor[]. 
     */
    void computeFourierCorr(const std::vector<double>& fourierSrc, const std::vector<double>& fourierDst, std::vector<double>& fourierCor);

    // --------------------------------------------------------
    // ARS 2D CLASS
    // --------------------------------------------------------

    /** Class for computing Angular Radon Spectum (ARS) of a set of points.
     */
    class AngularRadonSpectrum2d {
    public:

        //        enum ComputeMode {
        //            PNEBI_DOWNWARD, PNEBI_LUT
        //        };

        /** Default constructor. 
         */
        AngularRadonSpectrum2d();

        /** Default constructor. 
         */
        AngularRadonSpectrum2d(const std::vector<double>& coeffs);

        /** Destructor. 
         */
        ~AngularRadonSpectrum2d();

        /** Sets the order of truncated Fourier series expansion of ARS.
         * WARNING: It resets the coefficients!
         */
        void setARSFOrder(int n);

        /** Sets the maximum tollerance on theta during the computation of maximum.
         */
        void setThetaToll(double thetaToll);

        /**
         * Sets the mode for computing ARS coefficients in the case of isotropic kernels. 
         * @param mode the desired mode
         */
        void setComputeMode(ArsKernelIso2dComputeMode mode);

        /**
         * Returns a string description of the current mode. 
         * @return 
         */
        const std::string& getComputeModeName() const;

        /**
         * Sets the number of intervals used in the computation of Fourier coeffcients 
         * of anisotropic kernels. 
         * Since the closed-form equation of Fourier coefficients is unknown in 
         * anisotropic case, the numerical integration with step M_PI / anisotropicStep_
         * is used instead. 
         * @param as
         */
        //        void setAnisotropicStep(int as) {
        //            anisotropicStep_ = as;
        //        }

        /** Returns const reference to ARS Fourier coefficients. 
         * Coefficients are obtained from a Gaussian Mixture Model (GMM) representing
         * a point set distribution with uncertainty. 
         */
        const std::vector<double>& coefficients() const;

        /**
         * Returns the correlation norm of ARS. 
         */
        double normCorr() const;

        /**
         * Sets the ARS Fourier coefficients. 
         * @warning This method is a "backdoor" w.r.t. the insert methods that 
         * computes the coefficients directly from the point set in GMM form. 
         * Accordingly it should be used carefully to avoid inconsistencies. 
         * @param coeffs the coefficients
         */
        void setCoefficients(const std::vector<double>& coeffs);

        /**
         * Alternative to std::vector method; the only difference is the type of the parameter
         */
        void setCoefficients(double* coeffs, size_t coeffsSz);

        /** * Inserts the given points and computes all the data about point pairs and 
         * computes the coefficients of the Fourier series representing the ARS 
         * of the point set.
         * All the Gaussian distributions are isotropic, have the same standard deviation 
         * and the same weight in the mixture. 
         * @param means mean values of the distributions
         * @param sigma the standard deviation (not variance!) of the identical isotropic distributions
         */
        void insertIsotropicGaussians(const VecVec2d& means, double sigma);

        /**
         * Inserts the given points and computes all the data about point pairs and 
         * computes the coefficients of the Fourier series representing the ARS 
         * of the point set.
         * Hypothesis: the weights of the input Gaussian distributions of the mixture
         * are assumed to be equal. 
         * @param means mean values of the distributions
         * @param sigmas standard deviations (not variances!) of the isotropic distributions
         */
        void insertIsotropicGaussians(const VecVec2d& means, const std::vector<double>& sigmas);

        /**
         * Inserts the given points and computes all the data about point pairs and 
         * computes the coefficients of the Fourier series representing the ARS 
         * of the point set.
         * @param means mean values of the distributions
         * @param sigmas standard deviations (not variances!) of the isotropic distributions
         * @param weights the weights of each distribution of the mixture
         */
        void insertIsotropicGaussians(const VecVec2d& means, const std::vector<double>& sigmas, const std::vector<double>& weights);

        /**
         * Inserts the given anisotropic gaussians. 
         * Pre-condition: means.size() == covars.size(). 
         * @param means the mean values of Gaussians PDF representing points
         * @param covars the covariance matrices of Gaussians PDF representing point uncertainties
         */
        void insertAnisotropicGaussians(const VecVec2d& means, const VecMat2d& covars, const std::vector<double>& weights);

        /** Initializes LUT (the LUT is used by initARSFRecursDownLUT).
         */
        void initLUT(double precision);

        /** Evaluates the ARS Fourier using the coefficients obtained from downward recursion. 
         */
        double eval(double theta) const;

        /** Finds the maximum of ARS Fourier.
         */
        double findMax() const;

        /** Finds the maximum of ARS Fourier.
         */
        double findMax(double& thetaMax) const;

        /**
         * Finds the maximum of ARS Fourier on the given interval [thetaLow, thetaUpp].
         * @param thetaLow
         * @param thetaUpp
         * @return the maximum value of correlation function. 
         */
        double findMax(double thetaLow, double thetaUpp) const;

        /**
         *  Finds the maximum of ARS Fourier on the given interval.
         * @param thetaOpt
         * @param thetaMin
         * @param thetaMax
         * @return 
         */
        double findMax(double& thetaMax, double thetaLow, double thetaUpp) const;

    protected:
        std::vector<double> coeffs_;
        ArsKernelIsotropic2d isotropicKer_;
        int arsfOrder_;
        double thetaToll_;
        // Parameters for computation of the Fourier coefficients of anisotropic kernels
        //int anisotropicStep_;
    };

} // end of namespace

#endif /*ARS2D_H*/