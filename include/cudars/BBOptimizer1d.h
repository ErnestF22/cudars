/**
 * ARS - Angular Radon Spectrum 
 * Copyright (C) 2017 Dario Lodi Rizzini.
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
#ifndef OPTIMIZERBB1D_H
#define OPTIMIZERBB1D_H

#include "cudars/functions.h"

namespace cudars {

    // --------------------------------------------------------
    // BRANCH AND BOUND OPTIMIZER
    // --------------------------------------------------------

    /** 
     * Class OptimizerBB1D is a general interface for optimizing functions on 1D domain 
     * with Branch and Bound approach. 
     */
    class BBOptimizer1d {
    protected:

        /** 
         * Struct representing interval on domain, [xmin,xmax], and on the function values
         * given as lower and upper bounds
         */
        struct IntervalBound {
            double xmin;
            double xmax;
            double ylower;
            double yupper;
        };

        /**
         * Comparator of intervals to order interval with smaller upper bound before. 
         */
        struct UpperBoundLess {

            bool operator()(const IntervalBound& ib0, const IntervalBound& ib1) const {
                return (ib0.yupper < ib1.yupper);
            }
        };

        typedef std::priority_queue<IntervalBound, std::vector<IntervalBound>, UpperBoundLess> LeastUpperBoundFirstQueue;

    public:
        /**
         * Default constructor.
         */
        BBOptimizer1d();

        /**
         * Constructor with tolerances on domain and function value. 
         */
        BBOptimizer1d(double xtol, double ytol);

        /**
         * Default destructor.
         */
        virtual ~BBOptimizer1d();

        /** 
         * Returns the lower and upper bound of the function over the given interval [xmin, xmax].
         */
        virtual void findLU(double xmin, double xmax, double& ylower, double& yupper) = 0;

        /**
         * Sets the tolerance on domain to stop estimation when an accuracy on x is reached. 
         */
        void setXTolerance(double xtol);

        /**
         * Enables halting condition on x tolerance.
         */
        void enableXTolerance(bool xt);

        /**
         * Enables halting condition on x tolerance.
         */
        void enableYTolerance(bool yt);

        /**
         * Sets the tolerance on domain to stop estimation when an accuracy on y is reached. 
         */
        void setYTolerance(double ytol);

        /**
         * Finds the global maximum point of the function on the given interval [xmin, xmax]. 
         * If multiple maxima exists it returns the smallest point corresponding to the maximum. 
         * @param xmin minimum value of interval domain
         * @param xmax maximum value of interval domain
         * @param x the domain point corresponding to the global maximum
         * @param ylower the lower bound on the estimation of function maximum 
         * @param yupper the upper bound on the estimation of function maximum 
         */
        void findGlobalMax(double xmin, double xmax, double& x, double& ylower, double& yupper);

    protected:
        double xtol_;
        double ytol_;
        bool xtollOn_;
        bool ytollOn_;

        /**
         * Checks if the interval satisfies stop criteria. 
         * Stop criteria are based on tolerances, etc.
         */
        bool checkInterval(const IntervalBound& ib) const;
    };


    // --------------------------------------------------------
    // FOURIER SERIES OPTIMIZER 
    // --------------------------------------------------------

    /**
     * Class FourierOptimizerBB1D performs optimization on 1D functionts represented as 
     * Fourier series with finite terms. 
     * It is a specialization of OptimizerBB1D.
     */
    class FourierOptimizerBB1D : public BBOptimizer1d {
    public:
        /**
         * Default constructor.
         */
        FourierOptimizerBB1D();

        /**
         * Constructor with Fourier series coefficients.
         */
        FourierOptimizerBB1D(const std::vector<double>& coeffs);

        /**
         * Destructor.
         */
        virtual ~FourierOptimizerBB1D();

        /**
         * Sets the Fourier series coefficients. 
         */
        void setCoefficients(const std::vector<double>& coeffs);

        /**
         * Returns the lower and upper bound of the function over the given interval [xmin, xmax].
         */
        virtual void findLU(double xmin, double xmax, double& ylower, double& yupper);

    private:
        std::vector<double> coeffs_;
        int orderMax_;
    };


    /** Computes the maximum of a Fourier series function using Branch & Bound (BB) approach. 
     */
    void findGlobalMaxBBFourier(const std::vector<double>& coeffs, double theta0, double theta1, double thetaToll, double fourierToll, double& thetaMax, double& arsfMax);

} // end of namespace 

#endif /* OPTIMIZERBB1D_H */

