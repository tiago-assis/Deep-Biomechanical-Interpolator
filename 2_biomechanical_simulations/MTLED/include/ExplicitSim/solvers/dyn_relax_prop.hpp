/*
 * ExplicitSim - Software for solving PDEs using explicit methods.
 * Copyright (C) 2017  <Konstantinos A. Mountris> <konstantinos.mountris@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors (alphabetically):
 *      George C. BOURANTAS
 *      Grand R. JOLDES
 *      Konstantinos A. MOUNTRIS
 */



#ifndef EXPLICITSIM_SOLVERS_DYN_RELAX_PROP_HPP_
#define EXPLICITSIM_SOLVERS_DYN_RELAX_PROP_HPP_

/*!
   \file dyn_relax_prop.hpp
   \brief DynRelaxProp class header file.
   \author Konstantinos A. Mountris
   \date 30/09/2017
*/

#include <cmath>

#include <string>
#include <stdexcept>
#include <exception>

namespace ExplicitSim {

/*!
 *  \addtogroup Solvers
 *  @{
 */


/*!
 * \class DynRelaxProp
 * \brief Class implemmenting the dynamic relaxation properties for dynamic relaxation application in the MTLED solver.
 *
 */

class DynRelaxProp {
public:

    /*!
     * \brief The DynRelaxProp constructor.
     */
    DynRelaxProp();


    /*!
     * \brief The DynRelaxProp destructor.
     */
    virtual ~DynRelaxProp();


    /*!
     * \brief Set the equilibrium time of the dynamic relaxation.
     * \param [in] equilibrium_time The equilibrium time of the dynamic relaxation.
     * \return [void]
     */
    inline void SetEquilibriumTime(const double &equilibrium_time) { this->equilibrium_time_ = equilibrium_time; }

	inline void SetLoadTime(const double &load_time) { this->load_time_ = load_time; }


    /*!
     * \brief Compute the number of time steps required to achieve dynamic relaxation equilibrium.
     * \param [in] solver_time_step The time step of the explicit solver for stable solution.
     * \return [void]
     */
    void ComputeStepsNum(const double &solver_time_step);


    /*!
     * \brief Set the convergence rate during loading application.
     * \param [in] load_conv_rate The convergence rate during loading application.
     * \return [void]
     */
    inline void SetLoadConvRate(const double &load_conv_rate) { this->load_conv_rate_ = load_conv_rate; }


    /*!
     * \brief Set the convergence rate after loading application.
     * \param [in] after_load_conv_rate The convergence rate after loading application.
     * \return [void]
     */
    inline void SetAfterLoadConvRate(const double &after_load_conv_rate) { this->after_load_conv_rate_ = after_load_conv_rate; }


    /*!
     * \brief Set the convergence rate deviation, used for convergence rate stability check.
     * \param [in] conv_rate_deviation The convergence rate deviation, used for convergence rate stability check.
     * \return [void]
     */
    inline void SetConvRateDeviation(const double &conv_rate_deviation) { this->conv_rate_deviation_ = conv_rate_deviation; }


    /*!
     * \brief Set the number of dynamic relaxation steps after which the convergence rate is no longer updated.
     * \param [in] stop_update_conv_rate_steps_num The number of dynamic relaxation steps after which the convergence rate is no longer updated.
     * \return [void]
     */
    inline void SetStopUpdateConvRateStepsNum(const int &stop_update_conv_rate_steps_num) { this->stop_update_conv_rate_steps_num_ = stop_update_conv_rate_steps_num; }


    /*!
     * \brief Set the number of dynamic relaxation steps after which the force and displacement fields are updated.
     * \param [in] force_disp_update_steps_num The number of dynamic relaxation steps after which the force and displacement fields are updated.
     * \return [void]
     */
    inline void SetForceDispUpdateStepsNum(const int &force_disp_update_steps_num) { this->force_disp_update_steps_num_ = force_disp_update_steps_num; }


    /*!
     * \brief Set the number of dynamic relaxation steps after which the convergence rate is considered stable.
     * \param [in] stable_conv_rate_steps_num The number of dynamic relaxation steps after which the convergence rate is considered stable.
     * \return [void]
     */
    inline void SetStableConvRateStepsNum(const int &stable_conv_rate_steps_num) { this->stable_conv_rate_steps_num_ = stable_conv_rate_steps_num; }


    /*!
     * \brief Set the stopping deviation that sets when the convergence rate is stabilized.
     * \param [in] conv_rate_stop_deviation The stopping deviation that sets when the convergence rate is stabilized.
     * \return [void]
     */
    inline void SetConvRateStopDeviation(const double &conv_rate_stop_deviation) { this->conv_rate_stop_deviation_ = conv_rate_stop_deviation; }


    /*!
     * \brief Set the estimated error during convergence rate computation.
     * \param [in] stop_conv_rate_error The estimated error during convergence rate computation.
     * \return [void]
     */
    inline void SetStopConvRateError(const double &stop_conv_rate_error) { this->stop_conv_rate_error_ = stop_conv_rate_error;}


    /*!
     * \brief Set the accepted absolute error value for dynamic relaxation termination.
     * \param [in] stop_abs_error The accepted absolute error value for dynamic relaxation termination.
     * \return [void]
     */
    inline void SetStopAbsError(const double &stop_abs_error) { this->stop_abs_error_ = stop_abs_error; }


    /*!
     * \brief Set the number of consecutive steps that stopping criteria must be satisfied before stopping the simulation.
     * \param [in] stop_steps_num The number of consecutive steps that stopping criteria must be satisfied before stopping the simulation.
     * \return [void]
     */
    inline void SetStopStepsNum(const int &stop_steps_num) { this->stop_steps_num_ = stop_steps_num; }


    /*!
     * \brief Check if dynamic relaxation properties are initialized.
     * \return [bool] The conditional giving the state of the dynamic relaxation properties initialization.
     */
    bool IsInitialized() const;


    /*!
     * \brief Get the equilibrium time of the dynamic relaxation.
     * \return [double] The equilibrium time of the dynamic relaxation.
     */
    inline const double & EquilibriumTime() const { return this->equilibrium_time_; }

	inline const double & LoadTime() const { return this->load_time_; }


    /*!
     * \brief Get the number of time steps required to achieve dynamic relaxation equilibrium.
     * \return [int] The number of time steps required to achieve dynamic relaxation equilibrium.
     */
    inline const int & EquilibriumStepsNum() const { return this->equilibrium_steps_num_; }

	inline const int & LoadStepsNum() const { return this->load_steps_num_; }


    /*!
     * \brief Get the convergence rate during loading application.
     * \return [double] The convergence rate during loading application.
     */
    inline const double & LoadConvRate() const { return this->load_conv_rate_; }


    /*!
     * \brief Get the convergence rate after loading application.
     * \return [double] The convergence rate after loading application.
     */
    inline const double & AfterLoadConvRate() const { return this->after_load_conv_rate_; }


    /*!
     * \brief Get the convergence rate deviation, used for convergence rate stability check.
     * \return [double] The convergence rate deviation, used for convergence rate stability check.
     */
    inline const double & ConvRateDeviation() const { return this->conv_rate_deviation_; }


    /*!
     * \brief Get the number of dynamic relaxation steps after which the convergence rate is no longer updated.
     * \return [int] The number of dynamic relaxation steps after which the convergence rate is no longer updated.
     */
    inline const int & StopUpdateConvRateStepsNum() const { return this->stop_update_conv_rate_steps_num_; }


    /*!
     * \brief Get the number of dynamic relaxation steps after which the force and displacement fields are updated.
     * \return [int] The number of dynamic relaxation steps after which the force and displacement fields are updated.
     */
    inline const int & ForceDispUpdateStepsNum() const { return this->force_disp_update_steps_num_; }


    /*!
     * \brief Get the number of dynamic relaxation steps after which the convergence rate is considered stable.
     * \return [int] The number of dynamic relaxation steps after which the convergence rate is considered stable.
     */
    inline const int & StableConvRateStepsNum() const { return this->stable_conv_rate_steps_num_; }


    /*!
     * \brief Get the stopping deviation that sets when the convergence rate is stabilized.
     * \return [double] The stopping deviation that sets when the convergence rate is stabilized.
     */
    inline const double & ConvRateStopDeviation() const { return this->conv_rate_stop_deviation_; }


    /*!
     * \brief Get the estimated error during convergence rate computation.
     * \return [double] The estimated error during convergence rate computation.
     */
    inline const double & StopConvRateError() const { return this->stop_conv_rate_error_;}


    /*!
     * \brief Get the accepted absolute error value for dynamic relaxation termination.
     * \return [double] The accepted absolute error value for dynamic relaxation termination.
     */
    inline const double & StopAbsError() const { return this->stop_abs_error_; }


    /*!
     * \brief Get the number of consecutive steps that stopping criteria must be satisfied before stopping the simulation.
     * \return [int] The number of consecutive steps that stopping criteria must be satisfied before stopping the simulation.
     */
    inline const int & StopStepsNum() const { return this->stop_steps_num_; }


private:
	double load_time_;

	int load_steps_num_;

    double equilibrium_time_;               /*!< The dynamic relaxation equilibrium time. */

    int equilibrium_steps_num_;             /*!< The number of time steps required to achieve dynamic relaxation equilibrium. */

    double load_conv_rate_;                 /*!< The convergence rate during loading application.*/

    double after_load_conv_rate_;           /*!< The convergence rate after loading application. */

    double conv_rate_deviation_;            /*!< The convergence rate deviation, used for convergence rate stability check. */

    int stop_update_conv_rate_steps_num_;   /*!< The number of dynamic relaxation steps after which the convergence rate is no longer updated. */

    int force_disp_update_steps_num_;       /*!< The number of dynamic relaxation steps after which the force and displacement fields are updated. */

    int stable_conv_rate_steps_num_;        /*!< The number of dynamic relaxation steps after which the convergence rate is considered stable. */

    double conv_rate_stop_deviation_;       /*!< The stopping criterion that sets when the convergence rate is stabilized. */

    double stop_conv_rate_error_;           /*!< The estimated error during convergence rate computation. */

    double stop_abs_error_;                 /*!< The accepted absolute error value for dynamic relaxation termination. */

    int stop_steps_num_;                    /*!< The number of consecutive steps that stopping criteria must be satisfied before stopping the simulation. */

};




/*! @} End of Doxygen Groups*/
} //end of namespace ExplicitSim

#endif //EXPLICITSIM_SOLVERS_DYN_RELAX_PROP_HPP_


