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


#include "ExplicitSim/solvers/dyn_relax_prop.hpp"


namespace ExplicitSim {

DynRelaxProp::DynRelaxProp() : equilibrium_time_(0.), equilibrium_steps_num_(0), load_conv_rate_(0.), after_load_conv_rate_(0.),
                               conv_rate_deviation_(0.), stop_update_conv_rate_steps_num_(0), force_disp_update_steps_num_(0),
                               stable_conv_rate_steps_num_(0), conv_rate_stop_deviation_(0.), stop_conv_rate_error_(0.),
                               stop_abs_error_(0.), stop_steps_num_(0)
{}


DynRelaxProp::~DynRelaxProp()
{}


void DynRelaxProp::ComputeStepsNum(const double &solver_time_step)
{
    // Compute the number of time steps for dynamic relaxation equilibrium.
    this->equilibrium_steps_num_ = static_cast<int>(std::floor((this->equilibrium_time_ / solver_time_step) + 0.5));
	this->load_steps_num_ = static_cast<int>(std::floor((this->load_time_ / solver_time_step) + 0.5));
}


bool DynRelaxProp::IsInitialized() const
{
    // Check if the dynamic relaxation properties are initialized.
    if ((this->equilibrium_time_ == 0.) || (this->equilibrium_steps_num_ == 0) ||
        (this->load_conv_rate_ == 0.) || (this->after_load_conv_rate_ == 0.) ||
        (this->conv_rate_deviation_ == 0.) || (this->stop_update_conv_rate_steps_num_ == 0) ||
        (this->force_disp_update_steps_num_ == 0) || (this->stable_conv_rate_steps_num_ == 0) ||
        (this->conv_rate_stop_deviation_ == 0.) || (this->stop_conv_rate_error_ == 0.) ||
            (this->stop_abs_error_ == 0.) || (this->stop_steps_num_ == 0) ) {
        return false;
    }
    else { return true; }
}



} //end of namespace ExplicitSim
