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



#ifndef EXPLICITSIM_CONDITIONS_LOAD_CURVE_HPP_
#define EXPLICITSIM_CONDITIONS_LOAD_CURVE_HPP_

/*!
   \file load_curve.hpp
   \brief LoadCurve class header file.
   \author Konstantinos A. Mountris
   \date 30/09/2017
*/


#include <cmath>
#include <string>
#include <vector>

#include <stdexcept>
#include <exception>

namespace ExplicitSim {

/*!
 *  \addtogroup Conditions
 *  @{
 */


/*!
 * \class LoadCurve
 * \brief Class implemmenting a load curve to be used in load condition.
 *
 * The load curve applies the loading progressively, in load time steps, to maintain stability of the solution.
 *
 */

class LoadCurve {
public:
	LoadCurve() {};
	virtual ~LoadCurve() {};
	virtual double dValue(double dRelTime) = 0;
	virtual const std::string &GetName(void) = 0;
};

class LoadCurveSmooth : public LoadCurve {
public:
	virtual double dValue(double dRelTime);
	virtual const std::string &GetName(void) { return name; };
private:
	const std::string name = "smooth";
};

class LoadCurveLinear : public LoadCurve {
public:
	virtual double dValue(double dRelTime);
	virtual const std::string &GetName(void) { return name; };
private:
	const std::string name = "linear";
};

class LoadCurveFactory {
public:

#define LoadCurveFactory_NUM_LOAD_CURVES 2

	LoadCurveFactory() {};
	~LoadCurveFactory() {};

	int32_t i32GetLoadCurveIndex(const std::string name);
	double dValue(uint32_t u32Index) { return dValues[u32Index]; };
	void vComputeValues(double dRelTime);
	uint32_t u32GetNumLoadCurves(void) {
		return LoadCurveFactory_NUM_LOAD_CURVES;
	};

private:
	LoadCurveSmooth lcSmooth;
	LoadCurveLinear lcLinear;
	LoadCurve *capLoadCurves[LoadCurveFactory_NUM_LOAD_CURVES] = { &lcSmooth,  &lcLinear};
	double dValues[LoadCurveFactory_NUM_LOAD_CURVES];
};



/*! @} End of Doxygen Groups*/
} //end of namespace ExplicitSim

#endif //EXPLICITSIM_CONDITIONS_LOAD_CURVE_HPP_
