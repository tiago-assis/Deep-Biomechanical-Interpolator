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

#include <assert.h>

#include "ExplicitSim/conditions/load_curve.hpp"


namespace ExplicitSim {

double LoadCurveSmooth::dValue(double dRelTime)
{
	double dValue;
	if (dRelTime < 1)
	{
		dValue = (10.*dRelTime*dRelTime*dRelTime -
			15.*dRelTime*dRelTime*dRelTime*dRelTime +
			6.*dRelTime*dRelTime*dRelTime*dRelTime*dRelTime);
	}
	else dValue = 1;
	return dValue;
}

double LoadCurveLinear::dValue(double dRelTime)
{
	double dValue;
	if (dRelTime < 1)
	{
		dValue = (10.*dRelTime*dRelTime*dRelTime -
			15.*dRelTime*dRelTime*dRelTime*dRelTime +
			6.*dRelTime*dRelTime*dRelTime*dRelTime*dRelTime);
	}
	else dValue = 1;
	return dValue;
}

int32_t LoadCurveFactory::i32GetLoadCurveIndex(const std::string name)
{
	for (uint32_t i = 0; i < u32GetNumLoadCurves(); i++)
	{
		if (capLoadCurves[i]->GetName() == name)
		{
			return i;
		}
	}
	return -1;
}

void LoadCurveFactory::vComputeValues(double dRelTime)
{
	for (uint32_t i = 0; i < u32GetNumLoadCurves(); i++)
	{
		dValues[i] = capLoadCurves[i]->dValue(dRelTime);
	}
}

} //end of namespace ExplicitSim

