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


#ifndef EXPLICITSIM_OPTIONS_CONFIGURATION_CONFIG_MANAGER_TPP_
#define EXPLICITSIM_OPTIONS_CONFIGURATION_CONFIG_MANAGER_TPP_

#include "ExplicitSim/options_configuration/config_manager.hpp"


namespace ExplicitSim {

template <class DATATYPE>
DATATYPE ConfigManager::RetrieveArgument(const std::string &argument)
{
    if (!var_map_.count(argument)) {
        std::string error = "[ExplicitSim ERROR] Cannot retrieve configuration argument: " + argument;
        throw std::invalid_argument(error.c_str());
    }

    //cout << "Retrieving " << arg << " " << var_map_[ arg ].as<Type>() << flush;
    DATATYPE res = var_map_[argument].as<DATATYPE>();
    //cout<< "returning " << res << endl;
    return res;
}


template <class DATATYPE>
DATATYPE ConfigManager::RetrieveArgumentFromList(const std::string &argument, const std::size_t &pos)
{
    if (!var_map_.count(argument)) {
        std::string error = "[ExplicitSim ERROR] Cannot retrieve configuration argument: " + argument;
        throw std::invalid_argument(error.c_str());
    }

    //cout << "Retrieving " << arg << " " << var_map_[ arg ].as<Type>() << flush;
    DATATYPE res = var_map_[argument].as<std::vector<DATATYPE> >()[pos];
    //cout<< "returning " << res << endl;
    return res;
}


template <class DATATYPE>
int ConfigManager::OptionsNumInList(const std::string &list)
{
	if (!var_map_.count(list)) 
	{
        return 0;
    }
    return static_cast<int>(this->var_map_[list].as<std::vector<DATATYPE> >().size());
}


template <class OUTDATATYPE, class INDATATYPE>
OUTDATATYPE ConfigManager::Convert( const INDATATYPE &value)
{
  std::stringstream stream;
  stream << value; // insert value to stream
  OUTDATATYPE result; // store conversion's result here
  stream >> result; // write value to result
  return result;
}


} // End of namespace ExplicitSim

#endif //EXPLICITSIM_OPTIONS_CONFIGURATION_CONFIG_MANAGER_TPP_
