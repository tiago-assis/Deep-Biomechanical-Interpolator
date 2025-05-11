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


#ifndef EXPLICITSIM_OPTIONS_CONFIGURATION_CONFIG_MANAGER_HPP_
#define EXPLICITSIM_OPTIONS_CONFIGURATION_CONFIG_MANAGER_HPP_

/*!
   \file config_manager.hpp
   \brief ConfigManager class header file.
   \author Konstantinos A. Mountris
   \date 16/11/2017
*/


#include <boost/program_options.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cstring>


/*!
 * \namespace boost_po Shortcut for the boost::program_options namespace.
 */
namespace boost_po = boost::program_options;


namespace ExplicitSim {

/*!
 *  \addtogroup ConfigurationManagement
 *  @{
 */


/*!
 * \class ConfigManager
 * \brief Class implementing management of the configuration file for
 *        ExplicitSim execution based on the program_options library of Boost C++ Libraries.
 */

class ConfigManager {

public:
    /*!
     * \brief The ConfigManager default constructor.
     */
    ConfigManager();


    /*!
     * \brief The ConfigManager constructor with string argument.
     * \param bidon
     */
    ConfigManager(std::string bidon);


    /*!
     * \brief The ConfigManager destructor.
     */
    virtual ~ConfigManager();


    /*!
     * \brief Read configuration file.
     * \param [in] config_filename The name (full path) of the configuration file.
     * \return [void]
     */
    void ReadConfigFile(const std::string &config_filename);


    /*!
     * \brief Read configuration stream.
     * \param [in] config_stream The steam from where the configuration is parsed.
     * \return [void]
     */
    void ReadConfigStream(std::istream &config_stream);


    /*!
     * \brief Print a sample configuration file.
     * \return [std::string] The sample configuration file.
     */
    std::string PrintSampleFile();


    /*!
     * \brief Retrieve an argument from the configuration options.
     * \param [in] argument The name of the argument.
     * \return [void]
     */
    template <class DATATYPE>
    inline DATATYPE RetrieveArgument(const std::string &argument);


    template <class DATATYPE>
    DATATYPE RetrieveArgumentFromList(const std::string &argument, const std::size_t &pos);


    /*!
     * \brief Get the number of options in a options' list of the configuration.
     * \param [in] list The name of the options' list.
     * \void [int] The number of options in a options' list of the configuration.
     */
    template <class DATATYPE>
    inline int OptionsNumInList(const std::string &list);


    /*!
     * \brief Get the configuration's map of variables.
     * \return [boost::program_options::variables_map] The configuration's map of variables.
     */
    inline const boost_po::variables_map & VarMap() const { return this->var_map_; }


protected:

    template <class OUTDATATYPE, class INDATATYPE>
    inline OUTDATATYPE Convert(const INDATATYPE &value);


private:

    boost_po::variables_map var_map_;

    boost_po::options_description description_;


};

/*! @} End of Doxygen Groups*/


} // End of namespace ExplicitSim

#include "ExplicitSim/options_configuration/config_manager.tpp"

#endif // EXPLICITSIM_OPTIONS_CONFIGURATION_CONFIG_MANAGER_HPP_
