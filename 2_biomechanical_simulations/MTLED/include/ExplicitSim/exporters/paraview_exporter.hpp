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



/*!
   \file paraview_exporter.hpp
   \brief ParaviewExporter class header file.
   \author Konstantinos A. Mountris
   \date 11/10/2017
*/

#ifndef EXPLICITSIM_PARAVIEW_EXPORTER_PARAVIEW_EXPORTER_HPP_
#define EXPLICITSIM_PARAVIEW_EXPORTER_PARAVIEW_EXPORTER_HPP_

#include <boost/filesystem.hpp>
#include <boost/range.hpp>

#include <tinyxml2.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <string>
#include <cstddef>
#include <vector>
#include <map>

#include <stdexcept>
#include <exception>

#include "ExplicitSim/vectors/vectors.hpp"

#include "ExplicitSim/models/models.hpp"


namespace ExplicitSim {

/*!
 *  \addtogroup Exporters
 *  @{
 */


/*!
 * \class ParaviewExporter
 * \brief Class implemmenting the output of weak/strong models and their solution fields for post-processing in ParaView.
 *
 */

class ParaviewExporter {

public:
    /*!
     * \brief ParaviewExporter constructor.
     */
    ParaviewExporter();
    
    
    /*!
     * \brief ParaviewExporter destructor.
     */
    virtual ~ParaviewExporter();
    
    
    /*!
     * \brief  Create a ParaView (.vtu) file for the given model in xml format.
     * \param [in] weak_model_3d The 3D weak formulation model to be exported.
     * \return [void]
     */
    void CreateVtu(WeakModel3D &weak_model_3d);
    
    
    //void AddPointNormals();
    //void AddScalarField(const Eigen::VectorXd &scalar_field, const std::string &scalar_field_name);
    

    /*!
     * \brief Add a vector field in the xml tree of the ParaView (.vtu) file.
     * \param [in] vector_field The vector field given as [n x 3] matrix. n is the number of nodes of the model where the vector field is applied.
     * \param [in] vector_field_name The name of the vector field to be assigned in the corresponding branch of the Paraview (.vtu) xml file.
     * \return [void]
     */
    void AddVectorField(const Eigen::MatrixXd &vector_field, const std::string &vector_field_name);


    void ClearVectorFields();
    

    /*!
     * \brief Export the ParaView (.vtu) xml file.
     *
     * If a non-existing path is given it is generated automatically by the exporter.
     * Similarly if the file's name has not .vtu extension it is added automatically.
     *
     * \param [in] export_filename The filename of the exporting ParaView (.vtu) xml file.
     * \return [void]
     */
    void Export(const std::string &export_filename);


    /*!
     * \brief Create a ParaView animation (.pvd) xml file.
     *
     * If a non-existing path is given for the animation file it is generated automatically by the exporter.
     * Similarly if the file's name has not .pvd extension it is added automatically.
     *
     * \param [in] vtu_files_paths The vtu files to included in the animation.
     * \param [in] pvd_filename The filename of the ParaView animation (.pvd) xml file.
     * \return [void]
     */
    void CreatePvdAnimation(const std::map<float, std::string> &vtu_files_paths, const std::string &pvd_filename);
    


private:
    tinyxml2::XMLDocument output_;      /*!< The xml file that will be exported in Paraview (.vtu) format. */
    
    WeakModel3D *pModel_3d_;         /*!< The 3D weak formulation model to be represented by the Paraview (.vtu) xml file. */
};



/*! @} End of Doxygen Groups*/
} //end of namespace ExplicitSim

#endif  //EXPLICITSIM_PARAVIEW_EXPORTER_PARAVIEW_EXPORTER_HPP_
