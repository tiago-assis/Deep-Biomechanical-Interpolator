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



#ifndef EXPLICITSIM_CONDITIONS_CONDITIONS_HANDLER_HPP_
#define EXPLICITSIM_CONDITIONS_CONDITIONS_HANDLER_HPP_

/*!
   \file conditions_handler.hpp
   \brief ConditionsHandler class header file.
   \author Konstantinos A. Mountris
   \date 05/10/2017
*/


#include "ExplicitSim/elements/elements.hpp"
#include "ExplicitSim/sets/node_set.hpp"
#include "ExplicitSim/conditions/load_curve.hpp"
#include "ExplicitSim/models/weak_model_3d.hpp"
#include "ExplicitSim/support_domain/support_domain.hpp"

#include "ExplicitSim/conditions/contact_pair.hpp"
#include "ExplicitSim/approximants/mmls_3d.hpp"

#include <boost/smart_ptr.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
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
 * \class ConditionsHandler
 * \brief Class handling the imposition of the boundary conditions.
 *
 * \note Conditions are handled by reading the boundary id of the mesh.
 * Some nodes have more than one conditions and should see how to process correctly
 * under general conditions for extraction of the nodes ids.
 *
 */

class ConditionsHandler {
public:

    /*!
     * \brief The ConditionsHandler default constructor.
     */
    ConditionsHandler(uint32_t u32NumActiveNodes, const std::vector<NodeSet> *pNode_sets);


    /*!
     * \brief The ConditionsHandler default destructor.
     */
    virtual ~ConditionsHandler();


    /*!
     * \brief Add a loading condition in the imposition handler of boundary conditions.
     * \param [in] load_curve The load curve of the added loading condition for imposition.
     * \param [in] x The conditional stating if the added loading condition is applied on the x axis.
     * \param [in] y The conditional stating if the added loading condition is applied on the y axis.
     * \param [in] z The conditional stating if the added loading condition is applied on the z axis.
	 * \param [in] dx Displacement on the x axis.
	 * \param [in] dy Displacement on the y axis.
	 * \param [in] dz Displacement on the z axis.
     * \param [in] load_boundary_name The name of the boundary that the loading condition will be assigned.
     * \return [void]
     */
    void AddLoading(const std::string load_curve_name, const bool &x, const bool &y, const bool &z,
					const float dx, const float dy, const float dz,
                    const std::string &load_boundary_name);

	/*!
	* \brief Add a loading condition in the imposition handler of boundary conditions.
	* \param [in] file_name File containing the loads.
	* \return [void]
	*/
	void vReadLoadingFromFile(const std::string load_curve_name, std::string file_name, std::vector<std::pair<int32_t, int32_t> > &offsetted_nodes, const uint32_t *pu32ReindexingArray);

	void AddExternalForce(const std::string load_curve_name, const bool &x, const bool &y, const bool &z,
											const float fx, const float fy, const float fz, const std::string &external_force_boundary_name);

	void vReadExternalLoadingFromFile(const std::string load_curve_name, std::string file_name, std::vector<std::pair<int32_t, int32_t>> &external_loaded_nodes,const uint32_t *pu32ReindexingArray);
	void ExtractExternalForcesIndices(void);
	void AddGravity(const float g_x, const float g_y, const float g_z);
	void ApplyGravity();

	inline const float gx() const { return this-> pfGravity_[0];}
	inline const float gy() const { return this-> pfGravity_[1];}
	inline const float gz() const { return this-> pfGravity_[2];}

	void ApplyExternalForce(const double relTime, Eigen::MatrixXd &ExternalForces);
    /*!
     * \brief Add a dirichlet condition in the imposition handler of boundary conditions.
     * \param [in] x The conditional stating if the added dirichlet condition is applied on the x axis.
     * \param [in] y The conditional stating if the added dirichlet condition is applied on the y axis.
     * \param [in] z The conditional stating if the added dirichlet condition is applied on the z axis.
     * \param [in] dirichlet_boundary_name The name of the boundary that the dirichlet condition will be assigned.
     * \return [void]
     */
    void AddDirichlet(const bool &x, const bool &y, const bool &z, const std::string &dirichlet_boundary_name);

	void vAddContactPair(const ExplicitSim::Surface &surface, const std::string &contact_node_set, const ExplicitSim::TetraMesh &mesh);

    /*!
     * \brief Extract the indices of the nodes that boundary conditions will be applied from the given nodesets.
     * \param [in] node_sets The nodesets from where node indices for boundary conditions application will be extracted.
     * \return [void]
     */
    void vExtractConstrainedDOFs(void);

	void vApplyContacts(Eigen::MatrixXd &displacements, bool boInitialDisplacements, const Mmls3d &nodal_approximant);

    /*!
     * \brief Apply the loading conditions on the displacements matrix.
     * \param [in] time_step The timestep of which the corresponding loading conditions will be applied on the displacements matrix.
     * \param [out] displacements The displacements matrix to be processed for loading conditions application.
     * \return [void]
     * \note Maybe when load is applied in more than one directions should be divided homogeneously. Currently the same load applies to all directions.
     */
    void ApplyBoundaryConditions(const double relTime, Eigen::MatrixXd &displacements, const Mmls3d &nodal_approximant, 
				Eigen::MatrixXd &ExternalForces, double alfa);


    /*!
     * \brief Reset to zero the forces matrix rows that correspond to nodes where loading conditions are applied.
     * \param [out] forces The forces matrix where the rows corresponding to nodes that loading conditions are applied will be set to zero.
     * \return [void]
     */
    void vZeroConstrainedDOFs(Eigen::MatrixXd &forces) const;

	/*!
	* \brief Get the maximum absolute displacement configured.
	* \return [std::vector<ExplicitSim::Dirichlet>] The maximum absolute displacement configured.
	*/
	inline const float fGetMaxAbsDisplacement() const { return this->fMaxAbsDisplacement; }

	bool boLoadCurveImplemented(const std::string name) {
		return LoadCurves.i32GetLoadCurveIndex(name) >= 0;
	};

	void vComputeCorrectionMatrix(const WeakModel3D &model, const Mmls3d &nodal_mmls, bool boUseEBCIEM, bool boSimplified);

private:

	void vComputeSimplifiedVmat(Eigen::SparseMatrix<double> &VmatX, Eigen::SparseMatrix<double> &VmatY, Eigen::SparseMatrix<double> &VmatZ, const Mmls3d &nodal_mmls);
	void vComputeVmat(Eigen::SparseMatrix<double> &VmatX, Eigen::SparseMatrix<double> &VmatY, Eigen::SparseMatrix<double> &VmatZ, const WeakModel3D &model, const Mmls3d &nodal_mmls);
    
	uint32_t u32NumActiveNodes_;

	boost::shared_ptr<double[]> pdPreviousNodalPositions_;

	boost::shared_ptr<double[]> pdCurrentNodalPositions_;

	double *pdNodalCoordinates_;

	LoadCurveFactory LoadCurves;

	const std::vector<NodeSet> *pNode_sets_;

	boost::shared_ptr<float[]> pfDOFsDisplacements_;

	boost::shared_ptr<float[]> pfGravity_;

	boost::shared_ptr<int32_t[]> pi32DOFsLoadCurves_;

	boost::shared_ptr<uint8_t[]> pu8DOFsDisplaced_;

	boost::shared_ptr<uint8_t[]> pu8DOFsFixed_;

	boost::shared_ptr<uint8_t[]> pu8NodesContacts_;

	boost::shared_ptr<uint8_t[]> pu8NodesContactsActive_;
	boost::shared_ptr<uint8_t[]> pu8DOFsExternal_;
	boost::shared_ptr<float[]> pfDOFsExternalForces_ ;

	uint32_t u32NumDisplacedDOFs;

	uint32_t u32NumFixedDOFs;

	uint32_t u32NumExternalForcesDOFs;

	uint32_t u32NumEssentialDOFs;
	uint32_t u32NumEssentialDOFsX;
	uint32_t u32NumEssentialDOFsY;
	uint32_t u32NumEssentialDOFsZ;


	boost::shared_ptr<uint32_t[]> pu32DOFsDisplacedIdx_;

	boost::shared_ptr<uint32_t[]> pu32DOFsFixedIdx_;

	boost::shared_ptr<uint32_t[]> pu32DOFsExternalForcesIdx_;
	

	float fMaxAbsDisplacement;			   /*!< The maximum absolute value of the applied displacements. */

	std::vector<ContactPair> apCContactPairs_; /*!< Array of contact pairs. */

	Eigen::SparseMatrix<double> correctionMatrixX;
	Eigen::SparseMatrix<double> correctionMatrixY;
	Eigen::SparseMatrix<double> correctionMatrixZ;
	Eigen::SparseMatrix<double, Eigen::RowMajor> FImatX;
	Eigen::SparseMatrix<double, Eigen::RowMajor> FImatY;
	Eigen::SparseMatrix<double, Eigen::RowMajor> FImatZ;
	Eigen::SparseMatrix<double> MassMatrix;

	bool boUseEBCIEM_;
};




/*! @} End of Doxygen Groups*/
} //end of namespace ExplicitSim

#endif //EXPLICITSIM_CONDITIONS_CONDITIONS_HANDLER_HPP_
