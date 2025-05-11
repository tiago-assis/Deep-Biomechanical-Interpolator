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

#include "ExplicitSim/support_domain/support_domain.hpp"

namespace ExplicitSim {

SupportDomain::SupportDomain()
{
#ifdef EXPLICITSIM_NEAREST_NEIGHBOR_BUCKETSEARCH
	pCBuckets = NULL;
	pCBucketSearch = NULL;
	pPointsCoord_ = NULL;
#endif
}


SupportDomain::~SupportDomain()
{
#ifdef EXPLICITSIM_NEAREST_NEIGHBOR_BUCKETSEARCH
	if (pCBuckets != NULL) delete pCBuckets;
	if (pCBucketSearch != NULL) delete pCBucketSearch;
	if (pPointsCoord_ != NULL) delete[] pPointsCoord_;
#endif
}


void SupportDomain::SetInfluenceNodes(const std::vector<Node> &nodes, uint32_t numNodes)
{
    // Clear the current influence nodes.
    this->influence_nodes_.clear();

    // Set the influence nodes.
    this->influence_nodes_ = nodes;

	u32NumInfluenceNodes = numNodes;
}


void SupportDomain::SetInfluenceTetrahedra(const std::vector<Tetrahedron> &tetras)
{
    // Clear the current influence tetrahedra.
    this->influence_tetras_.clear();

    // Set the influence tetrahedra.
    this->influence_tetras_ = tetras;
}


void SupportDomain::ComputeInfluenceNodesRadiuses(double dilatation_coeff)
{
    // Check for initialized influence nodes and elements.
    if (this->influence_nodes_.empty() || this->influence_tetras_.empty()) {
        std::string error = "[ExplicitSim ERROR] Either influence nodes or influence tetrahedra were not initialized."
                            " Can't compute infuence radius.";
        throw std::runtime_error(error.c_str());
    }

    // Clear the influence radiuses container.
    this->influence_radiuses_.clear();

    // Initialize the influence radiuses container.
    this->influence_radiuses_.assign(u32NumInfluenceNodes, 0.);

    // The number of connected nodes to each influence node.
    std::vector<uint32_t> connected_nodes_number(u32NumInfluenceNodes, 0);


    // The coordinates of the nodes of an influence element.
    Vec3<double> n1(0., 0., 0.); Vec3<double> n2(0., 0., 0.);
    Vec3<double> n3(0., 0., 0.); Vec3<double> n4(0., 0., 0.);

    // The inter-node distances.
    double d12 = 0.; double d13 = 0.; double d14 = 0.;
    double d23 = 0.; double d24 = 0.; double d34 = 0.;

    // Iterate over the influence elements.
    for (auto &tetra : this->influence_tetras_) {
        // Extract element nodes coordinates.
        n1.Set(this->influence_nodes_[tetra.N1()].Coordinates());
        n2.Set(this->influence_nodes_[tetra.N2()].Coordinates());
        n3.Set(this->influence_nodes_[tetra.N3()].Coordinates());
        n4.Set(this->influence_nodes_[tetra.N4()].Coordinates());

        // Calculate distances between element nodes.
        d12 = std::sqrt( ( (n1.X()-n2.X()) * (n1.X()-n2.X()) ) +
                         ( (n1.Y()-n2.Y()) * (n1.Y()-n2.Y()) ) +
                         ( (n1.Z()-n2.Z()) * (n1.Z()-n2.Z()) ) );

        d13 = std::sqrt( ( (n1.X()-n3.X()) * (n1.X()-n3.X()) ) +
                         ( (n1.Y()-n3.Y()) * (n1.Y()-n3.Y()) ) +
                         ( (n1.Z()-n3.Z()) * (n1.Z()-n3.Z()) ) );

        d14 = std::sqrt( ( (n1.X()-n4.X()) * (n1.X()-n4.X()) ) +
                         ( (n1.Y()-n4.Y()) * (n1.Y()-n4.Y()) ) +
                         ( (n1.Z()-n4.Z()) * (n1.Z()-n4.Z()) ) );

        d23 = std::sqrt( ( (n2.X()-n3.X()) * (n2.X()-n3.X()) ) +
                         ( (n2.Y()-n3.Y()) * (n2.Y()-n3.Y()) ) +
                         ( (n2.Z()-n3.Z()) * (n2.Z()-n3.Z()) ) );

        d24 = std::sqrt( ( (n2.X()-n4.X()) * (n2.X()-n4.X()) ) +
                         ( (n2.Y()-n4.Y()) * (n2.Y()-n4.Y()) ) +
                         ( (n2.Z()-n4.Z()) * (n2.Z()-n4.Z()) ) );

        d34 = std::sqrt( ( (n3.X()-n4.X()) * (n3.X()-n4.X()) ) +
                         ( (n3.Y()-n4.Y()) * (n3.Y()-n4.Y()) ) +
                         ( (n3.Z()-n4.Z()) * (n3.Z()-n4.Z()) ) );

        // Accumulate contribution in influence radiuses of the influence nodes of the element.
        this->influence_radiuses_[tetra.N1()] += (d12 + d13 + d14);
        this->influence_radiuses_[tetra.N2()] += (d12 + d23 + d24);
        this->influence_radiuses_[tetra.N3()] += (d13 + d23 + d34);
        this->influence_radiuses_[tetra.N4()] += (d14 + d24 + d34);

        // Accumulate number of connected nodes to the influence nodes of the element.
        connected_nodes_number[tetra.N1()] += 3;
        connected_nodes_number[tetra.N2()] += 3;
        connected_nodes_number[tetra.N3()] += 3;
        connected_nodes_number[tetra.N4()] += 3;
    }

	double dMaxInfluenceRadius = 0;
    // Normalize the influence radiuses with the number of connected nodes for each influence node.
    for (auto &radius : this->influence_radiuses_) {
        // Index of the current influence node.
        auto id = &radius - &this->influence_radiuses_[0];

		if (connected_nodes_number[id] == 0)
		{
			std::string error = "[ExplicitSim ERROR] Node not connected to any tetrahedron."
				" Can't compute infuence radius.";
			throw std::runtime_error(error.c_str());
		}
        // Normalize the influence radius of the current influence node.
        radius /= connected_nodes_number[id];
		if (radius > dMaxInfluenceRadius)
		{
			dMaxInfluenceRadius = radius;
		}
    }

    // Apply dilatation coefficient if given and is positive.
    if (dilatation_coeff > 0.) {
        std::transform(this->influence_radiuses_.begin(), this->influence_radiuses_.end(),
                       this->influence_radiuses_.begin(), std::bind1st(std::multiplies<double>(), dilatation_coeff));
		dMaxInfluenceRadius *= dilatation_coeff;
    }
#ifdef EXPLICITSIM_NEAREST_NEIGHBOR_BUCKETSEARCH
	// Find bounding box of the nodes for bucket search
	double adBoundingBox[6];
	Node &N0 = this->influence_nodes_[0];
	adBoundingBox[0] = adBoundingBox[1] = N0.Coordinates()[0];
	adBoundingBox[2] = adBoundingBox[3] = N0.Coordinates()[1];
	adBoundingBox[4] = adBoundingBox[5] = N0.Coordinates()[2];

	if (pPointsCoord_ != NULL) delete[] pPointsCoord_;
	pPointsCoord_ = new double[3 * u32NumInfluenceNodes];
	pPointsCoord_[0] = N0.Coordinates()[0];
	pPointsCoord_[1] = N0.Coordinates()[1];
	pPointsCoord_[2] = N0.Coordinates()[2];

	for (size_t i = 1; i < u32NumInfluenceNodes; i++)
	{
		Node &N = this->influence_nodes_[i];
		double Nx = N.Coordinates()[0];
		double Ny = N.Coordinates()[1];
		double Nz = N.Coordinates()[2];
		if (Nx < adBoundingBox[0]) adBoundingBox[0] = Nx;
		if (Nx > adBoundingBox[1]) adBoundingBox[1] = Nx;
		if (Ny < adBoundingBox[2]) adBoundingBox[2] = Ny;
		if (Ny > adBoundingBox[3]) adBoundingBox[3] = Ny;
		if (Nz < adBoundingBox[4]) adBoundingBox[4] = Nz;
		if (Nz > adBoundingBox[5]) adBoundingBox[5] = Nz;

		pPointsCoord_[i*3] = Nx;
		pPointsCoord_[i*3 + 1] = Ny;
		pPointsCoord_[i*3 + 2] = Nz;
	}

	// Create buckets for bucket search
	double dBucketSize = dMaxInfluenceRadius;
	unsigned int uiNumBucketsX = (unsigned int)((adBoundingBox[1] - adBoundingBox[0]) / dBucketSize) + 1;
	unsigned int uiNumBucketsY = (unsigned int)((adBoundingBox[3] - adBoundingBox[2]) / dBucketSize) + 1;
	unsigned int uiNumBucketsZ = (unsigned int)((adBoundingBox[5] - adBoundingBox[4]) / dBucketSize) + 1;

	if (pCBuckets != NULL) delete pCBuckets;
	if (pCBucketSearch != NULL) delete pCBucketSearch;

	pCBuckets = new BucketSearch::buckets(adBoundingBox[0], adBoundingBox[2], adBoundingBox[4],
		dBucketSize, dBucketSize, dBucketSize, uiNumBucketsX, uiNumBucketsY, uiNumBucketsZ);

	pCBucketSearch = new BucketSearch::BucketSearch(pCBuckets, pPointsCoord_, u32NumInfluenceNodes);
	pCBucketSearch->vDistributePointsToBuckets();
#endif
}


#ifdef EXPLICITSIM_NEAREST_NEIGHBOR_BUCKETSEARCH
const std::vector<std::vector<uint32_t> > SupportDomain::ClosestNodesIdsTo(const std::vector<Vec3<double> > &eval_nodes_coords) const
{
    //Check if influence radiuses have been initialized.
    if (this->influence_radiuses_.empty()) {
        std::string error = "ERROR: Can not find closest nodes without knowledge of influence radiuses."
                            " Compute influence radiuses first";
        throw std::runtime_error(error.c_str());
    }

	auto numEvalNodes = eval_nodes_coords.size();

    // Container with lists of closest nodes indices to the given nodes.
    std::vector<std::vector<uint32_t> > closest_nodes_ids;
	
    // Neighbors indices of a node.
    std::vector<uint32_t> neighbors_indices;

    // Evaluate nodes for neighbors.
    for (auto i = 0; i < numEvalNodes; i++)
	{
		auto eval_node = eval_nodes_coords[i];
		double adCoords[3];
		adCoords[0] = eval_node.X();
		adCoords[1] = eval_node.Y();
		adCoords[2] = eval_node.Z();
		pCBucketSearch->vFindNeigbours(adCoords, this->influence_radiuses_, neighbors_indices);
		// Store the list of neighbors
		//std::sort(neighbors_indices.begin(), neighbors_indices.end());
		closest_nodes_ids.emplace_back(neighbors_indices);
    }
    // Return the list of closest nodes to the evaluation nodes.
    return closest_nodes_ids;
}
#endif

#ifdef EXPLICITSIM_NEAREST_NEIGHBOR_EXHAUSTIVE
const std::vector<std::vector<uint32_t> > SupportDomain::ClosestNodesIdsTo(const std::vector<Vec3<double> > &eval_nodes_coords) const
{
    //Check if influence radiuses have been initialized.
    if (this->influence_radiuses_.empty()) {
        std::string error = "ERROR: Can not find closest nodes without knowledge of influence radiuses."
                            " Compute influence radiuses first";
        throw std::runtime_error(error.c_str());
    }

    // Container with lists of closest nodes indices to the given nodes.
    std::vector<std::vector<uint32_t> > closest_nodes_ids;

    // Neighbors indices of a node.
    std::vector<uint32_t> neighbors_indices;

    // Squared distance of neighbor from the node.
    Vec3<double> squared_dist(0., 0., 0.);

    // Evaluate nodes for neighbors.
    for (auto &eval_node : eval_nodes_coords) {

        // Exhaustive check over all influence nodes.
        for (auto &neighbor : this->influence_nodes_) {
			uint32_t neighbor_id = &neighbor - &this->influence_nodes_[0];

            // Set squared distance between node and neighbor.
            squared_dist.SetX((neighbor.Coordinates().X() - eval_node.X()) *
                                  (neighbor.Coordinates().X() - eval_node.X()));

            squared_dist.SetY((neighbor.Coordinates().Y() - eval_node.Y()) *
                                  (neighbor.Coordinates().Y() - eval_node.Y()));

            squared_dist.SetZ((neighbor.Coordinates().Z() - eval_node.Z()) *
                                  (neighbor.Coordinates().Z() - eval_node.Z()));

            // Neighbor will be closest node if its squared distance is smaller than the squared influence radius.
            if ((squared_dist.X() + squared_dist.Y() + squared_dist.Z()) <
                    (this->influence_radiuses_[neighbor_id]*this->influence_radiuses_[neighbor_id]) ) {
                neighbors_indices.push_back(neighbor_id);
            }
        }
        // Store the list of neighbors.
        closest_nodes_ids.push_back(neighbors_indices);
        // Clear neighbors container to prepare for next evaluation node.
        neighbors_indices.clear();
    }

    // Return the list of closest nodes to the evaluation nodes.
    return closest_nodes_ids;
}
#endif // EXPLICITSIM_NEAREST_NEIGHBOR_EXHAUSTIVE

#ifdef EXPLICITSIM_NEAREST_NEIGHBOR_CGAL
const std::vector<std::vector<uint32_t> > SupportDomain::ClosestNodesIdsTo(const std::vector<Vec3<double> > &eval_nodes_coords) const
{
    typedef CGAL::Simple_cartesian<double> K;
    typedef K::Point_3 Point_3d;
    typedef boost::tuple<Point_3d,unsigned int> Point_and_int;
    typedef CGAL::Search_traits_3<K> Traits_base;
    typedef CGAL::Search_traits_adapter<Point_and_int,
    CGAL::Nth_of_tuple_property_map<0, Point_and_int>, Traits_base> Traits;
    typedef CGAL::Kd_tree<Traits> Tree;
    typedef CGAL::Fuzzy_sphere<Traits> Fuzzy_sphere;


    //Check if influence radiuses have been initialized.
    if (this->influence_radiuses_.empty()) {
        std::string error = "[ExplicitSim ERROR] Can not find closest nodes without knowing the influence radiuses."
                            " Compute influence radiuses first";
        throw std::runtime_error(error.c_str());
    }

    // Container with lists of closest nodes indices to the given nodes.
    std::vector<std::vector<uint32_t> > closest_nodes_ids;
    closest_nodes_ids.resize(eval_nodes_coords.size());

    //Vectors to store coordinates and ids of point to be passed in tree tuples.
    std::vector<Point_3d> points;
    std::vector<int> id;

    for(auto &eval_node : eval_nodes_coords) {
        auto idx = &eval_node - &eval_nodes_coords[0];

        points.push_back(Point_3d(eval_node.X(), eval_node.Y(), eval_node.Z()) );
        id.push_back(idx);
    }

    //Insert <point,id> tuples in the searching tree.
    Tree tree(boost::make_zip_iterator(boost::make_tuple(points.begin(), id.begin() )),
              boost::make_zip_iterator(boost::make_tuple(points.end(), id.end() )) );

    //Vector containing the domain nodes for a given grid node.
    std::vector<Point_and_int> domain_nodes;

    //Search domain_nodes for each grid node.
	for (uint32_t i = 0; i < u32NumInfluenceNodes; i++) {
		Node node = influence_nodes_[i];

        Point_3d center(node.Coordinates().X(), node.Coordinates().Y(), node.Coordinates().Z());

        // Searching sphere.
        Fuzzy_sphere fs(center, this->influence_radiuses_[i]);

        //Neighbors search
        tree.search( std::back_inserter(domain_nodes), fs);

        //Iterate over domain nodes.
        for (auto &domain_node : domain_nodes) {
            uint32_t dn_id = (uint32_t)(&domain_node - &domain_nodes[0]);
            //Store domain nodes indices.
            closest_nodes_ids[boost::get<1>(domain_nodes[dn_id])].emplace_back(i);
        }

        // Empty domain_nodes vector for next iteration.
        domain_nodes.clear();
    }

    // Return the indices of the closest nodes to each evaluation point.
    return closest_nodes_ids;

}
#endif // EXPLICITSIM_NEAREST_NEIGHBOR_CGAL

int SupportDomain::MinSupportNodesIn(const std::vector<std::vector<uint32_t> > &neighbor_ids) const
{
    // Initialize minimum number of support nodes.
    int min_support_nodes_ = std::numeric_limits<int>::max();

    // Compute the minimum number of support nodes.
    for (auto &neighs : neighbor_ids) {
        if (static_cast<int>(neighs.size()) < min_support_nodes_) {
            min_support_nodes_ = static_cast<int>(neighs.size());
        }
    }

    return min_support_nodes_;
}


int SupportDomain::MaxSupportNodesIn(const std::vector<std::vector<uint32_t> > &neighbor_ids) const
{
    // Initialize maximum number of support nodes.
    int max_support_nodes_ = 0;

    // Compute the maximum number of support nodes.
    for (auto &neighs : neighbor_ids) {
        if (static_cast<int>(neighs.size()) > max_support_nodes_) {
            max_support_nodes_ = static_cast<int>(neighs.size());
        }
    }

    return max_support_nodes_;
}


} //end of namespace ExplicitSim
