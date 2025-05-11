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


#include "ExplicitSim/exporters/paraview_exporter.hpp"


namespace ExplicitSim {

ParaviewExporter::ParaviewExporter()
{}


ParaviewExporter::~ParaviewExporter()
{}


void ParaviewExporter::CreateVtu(WeakModel3D &weak_model_3d)
{
    // Store in the exporter the 3D weak formulation model for for output.
    this->pModel_3d_ = &weak_model_3d;
    
    // Create xml declaration for the document.
    tinyxml2::XMLDeclaration *declaration = this->output_.NewDeclaration();
    this->output_.InsertFirstChild(declaration);
    
    // Create the main xml branch.
    tinyxml2::XMLElement *vtk_file = this->output_.NewElement("VTKFile");
    vtk_file->SetAttribute("type", "UnstructuredGrid");
    vtk_file->SetAttribute("version", "0.1");
    vtk_file->SetAttribute("byte_order", "LittleEndian");
    this->output_.InsertEndChild(vtk_file);
    
    // Create UnstructuredGrid branch.
    tinyxml2::XMLElement *unstruct_grid = this->output_.NewElement("UnstructuredGrid");
    vtk_file->InsertEndChild(unstruct_grid);

    // Create Piece branch.
    tinyxml2::XMLElement *piece = this->output_.NewElement("Piece");
    piece->SetAttribute("NumberOfPoints", this->pModel_3d_->TetrahedralMesh().NodesNum());
    piece->SetAttribute("NumberOfCells", this->pModel_3d_->TetrahedraMeshElemsNum());
    unstruct_grid->InsertEndChild(piece);
    
    // Create Points branch.
    tinyxml2::XMLElement *points = this->output_.NewElement("Points");
    piece->InsertEndChild(points);
    
    // Create DataArray branch for points coordinates.
    tinyxml2::XMLElement *points_coords = this->output_.NewElement("DataArray");
    points_coords->SetAttribute("type", "Float64");
    points_coords->SetAttribute("Name", "Points");
    points_coords->SetAttribute("NumberOfComponents", 3);
    points_coords->SetAttribute("format", "ascii");
    
    // Generate string with the nodes coordinates.
    std::string coordinates = "\n\t\t\t\t\t\t\t\t\t";
	for (uint32_t i = 0; i < this->pModel_3d_->TetrahedralMesh().NodesNum(); i++)
	{	
		Node node = this->pModel_3d_->TetrahedralMesh().Nodes()[i];
        coordinates += std::to_string(node.Coordinates().X());
        coordinates += " ";
        coordinates += std::to_string(node.Coordinates().Y());
        coordinates += " ";
        coordinates += std::to_string(node.Coordinates().Z());
        coordinates += "\n\t\t\t\t\t\t\t\t\t";
    }
    
    // Write the nodes coordinates and insert the branch in the xml tree.
    points_coords->SetText(coordinates.c_str());
    points->InsertEndChild(points_coords);
    
    // Create Cells branch.
    tinyxml2::XMLElement *cells = this->output_.NewElement("Cells");
    piece->InsertEndChild(cells);
    
    // Create Cells connectivity, offsets, and type DataArray branches.
    tinyxml2::XMLElement *cells_connectivity = this->output_.NewElement("DataArray");
    cells_connectivity->SetAttribute("type", "Int32");
    cells_connectivity->SetAttribute("Name", "connectivity");
    cells_connectivity->SetAttribute("format", "ascii");
    
    tinyxml2::XMLElement *cells_offsets = this->output_.NewElement("DataArray");
    cells_offsets->SetAttribute("type", "Int32");
    cells_offsets->SetAttribute("Name", "offsets");
    cells_offsets->SetAttribute("format", "ascii");
    
    tinyxml2::XMLElement *cells_types = this->output_.NewElement("DataArray");
    cells_types->SetAttribute("type", "UInt8");
    cells_types->SetAttribute("Name", "types");
    cells_types->SetAttribute("format", "ascii");

    // Generate strings with the connectivity, offsets, and types of the cells.
    std::string connectivity = "\n\t\t\t\t\t\t\t\t\t";
    std::string offsets = "\n\t\t\t\t\t\t\t\t\t";
    std::string types = "\n\t\t\t\t\t\t\t\t\t";
    
    for (auto &elem : this->pModel_3d_->TetrahedralMesh().Elements()) {

        // The id of the current element.
        auto id = &elem - &this->pModel_3d_->TetrahedralMesh().Elements()[0];
        
        connectivity += std::to_string(elem.N1()) + " " + std::to_string(elem.N2()) + " ";
        connectivity += std::to_string(elem.N3()) + " " + std::to_string(elem.N4()) + " ";
        offsets += std::to_string(4*id+4) + " ";
        types += std::to_string(10) + " ";
        
        // Add a new line every 10 elements.
        if ((id !=0) && !(id % 10)) {
            connectivity += "\n\t\t\t\t\t\t\t\t\t";
            offsets += "\n\t\t\t\t\t\t\t\t\t";
            types += "\n\t\t\t\t\t\t\t\t\t";
        }
    }

    // Add a new line in the end of the string if has not been added during the loop.
    if (!(this->pModel_3d_->TetrahedraMeshElemsNum() % 5)) {
        connectivity += "\n\t\t\t\t\t\t\t\t\t";
        offsets += "\n\t\t\t\t\t\t\t\t\t";
        types += "\n\t\t\t\t\t\t\t\t\t";
    }
    
    //Add attributes to connectivity, offsets, and types branches.
    cells_connectivity->SetText(connectivity.c_str());
    cells_offsets->SetText(offsets.c_str());
    cells_types->SetText(types.c_str());
    
    //Assign connectivity, offsets, and types branches.
    cells->InsertEndChild(cells_connectivity);
    cells->InsertEndChild(cells_offsets);
    cells->InsertEndChild(cells_types);

}


//void ParaviewExporter::AddPointNormals()
//{
//    //Get the first node of the vtu file.
//    tinyxml2::XMLNode *parent_node = this->vtu_doc_.FirstChildElement("VTKFile");
//    if (parent_node == nullptr) {
//        std::string error = "Error: Unknown XML file given. First node name expected to be VTKFile ...";
//        throw std::runtime_error(error.c_str());
//    }
    
    
//    //Find the branch with name Piece.
//    tinyxml2::XMLElement *piece = parent_node->FirstChildElement("UnstructuredGrid")->FirstChildElement("Piece");
//    if (piece == nullptr) {
//        std::string error = "Error: Searched for XML node named 'Piece' failed. Expected hierarchy is: VTKFile->UnstructuredGrid->Piece ...";
//        throw std::runtime_error(error.c_str());
//    }

    
//    //Find the branch with name PointData.
//    tinyxml2::XMLElement *point_data = piece->FirstChildElement("PointData");
//    if (point_data == nullptr) {
//        std::string error = "Error: Searched for XML node named 'PointData' failed. Expected to be child of node 'Piece' ...";
//        throw std::runtime_error(error.c_str());
//    }
    
//    //Set normals attribute in point_data branch.
//    point_data->SetAttribute("Normals", "normals");
    
//    //Create point normals branch.
//    tinyxml2::XMLElement *point_normals = this->vtu_doc_.NewElement("DataArray");
//    point_normals->SetAttribute("type", "Float32");
//    point_normals->SetAttribute("Name", "normals");
//    point_normals->SetAttribute("NumberOfComponents", 3);
//    point_normals->SetAttribute("format", "ascii");
    
//    //String to write point normals information
//    std::string point_normals_text = "\n\t\t\t\t\t\t\t\t\t";
//    for (int i = 0; i != this->grid_.GetNumNodes(); ++i) {
//        point_normals_text += std::to_string(this->grid_.Nodes().at(i).Nx()) + " ";
//        point_normals_text += std::to_string(this->grid_.Nodes().at(i).Ny()) + " ";
//        point_normals_text += std::to_string(this->grid_.Nodes().at(i).Nz());
//        point_normals_text += "\n\t\t\t\t\t\t\t\t\t";
//    }
    
//    //Assign point normals text to the point normals branch.
//    point_normals->SetText(point_normals_text.c_str());

//    //Assign point normals branch as child of point data branch.
//    point_data->InsertEndChild(point_normals);

//}


//void ParaviewExporter::AddScalarField(const Eigen::VectorXd& scalar_field, const std::string &scalar_field_name)
//{
//    //Get the first node of the vtu file.
//    tinyxml2::XMLNode *parent_node = this->vtu_doc_.FirstChildElement("VTKFile");
//    if (parent_node == nullptr) {
//        std::string error = "Error: Unknown XML file given. First node name expected to be VTKFile ...";
//        throw std::runtime_error(error.c_str());
//    }
    
//    //Find the branch with name Piece.
//    tinyxml2::XMLElement *piece = parent_node->FirstChildElement("UnstructuredGrid")->FirstChildElement("Piece");
//    if (piece == nullptr) {
//        std::string error = "Error: Searched for XML node named 'Piece' failed. Expected hierarchy is: VTKFile->UnstructuredGrid->Piece ...";
//        throw std::runtime_error(error.c_str());
//    }

//    //Find the branch with name PointData.
//    tinyxml2::XMLElement *point_data = piece->FirstChildElement("PointData");
//    if (point_data == nullptr) {
//        std::string error = "Error: Searched for XML node named 'PointData' failed. Expected to be child of node 'Piece' ...";
//        throw std::runtime_error(error.c_str());
//    }
    
//    //Set point scalar field attribute.
//    point_data->SetAttribute("Scalars", field_name.c_str());
    
//    //Create scalar field branch.
//    tinyxml2::XMLElement *point_scalars = this->vtu_doc_.NewElement("DataArray");
//    point_scalars->SetAttribute("type", "Float32");
//    point_scalars->SetAttribute("Name", field_name.c_str());
//    point_scalars->SetAttribute("NumberOfComponents", 1);
//    point_scalars->SetAttribute("format", "ascii");
    
//    //Generate string with the point scalar values of the field.
//    std::string point_scalars_text = "\n\t\t\t\t\t\t\t\t\t";
//    for (int i = 0; i != this->grid_.GetNumNodes(); ++i) {
//        point_scalars_text += std::to_string(scalar_field(i));
//        point_scalars_text += "\n\t\t\t\t\t\t\t\t\t";
//    }
    
//    //Assign point scalar values on the point scalar field branch.
//    point_scalars->SetText(point_scalars_text.c_str());
    
//    //Assign point scalar field as child of the point_data branch.
//    point_data->InsertEndChild(point_scalars);
//}


void ParaviewExporter::AddVectorField(const Eigen::MatrixXd &vector_field, const std::string &vector_field_name)
{
    // Get the parent branch of the exporting xml file.
    tinyxml2::XMLNode *parent_branch = this->output_.FirstChildElement("VTKFile");

    // Get the Piece branch.
    tinyxml2::XMLElement *piece = parent_branch->FirstChildElement("UnstructuredGrid")->FirstChildElement("Piece");

    // Get the PointData branch.
    tinyxml2::XMLElement *point_data = piece->FirstChildElement("PointData");

    // Create PointData branch if does not exist.
    if (point_data == nullptr) {
        point_data = this->output_.NewElement("PointData");
        piece->InsertEndChild(point_data);
    }
    
    // Add vector field attribute.
    if (point_data->Attribute("Vectors") != nullptr) {
        // Update existing attribute.
        std::string old_attribute = point_data->FirstAttribute()->Value();
        std::string new_attribute = old_attribute + " " + vector_field_name;
        point_data->SetAttribute("Vectors", new_attribute.c_str());
    }
    else {
        point_data->SetAttribute("Vectors", vector_field_name.c_str());
    }
    
    // Create vector field branch.
    tinyxml2::XMLElement *point_vectors = this->output_.NewElement("DataArray");
    point_vectors->SetAttribute("type", "Float64");
    point_vectors->SetAttribute("Name", vector_field_name.c_str());
    point_vectors->SetAttribute("NumberOfComponents", 3);
    point_vectors->SetAttribute("format", "ascii");
    
    //Generate string with the values of the vector field.
    std::string vectors = "\n\t\t\t\t\t\t\t\t\t";
    for (int i = 0; i != vector_field.rows(); ++i) {
        vectors += std::to_string(vector_field.coeff(i, 0)) + " ";
        vectors += std::to_string(vector_field.coeff(i, 1)) + " ";
        vectors += std::to_string(vector_field.coeff(i, 2));
        vectors += "\n\t\t\t\t\t\t\t\t\t";
    }
    
    //Assign vector values and insert vector field branch in the xml tree.
    point_vectors->SetText(vectors.c_str());
    point_data->InsertEndChild(point_vectors);
    

    /////////////NOTE////////////////////
    //    //cell_data should be created too...
}


void ParaviewExporter::ClearVectorFields()
{
    // Get the PointData branch.
    tinyxml2::XMLElement *point_data = this->output_.FirstChildElement("VTKFile")->FirstChildElement("UnstructuredGrid")->
                                        FirstChildElement("Piece")->FirstChildElement("PointData");

    //Delete PointData branch if it exists.
    if (point_data != nullptr) {
        this->output_.DeleteNode(point_data);
    }

}


void ParaviewExporter::Export(const std::string &export_filename)
{    

    // Initialize the path of the exporting file.
    std::string path = "";

    // Position of the last slash in the exporting file's name.
    std::size_t last_slash = export_filename.find_last_of("/\\");

    // Get the path directory of the exporting file name.
    if (last_slash != std::string::npos) {
        path = export_filename.substr(0, last_slash);
    }

    // Create the path's directory if it doesn't exist.
    boost::filesystem::path dir(path);
    if (!path.empty() && !boost::filesystem::exists(dir)) {
        boost::filesystem::create_directories(dir);
    }

    // Initialize the exporting file name's extension.
    std::string ext = "";

    // Search for the extension.
    if (export_filename.find_last_of(".") != std::string::npos) {
        ext = export_filename.substr(export_filename.find_last_of("."));
    }

    //Output result with error check.
    tinyxml2::XMLError out_result;

    // Add .vtu extension before exporting if it's missing from the exporting file's name.
    if (ext != ".vtu") {
        std::string complete_filename = export_filename + ".vtu";
        out_result = this->output_.SaveFile(complete_filename.c_str());
    }
    else {
        out_result = this->output_.SaveFile(export_filename.c_str());
    }

    // Check if export was sucessful.
    if (out_result != tinyxml2::XML_SUCCESS) {
        std::string error = "[ExplicitSim ERROR] Unable to export the file: " + export_filename;
        throw std::runtime_error(error.c_str());
    }
}


void ParaviewExporter::CreatePvdAnimation(const std::map<float, std::string> &vtu_files_paths, const std::string &pvd_filename)
{
    // Create pvd animation xml file.
    tinyxml2::XMLDocument animation;

    // Create xml declaration for the animation.
    tinyxml2::XMLDeclaration *declaration = animation.NewDeclaration();
    animation.InsertFirstChild(declaration);

    // Create the main xml branch.
    tinyxml2::XMLElement *vtk_file = animation.NewElement("VTKFile");
    vtk_file->SetAttribute("type", "Collection");
    vtk_file->SetAttribute("version", "0.1");
    vtk_file->SetAttribute("byte_order", "LittleEndian");
    animation.InsertEndChild(vtk_file);

    // Create Collection branch.
    tinyxml2::XMLElement *collection = animation.NewElement("Collection");

    // Generate vtu file paths string.
    for (auto &file_path : vtu_files_paths) {
        // Add file in collection.
        tinyxml2::XMLElement *dataset = animation.NewElement("DataSet");
        dataset->SetAttribute("timestep", file_path.first);
        dataset->SetAttribute("group","");
        dataset->SetAttribute("part",0);
        dataset->SetAttribute("file",file_path.second.c_str());
        collection->InsertEndChild(dataset);
    }
    vtk_file->InsertEndChild(collection);

    // Initialize the path of the animation file.
    std::string path = "";

    // Position of the last slash in the exporting file's name.
    std::size_t last_slash = pvd_filename.find_last_of("/\\");

    // Get the path directory of the exporting file name.
    if (last_slash != std::string::npos) {
        path = pvd_filename.substr(0, last_slash);
    }

    // Create the path's directory if it doesn't exist.
    boost::filesystem::path dir(path);
    if (!path.empty() && !boost::filesystem::exists(dir)) {
        boost::filesystem::create_directories(dir);
    }

    //Export result with error check.
    tinyxml2::XMLError out_result;

    // Add .pvd extension before exporting animation if it's missing from the file's name.
    std::string ext = pvd_filename.substr(pvd_filename.find_last_of("."));
    if (ext != ".pvd") {
        std::string complete_filename = pvd_filename + ".pvd";
        out_result = animation.SaveFile(complete_filename.c_str());
    }
    else {
        out_result = animation.SaveFile(pvd_filename.c_str());
    }

    // Check if export was sucessful.
    if (out_result != tinyxml2::XML_SUCCESS) {
        std::string error = "[ExplicitSim ERROR] Unable to export the ParaView animation file (.pvd).";
        throw std::runtime_error(error.c_str());
    }


}


} //end of namespace ExplicitSim

