#include "ExplicitSim/options_configuration/config_manager.hpp"

namespace ExplicitSim {


//post-commit test
//constructor
ConfigManager::ConfigManager()
{
    this->description_.add_options()
            ("Model.MeshFile", boost_po::value<std::string>()->required(),
             "Absolute path of model's mesh file. Supported formats: [.inp]")
            ("Model.MassScaling", boost_po::value<bool>()->default_value(false),
             "Mass scaling option." )

            ("IntegrationOptions.Adaptive", boost_po::value<bool>()->default_value(false),
             "Adaptive integration points generation option.")
            ("IntegrationOptions.AdaptiveEps", boost_po::value<double>()->default_value(0.1),
             "Numerical relative error for adaptive integration termination.")
            ("IntegrationOptions.AdaptiveLevel", boost_po::value<int>()->default_value(10),
             "Number of adaptive integration levels. Used when 2 or 4 tetrahedron divisions are requested.")
            ("IntegrationOptions.TetrahedronDivisions", boost_po::value<int>()->default_value(4),
             "Number of tetrahedron division. Used in adaptive integration.")
            ("IntegrationOptions.IntegPointsPerTetrahedron", boost_po::value<int>()->default_value(4),
             "Number of integration points per tetrahedron. Used in standard integration.")
			("IntegrationOptions.SaveToFile", boost_po::value<std::string>(),
			"File for saving the integration points (optional). Format: on line per integration point: coord_x, coord_y, coord_z, weight")
			("IntegrationOptions.ReadFromFile", boost_po::value<std::string>(),
			"File from where to read the integration points (optional).")

            ("Material.Type", boost_po::value<std::string>()->default_value("neohookean"),
             "Type of the model's material.")
			("Material.ReadFromFile", boost_po::value<std::string>(),
			"File from where to read the material properties (optional).")
            ("Material.Density", boost_po::value<double>()->default_value(1000.),
             "Value of the material's density.")
            ("Material.YoungModulus", boost_po::value<double>()->default_value(3000.),
             "Value of the material's Young modulus.")
            ("Material.PoissonRatio", boost_po::value<double>()->default_value(0.49),
             "Value of the material's Poisson's ratio.")
            // Ogden material parameters
            ("Material.Mu", boost_po::value<double>()->default_value(0.0),
             "Value of the material's shear modulus.")
            ("Material.Alpha", boost_po::value<double>()->default_value(0.0),
             "Value of the material's alpha value.")
            ("Material.D1", boost_po::value<double>()->default_value(0.0),
             "Value of the material's D1 value.")

            ("ShapeFunction.Type", boost_po::value<std::string>()->default_value("mmls"),
             "Type of the used shape function.")
            ("ShapeFunction.BasisFunctionType", boost_po::value<std::string>()->default_value("quadratic"),
             "Type of the used basis function.")
            ("ShapeFunction.UseExactDerivatives", boost_po::value<bool>()->default_value(true),
             "Use exact or approximate derivatives of the shape functions.")
            ("ShapeFunction.DilatationCoefficient", boost_po::value<double>()->default_value(1.5),
             "Dilatation (expansion) of the nodes' support domain size.")

            ("Boundary.FixedName", boost_po::value<std::vector<std::string> >(),
             "List of boundary names for fixed displacement conditions.")
            ("Boundary.FixedAxes", boost_po::value<std::vector<std::string> >(),
             "Fixed axes of fixed displacement conditions.")

            ("Loading.LoadName", boost_po::value<std::vector<std::string> >(),
             "Boundary name for loading condition application.")
            ("Loading.LoadAxes", boost_po::value<std::vector<std::string> >(),
             "Triplet to set loading application on the x, y, z axes. Values: [1 | 0]")
            ("Loading.Displacement", boost_po::value<std::vector<std::string> >(),
             "Displacement to be applied on each axis.")
			("Loading.LoadCurve", boost_po::value<std::vector<std::string> >(),
				"Load curve to use (smooth, linear).")
			("Loading.ReadFromFile", boost_po::value<std::vector<std::string> >(),
				"File from where to read loading.")
			("Loading.FileLoadCurve", boost_po::value<std::vector<std::string> >(),
				"Load curve for the file loadings.")
            ("External.ExternalName", boost_po::value<std::vector<std::string> >(),
             "Boundary name for loading condition application.")
            ("External.ExternalAxes", boost_po::value<std::vector<std::string> >(),
             "Triplet to set loading application on the x, y, z axes. Values: [1 | 0]")
            ("External.ExternalForces", boost_po::value<std::vector<std::string> >(),
             "ExternalForces to be applied on each axis.")
                  ("External.ExternalCurve", boost_po::value<std::vector<std::string> >(),
                        "Load curve to use (smooth, linear).")
                  ("External.ReadFromFile", boost_po::value<std::vector<std::string> >(),
                        "File from where to read loading.")
                  ("External.FileLoadCurve", boost_po::value<std::vector<std::string> >(),
                        "Load curve for the file loadings.")

            ("Gravity.gravity", boost_po::value<std::vector<std::string> >(),
            "gravity")
            
            ("Contacts.NodeSet", boost_po::value<std::vector<std::string> >(),
                              "Slave node set.")
            ("Contacts.Surface", boost_po::value<std::vector<std::string> >(),
                              "Surface node set.")

            ("EBCIEM.UseEBCIEM", boost_po::value<bool>()->default_value(false),
             "Use EBCIEM method for imposition of essential boundary conditions.")
            ("EBCIEM.UseSimplifiedVersion", boost_po::value<bool>()->default_value(false),
             "Use simplified version of EBCIEM (No FEM shape function contribution).")

			("DynamicRelaxation.LoadTime", boost_po::value<double>(),
			"Time for loading application.")
			("DynamicRelaxation.EquilibriumTime", boost_po::value<double>()->default_value(10.),
             "Time for solution to reach equilibrium in dynamic relaxation.")
            ("DynamicRelaxation.LoadConvRate", boost_po::value<double>()->default_value(0.999),
             "Convergence rate during loading application.")
            ("DynamicRelaxation.AfterLoadConvRate", boost_po::value<double>()->default_value(0.99),
             "Convergence rate after loading application (during dynamic relaxation).")
            ("DynamicRelaxation.StopUpdateConvRateStepsNum", boost_po::value<int>()->default_value(2000),
             "Number of time steps after which convergence rate updating stops to avoid truncation errors.")
            ("DynamicRelaxation.ConvRateDeviation", boost_po::value<double>()->default_value(0.0001),
             "Deviation to check convergence rate stability.")
            ("DynamicRelaxation.ForceDispUpdateStepsNum", boost_po::value<int>()->default_value(200),
             "Number of steps after which forces and displacements are updated.")
            ("DynamicRelaxation.StableConvRateStepsNum", boost_po::value<int>()->default_value(20),
             "Number of steps after which forces and displacements are updated.")
            ("DynamicRelaxation.ConvRateStopDeviation", boost_po::value<double>()->default_value(0.002),
             "Deviation to decide if the convergence rate has been stabilized between the updates.")
            ("DynamicRelaxation.StopConvRateError", boost_po::value<double>()->default_value(0.2),
             "Estimated error in convergence rate for simulation termination.")
            ("DynamicRelaxation.StopAbsError", boost_po::value<double>()->default_value(0.00001),
             "Absolute error for simulation termination.")
            ("DynamicRelaxation.StopStepsNum", boost_po::value<int>()->default_value(20),
             "Number of consecutive steps for which termination criteria must be satisfied before ending the simulation.")

            ("MTLED.SaveProgressSteps", boost_po::value<unsigned int>()->default_value(1),
             "Number of consecutive steps after which the MTLED progress in displacements and forces calculation is saved.")
            ("MTLED.UsePredefinedStableTimeStep", boost_po::value<bool>()->default_value(false),
             "Use the stable time step defined by the user.")
            ("MTLED.StableTimeStep", boost_po::value<double>()->default_value(0.000001),
             "User-defined stable time step.")
            ("Output.FilePath", boost_po::value<std::string>(),
             "Path to the folder where output should be saved.")
            ("Output.FileName", boost_po::value<std::string>(),
             "Filename for the output file(s) [.vtu].")
            ("Output.AnimationName", boost_po::value<std::string>(),
             "Name of the animation file [.pvd] that collects the [.vtu] files for easy processing in ParaView.");
}


ConfigManager::~ConfigManager()
{}



void ConfigManager::ReadConfigFile(const std::string &config_filename)
{
    std::ifstream input_stream(config_filename.c_str(), std::ifstream::in);
    if (!input_stream.is_open()) {
        std::string error = "[ExplicitSim ERROR] cannot open ExplicitSim configuration file: " + config_filename;
        throw std::invalid_argument(error.c_str());
    }

    // Store the parsed configuration file.
    boost_po::store(boost_po::parse_config_file(input_stream, this->description_), this->var_map_);
    input_stream.close();
    boost_po::notify(this->var_map_);
}


void ConfigManager::ReadConfigStream(std::istream &config_stream)
{
    boost_po::store(boost_po::parse_config_file(config_stream, this->description_), this->var_map_);
    boost_po::notify(this->var_map_);
}


std::string ConfigManager::PrintSampleFile()
{
    std::string sample = "#\n"
            "# ExplicitSim - Software for solving PDEs using explicit methods.\n"
            "# Copyright (C) 2017  <Konstantinos A. Mountris> <konstantinos.mountris@gmail.com>\n"
            "#\n"
            "# This program is free software: you can redistribute it and/or modify\n"
            "# it under the terms of the GNU General Public License as published by\n"
            "# the Free Software Foundation, either version 3 of the License, or\n"
            "# (at your option) any later version.\n"
            "#\n"
            "# This program is distributed in the hope that it will be useful,\n"
            "# but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
            "# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
            "# GNU General Public License for more details.\n"
            "#\n"
            "# You should have received a copy of the GNU General Public License\n"
            "# along with this program.  If not, see <http://www.gnu.org/licenses/>.\n"
            "#\n"
            "# Contributors (alphabetically):\n"
            "#      George C. BOURANTAS\n"
            "#      Grand R. JOLDES\n"
            "#      Konstantinos A. MOUNTRIS\n"
            "#\n\n\n"
            "######                  ExplicitSim V0.0.1 configuration file sample.                  ######\n"
            "\n"
            "# Section: Introduction\n"
            "# ---------------------\n"
            "\n"
            "# This text file demonstrates all the available arguments to launch and execute an ExplicitSim\n"
            "# simulation. These arguments are grouped in sections following a logic order in this sample,\n"
            "# but they can be given in any order in general. Other than the mandatory arguments [stated later],\n"
            "# the rest of the arguments can be ommitted and the default variables used in this sample will be used.\n"
            "\n\n"
            "[Model]                                                 # Section: Mesh\n"
            "                                                        # -------------\n"
            "\n"
            "MeshFile = /path/to/mesh/file                           # For now only mesh files in Abaqus\n"
            "                                                        # format (.inp) are supported. [mandatory]\n"
            "\n"
            "MassScaling = false                                     # Scale the mass of the domain nodes\n"
            "                                                        # in order to reduce the number of explicit time steps\n"
            "                                                        # of the nodes. Values: [true | 1]  [false | 0]\n"
            "\n\n"
            "[IntegrationOptions]                                    # Section: Integration Options\n"
            "                                                        # ----------------------------\n"
            "\n"
            "Adaptive = false                                        # Generate integration points adaptively.\n"
            "                                                        # Values: [true | 1]  [false | 0]\n"
            "\n"
            "AdaptiveEps = 0.1                                       # Numerical relative error value for\n"
            "                                                        # adaptive integration termination.\n"
            "\n"
            "AdaptiveLevel = 10                                      # Number of adaptive integration levels used\n"
            "                                                        # when [2] or [4] tetrahedron divisions are requested.\n"
            "\n"
            "TetrahedronDivisions = 4                                # Number of tetrahedron division. Used in adaptive\n"
            "                                                        # integration. Values: [2] [4] [8]\n"
            "\n"
            "IntegPointsPerTetrahedron = 4                           # Number of integration points per tetrahedron. Used in\n"
            "                                                        # standard integration. Values: [1] [4] [5]\n"
            "\n\n"
            "[Material]                                              # Section: Material\n"
            "                                                        # -----------------\n"
            "\n"
            "Type = neohookean                                       # Type of the model's material. Currently\n"
            "                                                        # only homogeneous models with material of\n"
            "                                                        # type [neohookean] are supported.\n"
            "\n"
            "Density = 1000                                          # Value of the material's density.\n"
            "                                                        # Measure unit: [kg/m3]\n"
            "\n"
            "YoungModulus = 3000                                     # Value of the material's Young modulus.\n"
            "                                                        # Measure unit: [Pa]\n"
            "\n"
            "PoissonRatio = 0.49                                     # Value of the material's Poisson's ratio.\n"
            "                                                        # Measure unit: [none]\n"
            "\n\n"
            "[ShapeFunction]                                         # Section: Shape Function\n"
            "                                                        # ------------------------\n"
            "\n"
            "Type = mmls                                             # Type of the shape function.\n"
            "                                                        # Supported: [mmls]\n"
            "\n"
            "BasisFunctionType = quadratic                           # The type of the basis function.\n"
            "                                                        # Supported: [linear | quadratic]\n"
            "\n"
            "UseExactDerivatives = true                              # Use either exact or approximate shape function derivatives.\n"
            "                                                        # Values: [true | 1]  [false | 0]\n"
            "\n"
            "DilatationCoefficient = 1.5                             # Coefficient to dilate (expand) the size\n"
            "                                                        # of the support domain of each node.\n"
            "\n\n"
            "[Boundary]                                              # Section: Boundary Conditions\n"
            "                                                        # ----------------------------\n"
            "\n"
            "FixedName = fixed_boundary_name                         # Name of the boundary nodes set to apply fixed displacement condition.\n"
            "\n"
            "FixedAxes = 1 1 1                                       # Triplet to set fixation on the x, y, z axes.\n"
            "                                                        # Values: [1 | 0]\n"
            "                                                        # [1] -> Fixed, [0] -> Not Fixed. Expects three values.\n"
            "\n\n"
            "[Loading]                                               # Section: Loading Conditions\n"
            "                                                        # ---------------------------\n"
            "\n"
            "LoadName = load_boundary_name                           # Name of the boundary nodes set to apply loading condition.\n"
            "\n"
            "LoadAxes = 1 1 1                                        # Triplet to set loading application on the x, y, z axes.\n"
            "                                                        # Values: [1 | 0]\n"
            "                                                        # [1] -> Application, [0] -> No Application. Expects three values.\n"
            "\n"
            "Displacement = -0.02  0.1  1.3                          # Displacement to be applied on each axis.\n"
            "\n\n"
            "[EBCIEM]                                                # Section: EBCIEM method\n"
            "                                                        # ----------------------\n"
            "\n"
            "UseEBCIEM = false                                       # Use EBCIEM method for imposition of essential\n"
            "                                                        # boundary conditions when aprroximants with no\n"
            "                                                        # Kronecker delta property (e.g. MMLS) are used.\n"
            "                                                        # Values: [true | 1]  [false | 0]\n"
            "\n"
            "UseSimplifiedVersion = false                            # Use the simplified version of EBCIEM where no\n"
            "                                                        # FEM shape function contribution is considered.\n"
            "                                                        # If [false | 0] FEM shape functions are calculated for\n"
            "                                                        # the boundary surfaces.\n"
            "                                                        # Values: [true | 1]  [false | 0]\n"
            "\n\n"
            "[DynamicRelaxation]                                     # Section: Dynamic Relaxation\n"
            "                                                        # ---------------------------\n"
			"\n"
			"LoadTime = 2.0                                          # Time for loading application.\n"
            "\n"
            "EquilibriumTime = 10.0                                  # Time for solution to reach equilibrium in dynamic relaxation.\n"
            "                                                        # Measure unit: [s]\n"
            "\n"
            "LoadConvRate = 0.999                                    # Convergence rate during loading application.\n"
            "\n"
            "AfterLoadConvRate = 0.99                                # Convergence rate after loading application (during dynamic relaxation).\n"
            "\n"
            "StopUpdateConvRateStepsNum = 2000                       # Number of time steps after which convergence rate updating stops\n"
            "                                                        # to avoid truncation errors.\n"
            "\n"
            "ConvRateDeviation = 0.0001                              # Deviation to check convergence rate stability.\n"
            "\n"
            "ForceDispUpdateStepsNum = 200                           # Number of steps after which forces and displacements are updated.\n"
            "\n"
            "StableConvRateStepsNum = 20                             # Number of consecutive steps after which convergence rate\n"
            "                                                        # stability requirement is considered to have been met.\n"
            "\n"
            "ConvRateStopDeviation = 0.002                           # Deviation to decide if the convergence rate has been\n"
            "                                                        # stabilized between the updates.\n"
            "\n"
            "StopConvRateError = 0.2                                 # Estimated error in convergence rate for simulation termination.\n"
            "\n"
            "StopAbsError = 0.00001                                  # Absolute error for simulation termination.\n"
            "\n"
            "StopStepsNum = 100                                      # Number of consecutive steps for which termination criteria must be\n"
            "\n"
            "                                                        # satisfied before ending the simulation.\n"
            "\n\n"
            "[MTLED]                                                 # Section: Meshless Total Lagrangian Explicit Dynamics\n"
            "                                                        # ----------------------------------------------------\n"
            "\n"
            "SaveProgressSteps = 0                                   # Number of steps after which the MTLED progress\n"
            "                                                        # in displacements and forces calculation is saved.\n"
            "                                                        # If Value: [0] only initial and final states are stored.\n"
            "\n"
            "UsePredefinedStableTimeStep = false                     # Use the stable time step defined by the user. If false\n"
            "                                                        # The stable time step is computed automatically.\n"
            "                                                        # Values: [true | 1]  [false | 0]\n"
            "\n"
            "StableTimeStep = 0.000001                               # User-defined stable time step. Lower time step leads to better\n"
            "                                                        # stability and increased computational time. Measure unit: [s]\n"
            "\n\n"
            "[Output]                                                # Section: Output\n"
            "                                                        # ---------------\n"
            "\n"
            "FilePath = /path/to/save/output/                        # Path to the folder where output should be saved.\n"
            "                                                        # WARNING: The last character of the path should be slash.\n"
            "\n"
            "FileName = output_filename                              # Filename for the output file(s) [.vtu].\n"
            "                                                        # ExplicitSim adds an id to the filename according to the\n"
            "                                                        # simulation timestep it refers if multiple states were saved.\n"
            "                                                        # For now the id corresponds to saved states.\n"
            "\n"
            "AnimationName = animation_filename                      # Name of the animation file [.pvd] that collects the\n"
            "                                                        # [.vtu] files for easy processing in ParaView.\n";

    return sample;

}


} // End of namespace ExplicitSim
