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
 *      Benjamin F. ZWICK
 *		Yu. Yue
 */

#include <ExplicitSim/explicit_sim.hpp>

#include <boost/filesystem.hpp>

#include <cstddef>
#include <string>
#include <fstream>

using namespace ExplicitSim;

int main(int argc, char *argv[]) {

    try {

        // Initialize configuration manager.
        ConfigManager *config = new ConfigManager();

        // Initialize configuration filename empty to be read during the ExplicitSim execution.
        std::string config_filename = "config.ini";

        // Check if configuration file was provided during execution.
        if (argc == 1) {
            std::cout << Logger::Warning("No input file was specified. Type configuration filename with absolute path.\n"
                         "Otherwise tap '-g' to generate sample configuration file or '-q' to exit ExplicitSim.\nInput filename: ");

            // Read configuration file given by the user.
            std::cin >> config_filename;

            if (config_filename == "-g") {
                std::cout << Logger::Message("Give text file name [.ini] with absolute path to store the sample configuration "
                             "file\nor tap '-t' to print in terminal.\nSample filename: ");

                std::string sample_filename = "";

                std::cin >> sample_filename;

                if (sample_filename == "-t") {
                    std::cout << config->PrintSampleFile() << std::endl;
                    std::cout << Logger::Message("Save the sample configuration in a text file [.ini], edit it according to your simulation,"
                                 " and relaunch ExplicitSimRun passing your configuration file as argument.\n");
                    return EXIT_SUCCESS;
                }
                else {
                    // Initialize the path of the sample configuration file.
                    std::string sample_path = "";

                    // Position of the last slash in the sample configuration file.
                    std::size_t last_slash = sample_filename.find_last_of("/\\");

                    // Get the path directory of the sample configuration file.
                    if (last_slash != std::string::npos) {
                        sample_path = sample_filename.substr(0, last_slash);
                    }

                    // Create the path's directory if it doesn't exist.
                    boost::filesystem::path dir(sample_path);
                    if (!sample_path.empty() && !boost::filesystem::exists(dir)) {
                        boost::filesystem::create_directories(dir);
                    }

                    // Position of the last slash in the exporting file's name.
                    std::size_t last_dot = sample_filename.find_last_of(".");

                    // Initialize sample configuration file extension.
                    std::string sample_ext = "";

                    // Get sample configuration file extension.
                    if (last_dot != std::string::npos) {
                        sample_ext = sample_filename.substr(last_dot);
                    }

                    // Add extension if necessary
                    if (sample_ext != ".ini") { sample_filename += ".ini"; }

                    // Output the sample file.
                    std::ofstream sample_output(sample_filename, std::ios_base::out | std::ios_base::trunc);
                    sample_output << config->PrintSampleFile();
                    std::cout << Logger::Message("Sample configuration file saved at: ") << sample_filename << std::endl;
                    std::cout << Logger::Message("Edit the sample configuration according to your simulation and "
                                 "relaunch ExplicitSimRun passing your configuration file as argument.\n");
                    return EXIT_SUCCESS;
                }

            }

            if (config_filename == "-q") { std::cout << Logger::Message("User requested termination. See you soon!\n"); exit(0); }
            std::cout << config_filename << std::endl;
            exit(0);
        }
        else { config_filename = argv[1]; }

        // Read configuration file.
        config->ReadConfigFile(config_filename);

        std::cout << "<< Welcome to ExplicitSim_optimise_gravity_version >>\n";
        std::cout << Logger::Message("Loading configuration file: ") << config_filename << std::endl;

        // Profiling spent time in ExplicitSim
        Timer timer;

        // Set the weak form model.
        WeakModel3D model;
        model.LoadMeshRepresentation(config->RetrieveArgument<std::string>("Model.MeshFile"));
        model.CreateGridRepresentation();

        std::cout << Logger::Message("Model has nodes: ") << model.TetrahedralMesh().Nodes().size() <<
                     " and elements: " << model.TetrahedralMesh().Elements().size() << std::endl;

        // Set support domain.
        SupportDomain support;
        support.SetInfluenceNodes(model.TetrahedralMesh().Nodes(), model.TetrahedralMesh().NodesNum());
        support.SetInfluenceTetrahedra(model.TetrahedralMesh().Elements());

        // Compute the influnce radiuses of the support domain.
        support.ComputeInfluenceNodesRadiuses(config->RetrieveArgument<double>("ShapeFunction.DilatationCoefficient"));

		if (config->VarMap().count("IntegrationOptions.ReadFromFile"))
		{
			// Integration points filename.
			std::string intput_filename = config->RetrieveArgument<std::string>("IntegrationOptions.ReadFromFile");

			try
			{
				model.IntegrationPoints().vReadFromFile(intput_filename);
				std::cout << Logger::Message("Integration points read from: ") << intput_filename << ".\n";
			}
			catch (std::exception& e)
			{
				std::cerr << e.what() << std::endl;
			}
		}
		if (model.IntegrationPoints().PointsNum() == 0)
		{
			// Setting options for integration points generation.
			IntegOptions options;
			options.is_adaptive_ = config->RetrieveArgument<bool>("IntegrationOptions.Adaptive");
			options.adaptive_eps_ = config->RetrieveArgument<double>("IntegrationOptions.AdaptiveEps");
			options.adaptive_level_ = config->RetrieveArgument<int>("IntegrationOptions.AdaptiveLevel");
			options.tetra_divisions_ = config->RetrieveArgument<int>("IntegrationOptions.TetrahedronDivisions");
			options.integ_points_per_tetra_ = config->RetrieveArgument<int>("IntegrationOptions.IntegPointsPerTetrahedron");

			if (options.is_adaptive_) {
				std::cout << Logger::Message("Adaptive integration: ON\n");
				std::cout << Logger::Message("Tetrahedron divisions used: ") << options.tetra_divisions_ << std::endl;
				std::cout << Logger::Message("Number of integration points per tetrahedron division used: 4\n");
			}
			else {
				std::cout << Logger::Message("Adaptive integration: OFF\n");
				std::cout << Logger::Message("Number of integration points per element used: ")
					<< options.integ_points_per_tetra_ << std::endl;
			}

			// Time only for integration points generation.
			timer.Reset();

			// Create integration points for the model.
			model.CreateIntegrationPoints(options, support);
			std::cout << Logger::Message("Execution time for integration points generation: ") << timer.PrintElapsedTime() << "\n";

			// Save integration points to file (if configured)
			if (config->VarMap().count("IntegrationOptions.SaveToFile") && (model.IntegrationPoints().PointsNum() > 0))
			{
				std::cout << Logger::Message("Saving integration points...\n");

				// Integration points filename.
				std::string output_filename = config->RetrieveArgument<std::string>("IntegrationOptions.SaveToFile");

				try
				{
					model.IntegrationPoints().vSaveToFile(output_filename);
					std::cout << Logger::Message("Integration points saved to: ") << output_filename << ".\n";
				}
				catch (std::exception& e)
				{
					std::cerr << e.what() << std::endl;
				}
			}
		}

		std::cout << Logger::Message("Model has integration points: ") << model.IntegrationPoints().PointsNum() << "\n";

        // Time only for closest nodes.
        timer.Reset();
        // Find influence nodes indices of integration points.
		auto neighs_ids = support.ClosestNodesIdsTo(model.IntegrationPoints().Coordinates());

		std::cout << Logger::Message("Execution time for neighbor nodes computation using ")
#ifdef EXPLICITSIM_NEAREST_NEIGHBOR_EXHAUSTIVE
				  << "exhaustive search: "
#endif
#ifdef EXPLICITSIM_NEAREST_NEIGHBOR_CGAL
				  << "CGAL: "
#endif
#ifdef EXPLICITSIM_NEAREST_NEIGHBOR_BUCKETSEARCH
				  << "bucket search: "
#endif
				  << timer.PrintElapsedTime() << "\n";

        std::cout << Logger::Message("The minimum and maximum number of support nodes: ")
                  << support.MinSupportNodesIn(neighs_ids) << " - " << support.MaxSupportNodesIn(neighs_ids) << std::endl;

		// Assign material to the model
		std::string material_type = config->RetrieveArgument<std::string>("Material.Type");
		Material *material;
		if (material_type == "neohookean")
		{
			material = new NeoHookean;
		}
		else if (material_type == "ogden")
		{
			material = new Ogden;
		}
		else
		{
			std::string error = Logger::Error("Unknown material type: " + material_type);
			throw std::invalid_argument(error.c_str());
		}

		if (config->VarMap().count("Material.ReadFromFile"))
		{
			// Material parameters filename.
			std::string intput_filename = config->RetrieveArgument<std::string>("Material.ReadFromFile");

			try
			{
				material->OReadFromFile(intput_filename);
				std::cout << Logger::Message("Material parameters read from: ") << intput_filename << ".\n";
			}
			catch (std::exception& e)
			{
				std::cerr << e.what() << std::endl;
			}
		}
		if ((material->PointsNumber() == 0) || (material->PointsNumber() != model.IntegrationPoints().Coordinates().size()))
		{

			material->SetPointsNumber(model.IntegrationPoints().Coordinates().size());

			// Set material parameters.
			//material->SetDensity(config->RetrieveArgument<double>("Material.Density"));
			//if (material_type == "neohookean")
			//{
			//	material->SetYoungModulus(config->RetrieveArgument<double>("Material.YoungModulus"));
			//	material->SetPoissonRatio(config->RetrieveArgument<double>("Material.PoissonRatio"));
			//}
			//else if (material_type == "ogden")
			//{
				//double mu = config->RetrieveArgument<double>("Material.Mu");
				//double alpha = config->RetrieveArgument<double>("Material.Alpha");
				//double D1 = config->RetrieveArgument<double>("Material.D1");

				/*double kappa=2.0/D1;
				double youngs_modulus = (9.0*kappa*mu)/(3.0*kappa+mu);
				double poissons_ratio = (3.0*kappa-2.0*mu)/(2.0*(3.0*kappa+mu));*/
/*
				material->SetYoungModulus(youngs_modulus);
				material->SetPoissonRatio(poissons_ratio);*/
				//material->SetLameMu(mu);
				//material->SetD1(D1);
				//material->SetAlpha(alpha);
				//static_cast<Ogden*>(material)->SetAlpha(config->RetrieveArgument<double>("Material.Alpha"));
			//}
			//else
			//{
			//	std::string error = Logger::Error("Unknown material type: " + material_type);
			//	throw std::invalid_argument(error.c_str());
			//}
		}

        // Compute elastic parameters and wave speed.
        material->ComputeBulkModulus();
        material->ComputeYM();
        material->ComputePR();
        material->ComputeLameLambdaMu();
        material->ComputeWaveSpeed();

		// Initialize the boundary and loading conditions handler.
		ConditionsHandler cond_handler(model.TetrahedralMesh().NodesNum(), &model.TetrahedralMesh().NodeSets());

		// Load BC from files
		/*if (config->VarMap().count("Loading.ReadFromFile")) {
			int load_file_num = config->OptionsNumInList<std::string>("Loading.ReadFromFile");
			int load_curves_num = config->OptionsNumInList<std::string>("Loading.FileLoadCurve");

			if (load_file_num != load_curves_num) {
				throw std::invalid_argument(Logger::Error("Number of boundary condition files names does not "
					"match the number of file load curves. "
					"Check the given configuration options.").c_str());
			}

			// Add load displacement conditions to the conditions handler.
			for (int cond_num = 0; cond_num != load_file_num; ++cond_num) {
				// String stream to parse conditions from string argument.
				std::string input_filename = config->RetrieveArgumentFromList<std::string>("Loading.ReadFromFile", cond_num);
				std::string LoadCurveName = config->RetrieveArgumentFromList<std::string>("Loading.FileLoadCurve", cond_num);

				if (!cond_handler.boLoadCurveImplemented(LoadCurveName))
				{
					std::string error = Logger::Error("Load curve " + LoadCurveName + " not found!");
					throw std::invalid_argument(error.c_str());
				}

				try
				{
					cond_handler.vReadLoadingFromFile(LoadCurveName, input_filename, ((TetraMesh &)(model.TetrahedralMesh())).pCGetMeshIO()->pvGetOffsettedNodes(), ((TetraMesh &)(model.TetrahedralMesh())).pu32GetReindexingArray());
					std::cout << Logger::Message("Loading read from: ") << input_filename << ".\n";
				}
				catch (std::exception& e)
				{
					std::cerr << e.what() << std::endl;
				}
			}

		}*/

		// Read the external loading file
		if (config->VarMap().count("External.ReadFromFile")) {
			int external_file_num = config->OptionsNumInList<std::string>("External.ReadFromFile");
			int external_curves_num = config->OptionsNumInList<std::string>("External.FileLoadCurve");

			if (external_file_num != external_curves_num) {
				throw std::invalid_argument(Logger::Error("Number of boundary condition files names does not "
					"match the number of file load curves. "
					"Check the given configuraton options.").c_str());
			}

			// Add external force conditions to the conditions handler.
			for (int cond_num = 0; cond_num != external_file_num; ++cond_num) {
				//String stream to parse condition from string argument. 
				std::string input_filename = config->RetrieveArgumentFromList<std::string>("External.ReadFromFile", cond_num);
				std::string LoadCurveName = config->RetrieveArgumentFromList<std::string>("External.FileLoadCurve",cond_num);

				if (!cond_handler.boLoadCurveImplemented(LoadCurveName))
				{
					std::string error = Logger::Error("Load curve for external force " + LoadCurveName + " not found");
					throw std::invalid_argument(error.c_str());
				}

				try
				{
					// to do cond_handler.vReadExternalLoadingFromFile
					cond_handler.vReadExternalLoadingFromFile(LoadCurveName, input_filename, ((TetraMesh &)(model.TetrahedralMesh())). pCGetMeshIO()->pvGetOffsettedNodes(), ((TetraMesh &)(model.TetrahedralMesh())).pu32GetReindexingArray());
					std::cout << Logger::Message("Loading read from: ") << input_filename << ".\n";
				}
				catch (std::exception& e)
				{
					std::cerr<<e.what()<<std::endl;
				}
			}
		}

        // Time only for shape functions.
        timer.Reset();

        // Shape Functions
        Mmls3d mmls3d;
        mmls3d.SetBasisFunctionType(config->RetrieveArgument<std::string>("ShapeFunction.BasisFunctionType"));
        mmls3d.SetExactDerivativesMode(config->RetrieveArgument<bool>("ShapeFunction.UseExactDerivatives"));
        mmls3d.ComputeShFuncAndDerivs(model.TetrahedralMesh().Nodes(), model.TetrahedralMesh().NodesNum(), model.IntegrationPoints().Coordinates(),
			(uint32_t)model.IntegrationPoints().Coordinates().size(), neighs_ids, support.InfluenceNodesRadiuses());

        std::cout << Logger::Message("Execution time computing shape functions at integration points: ") << timer.PrintElapsedTime() << "\n";

		timer.Reset();
		// Find influence nodes inidices the nodes of the model's geometry.
		auto nodal_neigh_ids = support.ClosestNodesIdsTo(model.TetrahedralMesh().NodeCoordinates());

		// Compute the mmls approximants for the nodes of the model's geometry.
		Mmls3d nodal_mmls;
		nodal_mmls.SetBasisFunctionType(mmls3d.BaseFunctionType());
		nodal_mmls.SetExactDerivativesMode(mmls3d.UseExactDerivatives());
		nodal_mmls.ComputeShFuncAndDerivs(model.TetrahedralMesh().Nodes(), model.TetrahedralMesh().NodesNum(),
			model.TetrahedralMesh().NodeCoordinates(), model.TetrahedralMesh().NodesNum(),
			nodal_neigh_ids, support.InfluenceNodesRadiuses());

		std::cout << Logger::Message("Execution time for computing shape functions at nodes: ") << timer.PrintElapsedTime() << "\n";

        // The dynamic relaxation properties.
        DynRelaxProp dr;
		dr.SetLoadTime(config->RetrieveArgument<double>("DynamicRelaxation.LoadTime"));
        dr.SetEquilibriumTime(config->RetrieveArgument<double>("DynamicRelaxation.EquilibriumTime"));
        dr.SetLoadConvRate(config->RetrieveArgument<double>("DynamicRelaxation.LoadConvRate"));
        dr.SetAfterLoadConvRate(config->RetrieveArgument<double>("DynamicRelaxation.AfterLoadConvRate"));
        dr.SetStopUpdateConvRateStepsNum(config->RetrieveArgument<int>("DynamicRelaxation.StopUpdateConvRateStepsNum"));
        dr.SetConvRateDeviation(config->RetrieveArgument<double>("DynamicRelaxation.ConvRateDeviation"));
        dr.SetForceDispUpdateStepsNum(config->RetrieveArgument<int>("DynamicRelaxation.ForceDispUpdateStepsNum"));
        dr.SetStableConvRateStepsNum(config->RetrieveArgument<int>("DynamicRelaxation.StableConvRateStepsNum"));
        dr.SetConvRateStopDeviation(config->RetrieveArgument<double>("DynamicRelaxation.ConvRateStopDeviation"));
        dr.SetStopConvRateError(config->RetrieveArgument<double>("DynamicRelaxation.StopConvRateError"));
        dr.SetStopAbsError(config->RetrieveArgument<double>("DynamicRelaxation.StopAbsError"));
        dr.SetStopStepsNum(config->RetrieveArgument<int>("DynamicRelaxation.StopStepsNum"));

        // Solve the model explicitly with MTLED.
        Mtled solver;

        // Compute the time steps.
        solver.ComputeTimeSteps(material->WaveSpeed(), neighs_ids, mmls3d);

        // Compute the mass of the model.
		bool boUseConfiguredTimeStep = config->RetrieveArgument<bool>("MTLED.UsePredefinedStableTimeStep");
		bool boScaleMass = config->RetrieveArgument<bool>("Model.MassScaling");
		double dConfiguredTimeStep = config->RetrieveArgument<double>("MTLED.StableTimeStep");
		double dSafetyFactor = 1.5;

		std::cout << "[ExplicitSim] Mass scaling: " << (boScaleMass?"On\n": "Off\n");

        // Compute the stable time step for the explicit solution.
        if (boUseConfiguredTimeStep) 
		{
			model.ComputeMass(material->Density(), solver.TimeSteps(), dConfiguredTimeStep*dSafetyFactor,
				neighs_ids, boScaleMass);

            solver.SetStableStep(dConfiguredTimeStep);
        }
        else 
		{ 
			model.ComputeMass(material->Density(), solver.TimeSteps(), solver.MaxStep(),
				neighs_ids, boScaleMass);

			solver.ComputeStableStep(boScaleMass, dSafetyFactor);
		}

        // Set steps for progress save.
        solver.SetSaveProgressSteps(config->RetrieveArgument<unsigned int>("MTLED.SaveProgressSteps"));

        std::cout << Logger::Message("Minimum time step: ") << solver.MinStep() << " s\n";
        std::cout << Logger::Message("Maximum time step: ") << solver.MaxStep() << " s\n";
        std::cout << Logger::Message("Used time step: ") << solver.StableStep() << " s\n";

        // Compute dynamic relaxation steps number for equilibrium.
        dr.ComputeStepsNum(solver.StableStep());
		std::cout << Logger::Message("The loading steps number: ") << dr.LoadStepsNum() << std::endl;
        std::cout << Logger::Message("The dynamic relaxation equilibrium steps number: ") << dr.EquilibriumStepsNum() << std::endl;

        // Compute total time steps for solution.
        solver.ComputeTotalTimeStepsNum(dr.LoadStepsNum(), dr.EquilibriumStepsNum());
        std::cout << Logger::Message("The total explicit solution time steps number: ") << solver.TotalTimeStepsNum() << std::endl;

        // Time only for conditions imposition.
        timer.Reset();

       

        // Set fixed displacement conditions.
        if (config->VarMap().count("Boundary.FixedName")) {
            int fixed_name_num = config->OptionsNumInList<std::string>("Boundary.FixedName");
            int fixed_axes_num = config->OptionsNumInList<std::string>("Boundary.FixedAxes");

            // Check if same number of fixed displacement conditions names and fixed axes are given.
            if (fixed_name_num != fixed_axes_num) {
                throw std::invalid_argument(Logger::Error("Number of fixed boundary condition names does not "
                                                          "match to number of fixed axes settings. "
                                                          "Check the given configuration options.").c_str());
            }

            // Fixed axes conditionals.
            bool is_x_fixed = false; bool is_y_fixed = false; bool is_z_fixed = false;

            // Add fixed displacement conditions to the conditions handler.
            for (int cond_num = 0; cond_num != fixed_axes_num; ++cond_num) {
                // String stream to parse conditions from string argument.
                std::stringstream ss(config->RetrieveArgumentFromList<std::string>("Boundary.FixedAxes", cond_num));

                if (!(ss >> is_x_fixed >> is_y_fixed >> is_z_fixed)) {
                    std::string error = Logger::Error("Could not process fixed axes triplet "
                                        "for condition:") + std::to_string(cond_num);
                    throw std::invalid_argument(error.c_str());
                }

                cond_handler.AddDirichlet(is_x_fixed, is_y_fixed, is_z_fixed,
                                          config->RetrieveArgumentFromList<std::string>("Boundary.FixedName", cond_num));

            }
        } // End of Set fixed displacement conditions.


		
		 // Set loaded boundary conditions.
        
		if (config->VarMap().count("Loading.LoadName")) {
			int load_name_num = config->OptionsNumInList<std::string>("Loading.LoadName");
			int load_axes_num = config->OptionsNumInList<std::string>("Loading.LoadAxes");
			int load_disp_num = config->OptionsNumInList<std::string>("Loading.Displacement");
			int load_curves_num = config->OptionsNumInList<std::string>("Loading.LoadCurve");

			// Check if same number of loaded displacement conditions names and load axes/displacements are given.
			if (load_name_num != load_axes_num) {
				throw std::invalid_argument(Logger::Error("Number of load boundary condition names does not "
                                                          "match the number of load axes settings. "
					"Check the given configuration options.").c_str());
			}

			if (load_name_num != load_disp_num) {
				throw std::invalid_argument(Logger::Error("Number of load boundary condition names does not "
					"match the number of displacements settings. "
					"Check the given configuration options.").c_str());
			}

			if (load_name_num != load_curves_num) {
				throw std::invalid_argument(Logger::Error("Number of load boundary condition names does not "
					"match the number of load curve settings. "
					"Check the given configuration options.").c_str());
			}

			// Fixed axes conditionals.
			bool is_x_loaded = false; bool is_y_loaded = false; bool is_z_loaded = false;
			float dx = 0; float dy = 0; float dz = 0;

			// Add load displacement conditions to the conditions handler.
			for (int cond_num = 0; cond_num != load_axes_num; ++cond_num) {
				// String stream to parse conditions from string argument.
				std::stringstream ss(config->RetrieveArgumentFromList<std::string>("Loading.LoadAxes", cond_num));

				if (!(ss >> is_x_loaded >> is_y_loaded >> is_z_loaded)) {
					std::string error = Logger::Error("Could not process fixed axes triplet "
						"for condition:") + std::to_string(cond_num);
					throw std::invalid_argument(error.c_str());
				}

				std::stringstream ss1(config->RetrieveArgumentFromList<std::string>("Loading.Displacement", cond_num));

				if (!(ss1 >> dx >> dy >> dz)) {
					std::string error = Logger::Error("Could not process displacements "
						"for condition:") + std::to_string(cond_num);
					throw std::invalid_argument(error.c_str());
				}

				std::string LoadCurveName = config->RetrieveArgumentFromList<std::string>("Loading.LoadCurve", cond_num);
				if (!cond_handler.boLoadCurveImplemented(LoadCurveName))
				{
					std::string error = Logger::Error("Load curve " + LoadCurveName + " not found!");
					throw std::invalid_argument(error.c_str());
				}

				cond_handler.AddLoading(LoadCurveName, is_x_loaded, is_y_loaded, is_z_loaded, dx, dy, dz,
					config->RetrieveArgumentFromList<std::string>("Loading.LoadName", cond_num));
			}
		} // End of Set loaded displacement conditions.
		// Set Gravity loading 
		if (config->VarMap().count("Gravity.gravity")) {
			int gravity_num = config->OptionsNumInList<std::string>("Gravity.gravity");
			//std::cout << Logger::Message("gravity starts")<<std::endl;
			//std::cout << Logger::Message("gravity number is ")<<gravity_num<<std::endl;

			// Initialise the parameter
			float g_x = 0; float g_y = 0; float g_z = 0;
			//std::cout << Logger::Message("gravity_x is")<<g_x<<std::endl;

			for (int cond_num = 0; cond_num != gravity_num; ++cond_num) {
				std::stringstream ss(config->RetrieveArgumentFromList<std::string>("Gravity.gravity", cond_num));
				ss >> g_x >> g_y >> g_z;
				cond_handler.AddGravity(g_x, g_y, g_z);
			}
		}
		// Set External force conditions
		if (config->VarMap().count("External.ExternalName")){
			int external_name_num = config->OptionsNumInList<std::string>("External.ExternalName");
			int external_axes_num = config->OptionsNumInList<std::string>("External.ExternalAxes");
			int external_forces_num = config->OptionsNumInList<std::string>("External.ExternalForces");
			int external_curves_num = config->OptionsNumInList<std::string>("External.Externalcurve");

			//Check if same number of external force conditions names and load axes/forces are given.
			bool is_x_forces = false;
			bool is_y_forces = false;
			bool is_z_forces = false;

			float fx = 0;
			float fy = 0;
			float fz = 0;

			//Add forces conditions to the conditions handler.
			for (int cond_num = 0; cond_num != external_axes_num; ++cond_num) {
				//string stream to parse conditions from string argument.
				std::stringstream ss(config->RetrieveArgumentFromList<std::string>("External.ExternalAxes", cond_num));

				if(!(ss >> is_x_forces >> is_y_forces >> is_z_forces)){
					std::string error = Logger::Error("Could not process fixed axes triplet"
						"for condition:") + std::to_string(cond_num);

					throw std::invalid_argument(error.c_str());
				}

				std::stringstream ss1(config->RetrieveArgumentFromList<std::string>("External.ExternalForces",cond_num));

				if(!(ss1 >> fx >> fy >> fz)) {
					std::string error = Logger::Error("Could not process ExternalForces" 
						"for condition:") + std::to_string(cond_num);
					throw std::invalid_argument(error.c_str());
				}

				std::string LoadCurveName = config->RetrieveArgumentFromList<std::string>("External.ExternalCurve", cond_num);
				if (!cond_handler.boLoadCurveImplemented(LoadCurveName))
				{
					std::string error = Logger::Error("Load curve" + LoadCurveName + "not found!");
					throw std::invalid_argument(error.c_str());
				}

				cond_handler.AddExternalForce(LoadCurveName, is_x_forces, is_y_forces, is_z_forces, fx, fy, fz,
					config->RetrieveArgumentFromList<std::string>("External.ExternalName", cond_num));

			}/*end of add forces to the conditions handler*/

		}/*end of check external force loading*/


		// Set contacts.
		if (config->VarMap().count("Contacts.NodeSet")) {
			int node_set_num = config->OptionsNumInList<std::string>("Contacts.NodeSet");
			int surfaces_num = config->OptionsNumInList<std::string>("Contacts.Surface");
			
			// Check if number of contact node sets equals the number of contact surfaces.
			if (node_set_num != surfaces_num) {
				throw std::invalid_argument(Logger::Error("Number of contact node sets does not "
					"match to number contact surfaces. "
					"Check the given configuration options.").c_str());
			}

			// Add load displacement conditions to the conditions handler.
			for (int i = 0; i != node_set_num; i++) {
				// surface not implemented yet
				cond_handler.vAddContactPair(model.TetrahedralMesh().Surfaces()[0],	
					config->RetrieveArgumentFromList<std::string>("Contacts.NodeSet", i), model.TetrahedralMesh());
			}
		} // End of Set contacts.

		// Extact nodes ids where conditions are imposed.
		cond_handler.vExtractConstrainedDOFs();
		cond_handler.ExtractExternalForcesIndices();

		// Add EBCIEM correction in the conditions handler.
		cond_handler.vComputeCorrectionMatrix(model, nodal_mmls, 
			config->RetrieveArgument<bool>("EBCIEM.UseEBCIEM"), 
			config->RetrieveArgument<bool>("EBCIEM.UseSimplifiedVersion"));

        std::cout << Logger::Message("Execution time for conditions initialization: ") << timer.PrintElapsedTime() << "\n";
        //std::cout << Logger::Message("fMaxAbsDisplacement is: ") << fMaxAbsDisplacement << std::endl;

        // Solving with explicit dynamics.
        std::cout << Logger::Message("Starting the explicit solution of the model...") << std::endl;
        timer.Reset();

        solver.Solve(model, neighs_ids, cond_handler, mmls3d, *material, dr,
                     config->RetrieveArgument<bool>("EBCIEM.UseEBCIEM"), nodal_mmls);
        std::cout << Logger::Message("Execution time for explicit solution: ") << timer.PrintElapsedTime() << "\n";

		// Save displacements field variables
		std::vector<Eigen::MatrixXd> disp_field_variables = solver.SavedDisplacements();

        // Apply shape function correction to the nodal displacements.
        std::cout << Logger::Message("Shape function values application on nodal displacements started...") << std::endl;
        timer.Reset();
        // Application of the shape functions on the displacements.
        solver.ApplyShapeFuncToDisplacements(model, nodal_mmls);
        std::cout << Logger::Message("Shape function values application on nodal displacements completed in: ")
                  << timer.PrintElapsedTime() << "\n";

        // Create the paraview output.
        if (config->VarMap().count("Output.FilePath") && config->VarMap().count("Output.FileName")) {

            timer.Reset();
            std::cout << Logger::Message("Saving results...\n");

            // Model output filename.
            std::string output_filepath = config->RetrieveArgument<std::string>("Output.FilePath");
            std::string output_filename = config->RetrieveArgument<std::string>("Output.FileName");

            // Strip extension from output filename.
            std::string out_ext = output_filename.substr(output_filename.find_last_of("."));
            if (out_ext == ".vtu") { output_filename.erase(output_filename.end()-4, output_filename.end()); }

            ParaviewExporter exporter;
            exporter.CreateVtu(model);

			// Container to store the vtu files paths.
			std::map<float, std::string> vtu_files_paths;

            // Export a .vtu file for each saved time step.
            for (int i = 0; i < solver.SavedDisplacements().size(); ++i) {
                exporter.AddVectorField(solver.SavedDisplacements()[i], "Displacements");
				exporter.AddVectorField(disp_field_variables[i], "FieldVariablesDisplacements");
                exporter.AddVectorField(solver.SavedForces()[i], "Forces");
				exporter.AddVectorField(solver.SavedExternalForces()[i], "ExternalForces");
				std::string file_name = output_filepath + output_filename + std::to_string(i) + ".vtu";
                exporter.Export(file_name);
                exporter.ClearVectorFields();
                std::cout << Logger::Message("Saved ") << i+1 << "/" << solver.SavedDisplacements().size()
                          << " stored model states at: " << output_filepath+output_filename+std::to_string(i)+".vtu\n";

				// Insert file paths in the container.
				boost::filesystem::path file(file_name);
				vtu_files_paths.emplace(solver.SavedTimes()[i], file.filename().string());
            }

            if (config->VarMap().count("Output.AnimationName")) {
                // Create animation.
                std::string animation_filename = output_filepath + config->RetrieveArgument<std::string>("Output.AnimationName");
                exporter.CreatePvdAnimation(vtu_files_paths, animation_filename);
                std::cout << Logger::Message("Saved animation of the stored model states at: ") << animation_filename << std::endl;
            }

            std::cout << Logger::Message("Time for output: ") << timer.PrintElapsedTime() << "\n";
        }

        std::cout << Logger::Message("Simulation terminated successfully.") << std::endl;

        // Release memory from configuration manager.
        delete config;

    }
    catch (const std::invalid_argument &e) {
        std::cerr << e.what() << std::endl;
    }
    catch (const std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
    }
    catch (const std::out_of_range &e) {
       std::cerr << e.what() << std::endl;
    }
    catch (const std::bad_alloc &e) {
       std::cerr << e.what() << std::endl;
    }
    catch (const boost::program_options::error &e) {
        std::cerr << "[Boost program_options error] " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "[ExplicitSim Unknown exception]" << std::endl;
    }



    return EXIT_SUCCESS;
}
