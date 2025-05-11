#include <ExplicitSim/explicit_sim.hpp>

#include <cstddef>
#include <string>
#include <fstream>
#include <iomanip>

using namespace ExplicitSim;

int main()
{
    try {

        std::cout << "<< ExplicitSim development testing script. Only for developers. >>\n";

        // Profiling spent time in ExplicitSim
        Timer timer;

        // Set the weak form model.
        WeakModel3D model;
        model.LoadMeshRepresentation("/home/mood/Desktop/cube.feb");
        model.CreateGridRepresentation();

        std::cout << Logger::Message("Model has nodes: ") << model.TetrahedralMesh().Nodes().size() <<
                     " and elements: " << model.TetrahedralMesh().Elements().size() << std::endl;

        // Set support domain.
        SupportDomain sd;
        sd.SetInfluenceNodes(model.TetrahedralMesh().Nodes(), model.TetrahedralMesh().NodesNum());
        sd.SetInfluenceTetrahedra(model.TetrahedralMesh().Elements());

        // Compute the influnce radiuses of the support domain.
        sd.ComputeInfluenceNodesRadiuses(1.6);

        // Setting options for integration points generation.
        IntegOptions options;
        options.is_adaptive_ = false;
        options.adaptive_eps_ = 0.1;
        options.adaptive_level_ = 10;
        options.tetra_divisions_ = 4;
        options.integ_points_per_tetra_ = 4;

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
        model.CreateIntegrationPoints(options, sd);
        std::cout << Logger::Message("Model has integration points: ") << model.IntegrationPoints().PointsNum() << "\n";
        std::cout << Logger::Message("Execution time for integration points generation: ") << timer.PrintElapsedTime() << "\n";

        // Time only for closest nodes.
        timer.Reset();
        // Find influence nodes inidices of integration points.
        auto ip_neigh_ids = sd.ClosestNodesIdsTo(model.IntegrationPoints().Coordinates());

        std::cout << Logger::Message("Execution time for neighbor nodes computation: ") << timer.PrintElapsedTime() << "\n";
        std::cout << Logger::Message("The minimum and maximum number of support nodes: ")
                  << sd.MinSupportNodesIn(ip_neigh_ids) << " - " << sd.MaxSupportNodesIn(ip_neigh_ids) << std::endl;

        // Assign a neo-hookean material to the model.
        NeoHookean material;
        material.SetPointsNumber(model.IntegrationPoints().Coordinates().size());

        // Set material parameters.
        material.SetDensity(1000.);
        material.SetYoungModulus(3000.);
        material.SetPoissonRatio(0.49);

        // Compute elastic parameters and wave speed.
        material.ComputeLameLambdaMu();
        material.ComputeBulkModulus();
        material.ComputeWaveSpeed();

        // Time only for shape functions.
        timer.Reset();

        // Shape Functions
        Mmls3d mmls3d;
        mmls3d.SetBasisFunctionType("quadratic");
        mmls3d.SetExactDerivativesMode(true);
        mmls3d.ComputeShFuncAndDerivs(model.TetrahedralMesh().Nodes(), model.TetrahedralMesh().NodesNum(),
                                          model.IntegrationPoints().Coordinates(), (uint32_t)model.IntegrationPoints().Coordinates().size(),
                                          ip_neigh_ids, sd.InfluenceNodesRadiuses());

        std::cout << Logger::Message("Execution time for shape function generation: ") << timer.PrintElapsedTime() << "\n";

		// Find influence nodes inidices of model nodes.
		auto nodal_neigh_ids = sd.ClosestNodesIdsTo(model.TetrahedralMesh().NodeCoordinates());

		Mmls3d nodal_mmls;
		nodal_mmls.SetBasisFunctionType(mmls3d.BaseFunctionType());
		nodal_mmls.SetExactDerivativesMode(mmls3d.UseExactDerivatives());
		nodal_mmls.ComputeShFuncAndDerivs(model.TetrahedralMesh().Nodes(), model.TetrahedralMesh().NodesNum(),
			model.TetrahedralMesh().NodeCoordinates(), model.TetrahedralMesh().NodesNum(),
			nodal_neigh_ids, sd.InfluenceNodesRadiuses());

        // The dynamic relaxation properties.
        DynRelaxProp dr;
		dr.SetLoadTime(5);
        dr.SetEquilibriumTime(10);
        dr.SetLoadConvRate(0.999);
        dr.SetAfterLoadConvRate(0.99);
        dr.SetStopUpdateConvRateStepsNum(2000);
        dr.SetConvRateDeviation(0.0001);
        dr.SetForceDispUpdateStepsNum(200);
        dr.SetStableConvRateStepsNum(20);
        dr.SetConvRateStopDeviation(0.002);
        dr.SetStopConvRateError(0.2);
        dr.SetStopAbsError(0.00001);
        dr.SetStopStepsNum(100);

        // Solve the model explicitly with MTLED.
        Mtled solver;

        // Compute the time steps.
        solver.ComputeTimeSteps(material.WaveSpeed(), ip_neigh_ids, mmls3d);

        // Compute the mass of the model.
        model.ComputeMass(material.Density(), solver.TimeSteps(), solver.MaxStep(),
                          ip_neigh_ids, true);

        solver.ComputeStableStep(model.IsMassScaled(), 1.5);

        // Set steps for progress save.
        solver.SetSaveProgressSteps(500);

        std::cout << Logger::Message("Minimum time step: ") << solver.MinStep() << " s\n";
        std::cout << Logger::Message("Maximum time step: ") << solver.MaxStep() << " s\n";
        std::cout << Logger::Message("Stable time step: ") << solver.StableStep() << " s\n";

		dr.ComputeStepsNum(solver.StableStep());
		std::cout << Logger::Message("The loading steps number: ") << dr.LoadStepsNum() << std::endl;
		std::cout << Logger::Message("The dynamic relaxation equilibrium steps number: ") << dr.EquilibriumStepsNum() << std::endl;

        // Compute total time steps for solution.
        solver.ComputeTotalTimeStepsNum(dr.LoadStepsNum(), dr.EquilibriumStepsNum());
        std::cout << Logger::Message("The total explicit solution time steps number: ") << solver.TotalTimeStepsNum() << std::endl;

        // Time only for conditions imposition.
        timer.Reset();

        // Initialize the boundary and loading conditions handler.
        ConditionsHandler cond_handler(model.TetrahedralMesh().NodesNum(), &model.TetrahedralMesh().NodeSets());

        cond_handler.AddDirichlet(1, 0, 0, "dx_0");
        cond_handler.AddDirichlet(0, 1, 0, "dy_0");
        cond_handler.AddDirichlet(0, 0, 1, "dz_0");
        cond_handler.AddLoading(0, 0, 0, 1, 0, 0, 0.5, "Displaced");
        cond_handler.vExtractConstrainedDOFs();

        std::cout << Logger::Message("Execution time for conditions initialization: ") << timer.PrintElapsedTime() << "\n";

        // Time only for explicit solution.
        timer.Reset();

        std::cout << Logger::Message("Starting the explicit solution of the model...") << std::endl;

        // Solving with explicit dynamics.
        solver.Solve(model, ip_neigh_ids, cond_handler, mmls3d, material, dr, true, nodal_mmls);

        solver.ApplyShapeFuncToDisplacements(model, nodal_mmls);

        std::cout << Logger::Message("Execution time for explicit solution: ") << timer.PrintElapsedTime() << "\n";

        timer.Reset();
        std::cout << Logger::Message("Saving results...\n");

        // Model output filename.
        std::string output_filepath = "/home/mood/Desktop/ExpSim_exper/output/cube/";
        std::string output_filename = "state.vtu";

        // Strip extension from output filename.
        std::string out_ext = output_filename.substr(output_filename.find_last_of("."));
        if (out_ext == ".vtu") { output_filename.erase(output_filename.end()-4, output_filename.end()); }

        ParaviewExporter exporter;
        exporter.CreateVtu(model);

		// Container to store the vtu files paths.
		std::map<float, std::string> vtu_files_paths;

        // Export a .vtu file for each saved time step.
        for (int i = 0; i != solver.SavedDisplacements().size(); ++i) {
             exporter.AddVectorField(solver.SavedDisplacements()[i], "Displacements");
             exporter.AddVectorField(solver.SavedForces()[i], "Forces");
			 exporter.AddVectorField(solver.SavedExternalForces()[i], "ExternalForces");
             exporter.Export(output_filepath+output_filename+std::to_string(i)+".vtu");
             exporter.ClearVectorFields();
             std::cout << Logger::Message("Saved ") << i+1 << "/" << solver.SavedDisplacements().size()
                       << " stored model states at: " << output_filepath+output_filename+std::to_string(i)+".vtu\n";

			 // Insert file paths in the container.
			 vtu_files_paths.emplace((float)i, output_filepath + output_filename + std::to_string(i) + ".vtu");
        }

        // Create animation.
        std::string animation_filename = output_filepath + "compression.pvd";
        exporter.CreatePvdAnimation(vtu_files_paths, animation_filename);
        std::cout << Logger::Message("Saved animation of the stored model states at: ") << animation_filename << std::endl;

        std::cout << Logger::Message("Time for output: ") << timer.PrintElapsedTime() << "\n";

        std::ofstream test_res("/home/mood/Desktop/test_res.txt");
        test_res << std::setprecision(15) << solver.SavedDisplacements().back() << std::endl;
        test_res.close();

        std::cout << std::setprecision(15) << solver.SavedDisplacements().back() << std::endl;


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
    catch (...) {
        std::cerr << "[ExplicitSim Unknown exception]" << std::endl;
    }

    return EXIT_SUCCESS;
}
