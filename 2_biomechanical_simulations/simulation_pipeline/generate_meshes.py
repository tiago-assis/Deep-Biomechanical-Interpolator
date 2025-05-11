#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:25:07 2021

@author: yue
"""

# Edited and optimized by: 
# Tiago Assis
# Faculty of Sciences, University of Lisbon
# March 2025

import os
import gmsh
import math
import sys
import meshio


def generate_mesh(model_file: str) -> None:
    """
    Generates and optimizes 3D and surface meshes from an input STL file, 
    and converts the resulting meshes into multiple formats for simulation and visualization.

    Parameters:
        model_file (str): Path to the input file representing the surface geometry to be meshed (.stl).
    """
    gmsh.initialize()

    # Merge an STL mesh that we would like to remesh
    gmsh.merge(model_file)

    # Angle between two triangles above which an edge is considered as sharp:
    angle = 40

    # For complex geometries, patches can be too complex, too elongated or too large
    # to be parametrized; setting the following option will force the creation of
    # patches that are amenable to reparametrization:
    forceParametrizablePatches = True

    # For open surfaces include the boundary edges in the classification process:
    includeBoundary = False

    # Force curves to be split on given angle:
    curveAngle = 120

    gmsh.model.mesh.classifySurfaces(angle * math.pi / 180., includeBoundary,
                                    forceParametrizablePatches,
                                    curveAngle * math.pi / 180.)

    # Create a geometry for all the discrete curves and surfaces in the mesh, by
    # computing a parametrization for each one
    gmsh.model.mesh.createGeometry()

    # Create a volume from all the surfaces
    s = gmsh.model.getEntities(2)
    l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
    gmsh.model.geo.addVolume([l])

    gmsh.model.geo.synchronize()

    # We specify element sizes imposed by a size field
    f = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f, "F", "4")
    gmsh.model.mesh.field.setAsBackgroundMesh(f)

    # Aditional mesh optimization steps
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.5)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize()
    gmsh.write('./1_brain_meshes/brain_tetravolume.msh')

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize()
    gmsh.write("./1_brain_meshes/brain_triangsurface.msh")
    gmsh.finalize()

    # Convert .msh to .vtu so we can visualise it in Paraview/Slicer
    msh = meshio.read("./1_brain_meshes/brain_tetravolume.msh")
    nodes = msh.points
    tetras = msh.cells_dict["tetra"]
    meshio.write_points_cells("./1_brain_meshes/brain_tetravolume.vtu", nodes, [("tetra",tetras)])


    msh = meshio.read("./1_brain_meshes/brain_tetravolume.vtu")
    # Generate necessary .inp file for the simulation
    meshio.write("./1_brain_meshes/brain_tetravolume.inp", msh)

    msh = meshio.read("./1_brain_meshes/brain_triangsurface.msh")
    nodes = msh.points
    triangles = msh.cells_dict["triangle"]

    meshio.write_points_cells("./1_brain_meshes/brain_triangsurface.vtu", nodes, [("triangle",triangles)])


if __name__ == '__main__':
    model_file = sys.argv[1]
    assert os.path.exists(model_file), "The path to the model file does not exist."
    assert model_file.lower().endswith(".stl"), "The model file must be in .stl format."
    generate_mesh(model_file)
