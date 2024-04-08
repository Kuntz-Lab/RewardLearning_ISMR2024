#!/usr/bin/env python3

import numpy as np
import trimesh
import os
import pickle
os.chdir('/home/dvrk/fTetWild/build') 

def create_tet(mesh_dir, object_name):
    """Convert .stl mesh file to .tet mesh file"""  
    
    # STL to mesh
    # install fTetWild here https://github.com/wildmeshing/fTetWild

    stl_mesh_path = os.path.join(mesh_dir, object_name+'.stl')
    fTetwild_mesh_path = os.path.join(mesh_dir, object_name + '.mesh')
    os.system("./FloatTetwild_bin -o " + fTetwild_mesh_path + " -i " + stl_mesh_path)


    # Mesh to tet:
    mesh_file = open(os.path.join(mesh_dir, object_name + '.mesh'), "r")
    tet_output = open(
        os.path.join(mesh_dir, object_name + '.tet'), "w")

    # Parse .mesh file
    mesh_lines = list(mesh_file)
    mesh_lines = [line.strip('\n') for line in mesh_lines]
    vertices_start = mesh_lines.index('Vertices')
    num_vertices = mesh_lines[vertices_start + 1]

    vertices = mesh_lines[vertices_start + 2:vertices_start + 2
                        + int(num_vertices)]

    tetrahedra_start = mesh_lines.index('Tetrahedra')
    num_tetrahedra = mesh_lines[tetrahedra_start + 1]
    tetrahedra = mesh_lines[tetrahedra_start + 2:tetrahedra_start + 2
                            + int(num_tetrahedra)]

    print("# Vertices, # Tetrahedra:", num_vertices, num_tetrahedra)

    # Write to tet output
    tet_output.write("# Tetrahedral mesh generated using\n\n")
    tet_output.write("# " + num_vertices + " vertices\n")
    for v in vertices:
        tet_output.write("v " + v + "\n")
    tet_output.write("\n")
    tet_output.write("# " + num_tetrahedra + " tetrahedra\n")
    for t in tetrahedra:
        line = t.split(' 0')[0]
        line = line.split(" ")
        line = [str(int(k) - 1) for k in line]
        l_text = ' '.join(line)
        tet_output.write("t " + l_text + "\n")

### Examples of generating deformable objects' .tet mesh files from .stl files. Use trimesh to generate .stl mesh
# mesh_dir = '/home/dvrk/dvrk_ws/src/dvrk_env/shape_servo_control/src/teleoperation/random_stuff'
# object_name = "obstacle_1"


# mesh1 = trimesh.creation.annulus(r_min=0.1, r_max=0.13, height=0.1)
# mesh1 = trimesh.intersections.slice_mesh_plane(mesh=mesh1, plane_normal=[0,-1,0], plane_origin=[0,0.02,0], cap=True)

# # mesh1 = trimesh.creation.cylinder(radius=0.08, height=0.2) # cylinder
# # mesh2 = trimesh.creation.box((0.1, 0.1, 0.04))      # box
# mesh3 = trimesh.creation.icosphere(radius = 0.02)   # hemisphere
# mesh4 = trimesh.creation.cone(radius=0.08, height=0.2)

# meshes = [mesh1, mesh3, mesh4]
# trimesh.Scene(meshes).show()

# mesh1.export(os.path.join(mesh_dir, object_name+'.obj'))
# create_tet(mesh_dir, object_name) 

mesh_dir = '/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation/random_stuff'
#object_names = ["small_sphere_r0.005"]
#object_names = ["support_box_thin"]
object_names = ['new_push_box', 'new_push_target', 'new_cone']

# mesh1 = trimesh.creation.cylinder(radius=0.1, height=0.5, sections=None, segment=None, transform=None)
############### push ###################
# mesh1 = trimesh.creation.cone(radius=0.05, height=0.05, sections=None, transform=None)
# mesh1 = trimesh.creation.box(extents=[0.025,0.05, 0.1], transform=None)
############## new push ################
mesh3 = trimesh.creation.cone(radius=0.02, height=0.08, sections=None, transform=None)
mesh2 = trimesh.creation.box(extents=[0.04,0.04, 0.08], transform=None)
mesh1 = trimesh.creation.box(extents=[0.04,0.04, 0.05], transform=None)
#mesh1 = trimesh.creation.icosphere(radius=0.01)
#mesh1 = trimesh.creation.box((2, 2, 0.05)) 
meshes = [ mesh1, mesh2, mesh3]
trimesh.Scene(meshes).show()

assert(len(object_names) == len(meshes))

trimesh.Scene(meshes).show()
for i, object_name in enumerate(object_names):
    meshes[i].export(os.path.join(mesh_dir, object_name+'.obj')) #export
