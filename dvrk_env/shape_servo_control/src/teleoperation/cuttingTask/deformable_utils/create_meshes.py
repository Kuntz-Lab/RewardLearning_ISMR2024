import numpy as np
import trimesh
import os
import pickle
import random
import open3d
from copy import deepcopy
import argparse

def create_tet(mesh_dir, object_name):
    # STL to mesh
    import os
    os.chdir('/home/dvrk/fTetWild/build') 
    mesh_path = os.path.join(mesh_dir, object_name+'.stl')
    save_fTetwild_mesh_path = os.path.join(mesh_dir, object_name + '.mesh')
    os.system("./FloatTetwild_bin -o " + save_fTetwild_mesh_path + " -i " + mesh_path + " >/dev/null")


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

    # print("# Vertices, # Tetrahedra:", num_vertices, num_tetrahedra)

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


def create_handcrafted_box_mesh_datatset(save_mesh_dir, save_pickle=True, seed=None, vis=False):
    if seed is not None:
        np.random.seed(seed)

    
    primitive_dict = {'count':0}
    
    youngs = 1000
    height, width, thickness = (0.2, 0.2, 0.02)
    base_radius = 0.01
    base_thickness = 0.001  #0.005

    # Create and save object and base meshes
    mesh_obj = trimesh.creation.box((height, width, thickness))  
    #mesh_base = trimesh.creation.icosphere(radius = base_radius) 
    mesh_base = trimesh.creation.cylinder(base_radius, height=base_thickness) 

   

    if vis:
        coordinate_frame = trimesh.creation.axis()  
        coordinate_frame.apply_scale(0.05)    
        copied_mesh_obj = deepcopy(mesh_obj)
        T = trimesh.transformations.translation_matrix([0., 0, -0.1])
        copied_mesh_obj.apply_transform(T)
        meshes = [copied_mesh_obj, mesh_base]
        trimesh.Scene(meshes+[coordinate_frame]).show()
        #trimesh.Scene(meshes).show()

    
    object_name = f"box"
    base_name = f"base"
    mesh_obj.export(os.path.join(save_mesh_dir, object_name+'.stl'))
    create_tet(save_mesh_dir, object_name)
    mesh_base.export(os.path.join(save_mesh_dir, base_name+'.obj'))
    
    primitive_dict[object_name] = {'height': height, 'width': width, 'thickness': thickness, 'youngs': youngs}
    primitive_dict[base_name] = {"thickness": base_thickness}
    primitive_dict['count'] += 1
    
    if save_pickle == False:   
        return primitive_dict

    data = primitive_dict
    with open(os.path.join(save_mesh_dir, "primitive_dict.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--object_mesh_path', type=str, help="where you want to save the mesh")
    
    args = parser.parse_args()
    mesh_dir = args.object_mesh_path
    os.makedirs(mesh_dir,exist_ok=True)
    create_handcrafted_box_mesh_datatset(mesh_dir,  vis=True)