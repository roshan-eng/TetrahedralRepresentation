"""
Import Numpy for handling arrays
Import os for navigating enviroment
Import itertools combinations for calculating triangle mesh vertices
Import Path to navigate through paths
Import csv to write .csv files

Import Chimerax Models to create a model
Import Chimerax run to run command line
Import Drawing to create a tetrahedron drawings
"""

import numpy as np
import os
from itertools import combinations
from pathlib import Path

from chimerax.core.models import Model
from chimerax.core.commands import run
from chimerax.graphics import Drawing


# Function to find an average tetrahedron edge length in case of global regularization
def avg_length(model_list, model_ids, chain_ids):

    avg_edge_length = 0
    count = 0

    for model in model_list:

        # Only specified models
        if model_ids is None:
            pass
        elif model.name not in model_ids:
            continue

        for chain in model.chains:

            # Only specified chains
            if chain_ids is None:
                pass
            elif chain.chain_id not in chain_ids[model.name]:
                continue

            limit = len(chain.residues)
            if limit > 50:
                limit = 50
            for amino_index in range(limit):

                if chain.residues[amino_index] is None:
                    continue

                residue = chain.residues[amino_index].atoms

                # If CA is not in residues means the current residue is of hetroatoms and is not an amino acid
                if 'CA' not in residue.names:
                    continue

                mid_N_point = residue[0].coord
                mid_CO_point = residue[2].coord

                if amino_index != 0 and chain.residues[amino_index - 1] is not None:
                    mid_N_point = (mid_N_point + chain.residues[amino_index - 1].atoms[2].coord) * 0.5

                if amino_index != len(chain.residues) - 1 and chain.residues[amino_index + 1] is not None:
                    mid_CO_point = (mid_CO_point + chain.residues[amino_index + 1].atoms[0].coord) * 0.5

                e = np.linalg.norm(mid_N_point - mid_CO_point)
                avg_edge_length += e
                count += 1

        avg_edge_length /= count

    return avg_edge_length


def provide_model(prev_residue, prev_co_cord, residue, next_residue, avg_edge_length, regular=False):
    
    # Start calculating vertices
    # all_vertex = []                 # It will contain the viewing coordinates of all the atoms
    # # all_original_vertex = []        # It will contain the original coordinates of all the atoms
    # all_face_indices = []           # It will store the info of vertices to join to form a face

    # # c_alpha_vertex = []             # It will contain the viewing coordinates of c_alpha atoms
    # # original_c_alpha_vertex = []    # It will contain the original coordinates of c_alpha atoms

    # amino_count = 0
    # amino_skipped_count = 0         # The amino acids whose model were skipped
    # csv_data = []                   # It will store the data to be written in .csv file
    # fields = ['Index', 
    #             'N-CO Distance', 'N-CB Distance', 'CO-CB Distance', 'CO-H Distance', 'CB-H Distance', 'N-H Distance',
    #             'N-CO Model Length', 'N-CB Model Length', 'CO-CB Model Length', 'CO-H Model Length', 'CB-H Model Length', 'N-H Model Length',
    #             'Average Model Length']

    # for chain in model.chains:

    #     # Only specified chains
    #     if chain_id is None:
    #         pass
    #     elif chain.chain_id not in chain_id:
    #         continue

    #     # A variable to store the previous CO point to place next amino acid joind vertex-to-vertex with previous one
    #     prev_co_cord = None

    #     # Add the Chain ID and the Fields in .csv file
    #     # csv_data.extend([[], [f"Chain: '{chain.chain_id}'"], []])
    #     # csv_data.append(fields)
    #     # csv_data.append([])

    #     for amino_index in range(len(chain.residues)):

    vertex_points = []
    n_cord = residue[0].coord
    co_cord = residue[2].coord

    if prev_residue is not None:
        if regular and prev_co_cord is not None:
            n_cord = prev_co_cord
        else:
            n_cord = (n_cord + prev_residue[2].coord) * 0.5

    if next_residue is not None:
        co_cord = (co_cord + next_residue[0].coord) * 0.5

    co_n = n_cord - co_cord
    norm_co_n = np.linalg.norm(co_n)
    c_beta_coord = None
    c_alpha_cord = None

    if regular:

        # If a case of global regularization then use average edge length as given and store a value to prev_co_cord
        co_n_dir = co_n / norm_co_n
        co_n = co_n_dir * avg_edge_length
        co_cord = n_cord - co_n
        norm_co_n = avg_edge_length
        prev_co_cord = co_cord

    # Shift C-Beta to a position that creates an equilateral triagle with N and CO to make a perfect tetrahedron
    if len(residue) == 4:
        c_alpha_cord = residue[1].coord
        mid_point_vector = np.random.randint(3, 10, 3)
        mid_point_vector = np.cross(mid_point_vector, (n_cord - co_cord))

    elif len(residue) > 4:
        c_beta_coord = residue[4].coord
        c_alpha_cord = residue[1].coord

        co_c_beta = c_beta_coord - co_cord
        norm_co_c_beta = np.linalg.norm(co_c_beta)

        move_to_mid_line = (0.5 * norm_co_n - (np.dot(co_n, co_c_beta) / norm_co_n)) * (co_n / norm_co_n)
        mid_point_vector = c_beta_coord + move_to_mid_line - (co_cord + n_cord) * 0.5

    mid_point_vector *= np.sqrt(3) * 0.5 * norm_co_n / np.linalg.norm(mid_point_vector)
    c_beta_coord = (co_cord + n_cord) * 0.5 + mid_point_vector

    centroid = (c_beta_coord + co_cord + n_cord) / 3
    direction = np.cross((c_beta_coord - co_cord), (n_cord - co_cord))
    unit_dir = direction / np.linalg.norm(direction)

    vec = c_alpha_cord - centroid
    cos_theta = np.dot(unit_dir, vec) / np.linalg.norm(vec)

    if cos_theta < 0:
        # Reverse the unit direction
        unit_dir *= -1

    H_vector = np.sqrt(2 / 3) * norm_co_n * unit_dir
    h_cord = centroid + H_vector

    # Calculate all the edge lengths
    # norm_c_beta_n = np.linalg.norm(c_beta_coord - n_cord)
    # norm_co_c_beta = np.linalg.norm(co_cord - c_beta_coord)
    # norm_co_h = np.linalg.norm(co_cord - h_cord)
    # norm_c_beta_h = np.linalg.norm(c_beta_coord - h_cord)
    # norm_n_h = np.linalg.norm(n_cord - h_cord)

    # Check weather its a case of Glycine or not
    # if len(residue) == 4:
    #     orinigal_cbeta = c_beta_coord
    # else:
    #     orinigal_cbeta = residue[4].coord

    vertices = [n_cord, co_cord, c_beta_coord, h_cord]
    # original_vertices = [residue[0].coord, residue[2].coord, orinigal_cbeta, vertices[-1]]
    # edges = [norm_co_n, norm_c_beta_n, norm_co_c_beta, norm_co_h, norm_c_beta_h, norm_n_h]
    # original_edges = [np.linalg.norm(original_vertices[0] - original_vertices[1]),
    #                     np.linalg.norm(original_vertices[0] - original_vertices[2]),
    #                     np.linalg.norm(original_vertices[1] - original_vertices[2]),
    #                     np.linalg.norm(original_vertices[1] - original_vertices[3]),
    #                     np.linalg.norm(original_vertices[2] - original_vertices[3]),
    #                     np.linalg.norm(original_vertices[0] - original_vertices[3])]

    face_index = np.ravel(list(combinations(np.arange(0, 4), 3)))
    
    vertices = np.array(vertices, np.float32)
    face_index = np.array(face_index, np.int32)

    # # Add all the values to the final list
    # all_vertex.extend(vertices)
    # # all_original_vertex.extend(original_vertices)
    # all_face_indices.append(face_index)
    # # c_alpha_vertex.append(c_alpha_cord)
    # # original_c_alpha_vertex.append((n_cord + co_cord + c_beta_coord + h_cord) / 4)
    # # csv_data.append([amino_index + 1, *original_edges, *edges, sum(edges) / 6])

    return vertices, vertices, face_index, prev_co_cord

    # all_vertex = np.array(all_vertex, np.float32)
    # all_face_indices = np.array(all_face_indices, np.int32)

    # Provide the sum of squared distance between original coordinates and tetrahedron vertices representing them
    # sqrt_dists = lambda lst, org_lst, N: sum(map(lambda x, y: np.linalg.norm(x - y) ** 2, lst, org_lst))

    # sqrd_All = 0
    # sqrd_CA = 0
    # if amino_count != 0:
    #     sqrd_All = sqrt_dists(all_vertex, all_original_vertex, amino_count)
    #     sqrd_CA = sqrt_dists(c_alpha_vertex, original_c_alpha_vertex, amino_count)

    # return all_vertex, all_vertex, all_face_indices


def tetrahedron_model(session, pdb_name = '1dn3', model_ids = None, chain_ids = None, col = None, struct_color = True, reg = True):

    # Creating a drawing which will takes other drawings as its child
    t = Drawing('tetrahedrons')

    # Run the command line to close the current session if open
    #run(session, 'close')
    # Fetch the downloaded file or download if its not present
    #run(session, f'open {pdb_name}')

    # Create a CSV file
    # path = os.environ["HOMEPATH"]
    # filename = 'Desktop/model_result.csv'

    # Colors to use
    magenta = (37, 137, 165, 255)
    navy_blue = (93, 63, 211, 255)
    grey = (170, 170, 170, 255)
    cyan = (0, 255, 255, 255)
    gold = (255, 215, 0, 255)
    bright_green = (70, 255, 0, 255)
    red = (255, 0, 0, 255)
    green = (0, 255, 0, 255)
    blue = (0, 0, 255, 255)

    colors = [magenta, grey, navy_blue, bright_green, navy_blue, red, green, blue]

    model_list = session.models.list()

    # Remove undefined models
    i = 0
    while i < len(model_list):
        try: 
            model_list[i].chains
        except AttributeError: 
            model_list.pop(i)
        else:
            i += 1

    if chain_ids is None:
        chains = []
        for model in model_list:
            chains.append([id.chain_id for id in model.chains])

        chain_ids = {model.name:chain_id for (model, chain_id) in zip(model_list, chains)}

    if model_ids is None:
        model_ids = [model.name for model in model_list]

    # Sqrd_all = 0
    # Sqrd_CA = 0
    # count = 0
    # csv_data = []
    avg_edge_length = avg_length(model_list, model_ids, chain_ids)
    
    for model in model_list:
        if model.name not in model_ids:
            continue

        for chain in model.chains:
            if chain.chain_id not in chain_ids[model.name]:
                continue

            prev_co_cord = None
            for amino_index in range(len(chain.residues)):

                prev_residue = None
                residue = chain.residues[amino_index]
                next_residue = None

                if amino_index > 0:
                    prev_residue = chain.residues[amino_index - 1]
                    if prev_residue is not None:
                        prev_residue = prev_residue.atoms

                if residue is not None:
                    residue = residue.atoms

                if amino_index < len(chain.residues) - 1:
                    next_residue = chain.residues[amino_index + 1]
                    if next_residue is not None:
                        next_residue = next_residue.atoms

                if residue is None:
                    continue

                # If the residue doesn't contains a CA means its and HETATM not an amino acid
                if residue.names[1] != 'CA':
                    # Remove the assignment of prev_co_coord
                    prev_co_cord = None
                    continue

                va, na, ta, prev_co_cord = provide_model(prev_residue, prev_co_cord, residue, next_residue, avg_edge_length, reg)
                p = t.new_drawing(str(i))
                p.set_geometry(va, na, ta)

                # Sqrd_all += sqrd_all
                # Sqrd_CA += sqrd_ca
                # count += amino_count
                # csv_data.extend(data)

                # if not struct_color:
                #     if col is not None and i < len(col):
                #         p.set_color(col[i])

                #     else:
                #         p.set_color(colors[i % len(colors)])

                if struct_color:
                    if chain.residues[amino_index].is_helix:
                        p.set_color(colors[0])
                    elif chain.residues[amino_index].is_strand:
                        p.set_color(colors[1])
                    else:
                        p.set_color(colors[2])

    # Compute the RMSD values
    # rmsd_all = (Sqrd_all / count) ** 0.5
    # rmsd_CA = (Sqrd_CA / count) ** 0.5

    # Write the .csv file
    # with open(Path(path) / filename, 'w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerow(['RMSD_All', 'RMSD_CA'])
    #     csv_writer.writerows([[rmsd_all, rmsd_CA], []])
    #     csv_writer.writerows(csv_data)

    # Create and show the model
    m0 = Model('m0', session)
    m0.add([t])
    session.models.add([m0])

    # print("RMSD_ALL: ", rmsd_all, "\n", "RMSD_CA:", rmsd_CA)
    # print("Saved results to: ", Path(path)/filename)
