"""
Imported a Custom created class Polyhedron
Import Numpy for handling arrays
"""

import numpy as np
import os
from itertools import combinations
from pathlib import Path
import shutil
import math

from chimerax.core.models import Model
from chimerax.core.commands import run
from chimerax.graphics import Drawing
from chimerax.model_panel import tool


# colors to use
magenta = (255, 0, 255, 255)
cyan = (0, 255, 255, 255)
gold = (255, 215, 0, 255)
bright_green = (70, 255, 0, 255)
navy_blue = (0, 30, 128, 255)
red = (255, 0, 0, 255)
green = (0, 255, 0, 255)
blue = (0, 0, 255, 255)


def face_indices(start_idx):
    return list(combinations(np.arange(start_idx, start_idx + 4), 3))


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

            for amino_index in range(len(chain.residues)):

                if chain.residues[amino_index] is None:
                    continue

                residue = chain.residues[amino_index].atoms
                if 'CA' != residue.names[1]:
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


def provide_model(model_list, avg_edge_length, model_ids, chain_ids, regular=False):
    # Start calculating vertices
    all_vertex = []  # It will contain the viewing coordinates of all the atoms
    all_original_vertex = []  # It will contain the original coordinates of all the atoms
    all_face_indices = []  # It will store the info of vertices to join to form a face

    c_alpha_vertex = []  # It will contain the viewing coordinates of c_alpha atoms
    original_c_alpha_vertex = []  # It will contain the original coordinates of c_alpha atoms

    amino_count = 0
    amino_skipped_count = 0  # The amino acids whose model were skipped
    # csv_data = [] fields = ['Index', 'N-CO Distance', 'N-CB Distance', 'CO-CB Distance', 'CO-H Distance',
    # 'CB-H Distance', 'N-H Distance', 'N-CO Model Length', 'N-CB Model Length', 'CO-CB Model Length', 'CO-H Model
    # Length', 'CB-H Model Length', 'N-H Model Length', 'Average Model Length']

    for model in model_list:
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

            prev_co_cord = None
            # csv_data.extend([[], [f"Chain: '{chain.chain_id}'"], []])
            # csv_data.append(fields)
            # csv_data.append([])

            for amino_index in range(len(chain.residues)):
                if chain.residues[amino_index] is None:
                    continue

                vertex_points = []
                residue = chain.residues[amino_index].atoms

                n_cord = residue[0].coord
                co_cord = residue[2].coord

                if amino_index != 0 and chain.residues[amino_index - 1] is not None:
                    if regular and prev_co_cord is not None:
                        n_cord = prev_co_cord
                    else:
                        n_cord = (n_cord + chain.residues[amino_index - 1].atoms[2].coord) * 0.5

                if amino_index != len(chain.residues) - 1 and chain.residues[amino_index + 1] is not None:
                    co_cord = (co_cord + chain.residues[amino_index + 1].atoms[0].coord) * 0.5

                co_n = n_cord - co_cord
                norm_co_n = np.linalg.norm(co_n)
                c_beta_coord = None
                c_alpha_cord = None

                if regular:
                    co_n_dir = co_n / norm_co_n
                    co_n = co_n_dir * avg_edge_length
                    co_cord = n_cord - co_n
                    norm_co_n = avg_edge_length
                    prev_co_cord = co_cord

                # If the residue doesn't contains a CA means its and HETATM not an amino acid
                if 'CA' != residue.names[1]:
                    # Remove the assignment of prev_co_coord
                    prev_co_cord = None
                    continue

                # Extend the c_beta by a factor to make a regular tetrahedron
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

                mid_point_vector *= np.longdouble(np.sqrt(3, dtype=np.longdouble) * 0.5) * norm_co_n / np.linalg.norm(
                    mid_point_vector)
                c_beta_coord = (co_cord + n_cord) * 0.5 + mid_point_vector

                centroid = (c_beta_coord + co_cord + n_cord) / 3
                direction = np.cross((c_beta_coord - co_cord), (n_cord - co_cord))
                unit_dir = direction / np.linalg.norm(direction)

                vec = c_alpha_cord - centroid
                cos_theta = np.dot(unit_dir, vec) / np.linalg.norm(vec)

                if cos_theta < 0:
                    # Reverse the unit direction
                    unit_dir *= -1

                H_vector = np.longdouble(
                    np.sqrt(np.longdouble(2) / np.longdouble(3), dtype=np.longdouble)) * norm_co_n * unit_dir
                h_cord = centroid + H_vector

                # Ambiguously High or Low Size
                norm_c_beta_n = np.linalg.norm(c_beta_coord - n_cord)
                norm_co_c_beta = np.linalg.norm(co_cord - c_beta_coord)
                norm_co_h = np.linalg.norm(co_cord - h_cord)
                norm_c_beta_h = np.linalg.norm(c_beta_coord - h_cord)
                norm_n_h = np.linalg.norm(n_cord - h_cord)

                if len(residue) == 4:
                    orinigal_cbeta = c_beta_coord
                else:
                    orinigal_cbeta = residue[4].coord

                vertices = [n_cord, co_cord, c_beta_coord, h_cord]
                original_vertices = [residue[0].coord, residue[2].coord, orinigal_cbeta, vertices[-1]]
                edges = [norm_co_n, norm_c_beta_n, norm_co_c_beta, norm_co_h, norm_c_beta_h, norm_n_h]
                original_edges = [np.linalg.norm(original_vertices[0] - original_vertices[1]),
                                  np.linalg.norm(original_vertices[0] - original_vertices[2]),
                                  np.linalg.norm(original_vertices[1] - original_vertices[2]),
                                  np.linalg.norm(original_vertices[1] - original_vertices[3]),
                                  np.linalg.norm(original_vertices[2] - original_vertices[3]),
                                  np.linalg.norm(original_vertices[0] - original_vertices[3])]

                face_index = face_indices(amino_count * 4)
                amino_count += 1

                all_vertex.extend(vertices)
                all_original_vertex.extend(original_vertices)
                all_face_indices.extend(face_index)
                c_alpha_vertex.append(c_alpha_cord)
                original_c_alpha_vertex.append((n_cord + co_cord + c_beta_coord + h_cord) / 4)
                # csv_data.append([amino_index + 1, *original_edges, *edges, sum(edges) / 6])

    all_vertex = np.array(all_vertex, np.float32)
    all_face_indices = np.array(all_face_indices, np.int32)

    # sqrt_dists = lambda lst, org_lst, N: sum(map(lambda x, y: np.linalg.norm(x - y) ** 2, lst, org_lst))

    # sqrd_All = 0
    # sqrd_CA = 0
    # if amino_count != 0:
    #     sqrd_All = sqrt_dists(all_vertex, all_original_vertex, amino_count)
    #     sqrd_CA = sqrt_dists(c_alpha_vertex, original_c_alpha_vertex, amino_count)

    return all_vertex, all_vertex, all_face_indices


def tetrahedron_model(session, pdb_name='1dn3', model_ids=None, chain_ids=None, reg=True):
    t = Drawing('tetrahedrons')

    # run(session, 'close')
    # run(session, f'open {pdb_name}')

    # Create a CSV file
    # path = os.environ["HOMEPATH"]
    # filename = 'Desktop/model_result.csv'

    colors = [magenta, cyan, gold, bright_green, navy_blue, red, green, blue]

    model_list = session.models.list()

    # Remove undefined models
    for model in model_list:
        try:
            model.chains
        except AttributeError:
            model_list.remove(model)

    if chain_ids is None:
        chains = []
        for model in model_list:
            chains.append([i.chain_id for i in model.chains])

        chain_ids = {model.name: chain_id for (model, chain_id) in zip(model_list, chains)}

    if model_ids is None:
        model_ids = [model.name for model in model_list]

    # Sqrd_all = 0
    # Sqrd_CA = 0
    # count = 0
    # csv_data = []
    avg_edge_length = avg_length(model_list, model_ids, chain_ids)
    va, na, ta = provide_model(model_list, avg_edge_length, model_ids, chain_ids, reg)
    t.set_geometry(va, na, ta)

    ss_ids = []
    for model in model_list:
        if model.name in model_ids:
            ss_ids.extend(model.residues.ss_types)

    vertex_len = len(va)
    vertex_col = []
    for i in range(vertex_len):
        vertex_col.append(colors[ss_ids[i // 4]])
    vertex_col = np.array(vertex_col, np.uint8)
    t.vertex_colors = model_list[0].atoms.colors

    # if col is not None and i < len(col):
    #    p.set_color(col[i])

    # else:
    #    p.set_color(colors[i % len(colors)])

    # rmsd_all = (Sqrd_all / count) ** 0.5
    # rmsd_CA = (Sqrd_CA / count) ** 0.5

    # with open(Path(path) / filename, 'w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerow(['RMSD_All', 'RMSD_CA'])
    #     csv_writer.writerows([[rmsd_all, rmsd_CA], []])
    #     csv_writer.writerows(csv_data)

    m0 = Model('m0', session)
    m0.add([t])
    session.models.add([m0])

    # print("RMSD_ALL: ", rmsd_all, "\n", "RMSD_CA:", rmsd_CA)
    # print("Saved results to: ", Path(path) / filename)


def massing(session):
    pass
