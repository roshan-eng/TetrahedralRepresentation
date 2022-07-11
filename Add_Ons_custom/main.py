"""
Imported a Custom created class Polyhedron
Import Numpy for handling arrays
Import Bio that will be used to parse the pdb file to extract original coordinates (Now built-in as PDB)
"""

import numpy as np
import os
from Bio import PDB
from itertools import combinations
from pathlib import Path
import shutil
import math

from chimerax.core.models import Model
from chimerax.graphics import Drawing


class VertexCalc:

    # Create a method to list all vertices in order that needed to create a face
    @staticmethod
    def face_indices(start_idx):
        return np.ravel(list(combinations(np.arange(start_idx, start_idx + 4), 3)))

    # Create a method to list all vertices in order that needed to create an outline
    @staticmethod
    def outline_indices(start_idx):
        return np.ravel(list(combinations(np.arange(start_idx, start_idx + 4), 2)))


def create_PDB_DATA():
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'PDB_DATA')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    return final_directory


def rm(dir_path):
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)


def provide_model(struct, chain_ids=None, regular=False):
    avg_edge_length = 0

    if regular:
        avg_edge_length = 0
        count = 0
        for model in struct:
            for chain in model:
                for amino_index in range(len(chain)):
                    residue = list(list(chain)[amino_index])

                    # If any residue contains less than or equals 4 atoms then it can't be modeled
                    if len(residue) <= 3:
                        continue

                    e = np.linalg.norm(residue[2].get_coord() - residue[0].get_coord())
                    if (e > 4) or (e < 2):
                        continue
                    else:
                        avg_edge_length += e
                        count += 1

        avg_edge_length /= count

    # Start calculating vertices
    all_vertex = []  # It will contain the viewing coordinates of all the atoms
    all_original_vertex = []  # It will contain the original coordinates of all the atoms
    all_face_indices = []  # It will store the info of vertices to join to form a face
    all_outline_indices = []  # Stores the info of vertices to join to form an outline

    c_alpha_vertex = []  # It will contain the viewing coordinates of c_alpha atoms
    original_c_alpha_vertex = []  # It will contain the original coordinates of c_alpha atoms

    amino_count = 0
    amino_skipped_count = 0  # The amino acids whose model were skipped

    for model in struct:
        for chain in model:
            if chain_ids is None:
                pass
            elif chain.get_id() not in chain_ids:
                continue

            prev_co_cord = None
            for amino_index in range(len(chain)):
                vertex_points = []
                residue = list(list(chain)[amino_index])

                # If any residue contains less than or equals 4 atoms then it can't be modeled
                if len(residue) <= 3:
                    amino_skipped_count += 1
                    continue

                n_cord = np.longdouble(residue[0].get_coord())
                co_cord = np.longdouble(residue[2].get_coord())

                if amino_index != 0:
                    if regular:
                        n_cord = np.longdouble(prev_co_cord)
                    else:
                        n_cord = (n_cord + list(list(chain)[amino_index - 1])[2].get_coord()) * 0.5
                        n_cord = np.longdouble(n_cord)

                if amino_index != len(chain) - 1:
                    co_cord = (co_cord + list(list(chain)[amino_index + 1])[0].get_coord()) * 0.5
                    co_cord = np.longdouble(co_cord)

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

                # Extend the c_beta by a factor to make a regular tetrahedron
                if len(residue) == 4:
                    mid_point_vector = np.random.randint(3, 10, 3)
                    mid_point_vector = np.longdouble(mid_point_vector)
                    mid_point_vector = np.cross(mid_point_vector, (n_cord - co_cord))
                    mid_point_vector *= np.longdouble(
                        np.sqrt(3, dtype=np.longdouble) * 0.5) * norm_co_n / np.linalg.norm(mid_point_vector)
                    c_beta_coord = (co_cord + n_cord) * 0.5 + mid_point_vector
                    c_alpha_cord = np.longdouble(residue[1].get_coord())

                elif len(residue) > 4:
                    c_beta_coord = np.longdouble(residue[4].get_coord())
                    c_alpha_cord = np.longdouble(residue[1].get_coord())

                    co_c_beta = c_beta_coord - co_cord
                    norm_co_c_beta = np.linalg.norm(co_c_beta)

                    move_to_mid_line = (0.5 * norm_co_n - (np.dot(co_n, co_c_beta) / norm_co_n)) * (co_n / norm_co_n)
                    mid_point_vector = c_beta_coord + move_to_mid_line - (co_cord + n_cord) * 0.5
                    mid_point_vector *= np.longdouble(
                        np.sqrt(3, dtype=np.longdouble) * 0.5) * norm_co_n / np.linalg.norm(mid_point_vector)

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

                edges = [norm_co_n, norm_c_beta_n, norm_co_c_beta, norm_co_h, norm_c_beta_h, norm_n_h]
                edge_length = sum(edges) / 6
                # print(norm_co_n, norm_c_beta_n, norm_co_c_beta, norm_co_h, norm_c_beta_h, norm_n_h)

                flag = False
                for e in edges:
                    if (e > 4) or (e < 2):
                        flag = True
                if flag:
                    continue

                vertices = [n_cord, co_cord, c_beta_coord, h_cord]
                original_vertices = [residue[0].get_coord(), residue[2].get_coord(), residue[4].get_coord(),
                                     vertices[-1]]
                face_index = VertexCalc().face_indices(amino_count * 4)
                outline_index = VertexCalc().outline_indices(amino_count * 4)

                amino_count += 1
                all_face_indices.append(face_index)
                c_alpha_vertex.append(c_alpha_cord)
                original_c_alpha_vertex.append((n_cord + co_cord + c_beta_coord + h_cord) / 4)

                for vertex_elem in vertices:
                    all_vertex.append(list(vertex_elem))

                for original_vertex_elements in original_vertices:
                    all_original_vertex.append(original_vertex_elements)

                for outline_elem in outline_index:
                    all_outline_indices.append(outline_elem)

                # Store the distance between c_alpha_cord and other four atoms
                for idx in range(4):
                    dist = np.linalg.norm(vertices[idx] - c_alpha_cord)
                    dist_from_c_alpha.append(dist)

    all_vertex = np.array(all_vertex, np.float32)
    all_face_indices = np.array(all_face_indices, np.int32)

    rmsd = lambda lst, org_lst, N: (sum(map(lambda x, y: (x - y) ** 2, lst, org_lst)) / N) ** 0.5

    RMSD_All = rmsd(all_vertex, all_original_vertex, amino_count)
    RMSD_CA = rmsd(c_alpha_vertex, original_c_alpha_vertex, amino_count)

    return all_vertex, all_vertex, all_face_indices


def tetrahedron_model(pdb_name='1dn3', chain_ids=None, col=None, reg=False):
    t = Drawing('tetrahedrons')

    PDB_path = create_PDB_DATA()

    # Clear all the previous downloaded pdb files
    rm(PDB_path)

    # Download the required pdb file
    pdb_lst = PDB.PDBList()
    native_pdb = pdb_lst.retrieve_pdb_file(str(pdb_name), pdir=PDB_path, file_format='pdb')

    # Parse the PDB file
    parser = PDB.PDBParser()
    io = PDB.PDBIO()
    struct = parser.get_structure(str(pdb_name), Path(PDB_path) / native_pdb)

    # colors to use
    magenta = (255, 0, 255, 150)
    cyan = (0, 255, 255, 150)
    gold = (255, 215, 0, 150)
    bright_green = (70, 255, 0, 150)
    navy_blue = (0, 30, 128, 150)
    red = (255, 0, 0, 150)
    green = (0, 255, 0, 150)
    blue = (0, 0, 255, 150)

    colors = [magenta, cyan, gold, bright_green, navy_blue, red, green, blue]

    if chain_ids is None:
        chain_ids = []
        for model in struct:
            for chain in model:
                chain_ids.append(chain.get_id())

    for i in range(len(chain_ids)):
        va, na, ta = provide_model(struct, chain_ids[i], reg)
        p = t.new_drawing(str(i))
        p.set_geometry(va, na, ta)

        if col is not None and i < len(col):
            p.set_color(np.array((col[i],), np.uint8))

        else:
            p.set_color(colors[i % len(colors)])

    return t
