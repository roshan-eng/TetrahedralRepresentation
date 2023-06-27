"""
Imported a Custom created class Polyhedron
Import Numpy for handling arrays
"""
import bisect
import os
import numpy as np
import alphashape
import trimesh
from itertools import permutations, combinations
from pathlib import Path
import shutil
import math

from chimerax.core.models import Model
from chimerax.core.commands import run
from chimerax.graphics import Drawing
from chimerax.model_panel import tool
from chimerax.surface import calculate_vertex_normals

# Constants
ht3 = 0.81649658092772603273242802490196379732198249355222337614423086
ht2 = 0.86602540378443864676372317075293618347140262690519031402790349


class Tetra:

    def __init__(self, session):
        self.all_points = None
        self.session = session
        self.model_list = None
        self.t = Model('model_tetra', self.session)      # The Drawing for Protein Chain
        self.va = []                            # Matrix of all the points in forming Protein Chain Model with Tetrahedrons
        self.ta = []                            # Matrix of all the faces in forming Protein Chain Model with Tetrahedrons
        self.massing_vertices = []              # Matrix of all the faces in forming Protein Chain Massed Model with Tetrahedrons
        self.avg_edge_length = 0                # The final constant edge length of Tetrahedrons forming Protein Chain Model

        self.model_list = self.session.models.list()    # All the protein models need to form the final Tetrahedron model
        for model in self.model_list:
            try:
                model.chains
            except AttributeError:
                self.model_list.remove(model)

    # Get the Final Constant Tetrahedron Edge Length as average of all CO-N Bond Length over all the present amino acids
    def avg_length(self):
        count = 0
        for model in self.model_list:
            for chain in model.chains:
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
                    self.avg_edge_length += e
                    count += 1
            self.avg_edge_length /= count

    # Filling the self.va and self.ta matrix with required values using the protein amino acids
    def provide_model(self, regular=True):
        amino_count = 0
        amino_skipped_count = 0
        c_alpha_vertex = []
        all_original_vertex = []
        original_c_alpha_vertex = []

        for model in self.model_list:
            for chain in model.chains:
                chain_vertex = []

                prev_co_cord = None
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
                        co_n = co_n_dir * self.avg_edge_length
                        co_cord = n_cord - co_n
                        norm_co_n = self.avg_edge_length
                        prev_co_cord = co_cord
                    if 'CA' != residue.names[1]:
                        prev_co_cord = None
                        continue
                    if len(residue) == 4:
                        c_alpha_cord = residue[1].coord
                        mid_vec = n_cord - co_cord
                        mid_point_vector = np.array([-1 / mid_vec[0], 1 / mid_vec[1], 0])
                    elif len(residue) > 4:
                        c_beta_coord = residue[4].coord
                        c_alpha_cord = residue[1].coord
                        co_c_beta = c_beta_coord - co_cord
                        norm_co_c_beta = np.linalg.norm(co_c_beta)
                        move_to_mid_line = (0.5 * norm_co_n - (np.dot(co_n, co_c_beta) / norm_co_n)) * (
                                    co_n / norm_co_n)
                        mid_point_vector = c_beta_coord + move_to_mid_line - (co_cord + n_cord) * 0.5
                    mid_point_vector *= ht2 * norm_co_n / np.linalg.norm(mid_point_vector)
                    c_beta_coord = (co_cord + n_cord) * 0.5 + mid_point_vector
                    centroid = (c_beta_coord + co_cord + n_cord) / 3
                    direction = np.cross((n_cord - co_cord), (c_beta_coord - co_cord))
                    unit_dir = direction / np.linalg.norm(direction)
                    # vec = c_alpha_cord - centroid
                    # cos_theta = np.dot(unit_dir, vec) / np.linalg.norm(vec)
                    # if cos_theta < 0:
                    #     unit_dir *= -1
                    H_vector = ht3 * norm_co_n * unit_dir
                    h_cord = centroid + H_vector
                    norm_c_beta_n = np.linalg.norm(c_beta_coord - n_cord)
                    norm_co_c_beta = np.linalg.norm(co_cord - c_beta_coord)
                    norm_co_h = np.linalg.norm(co_cord - h_cord)
                    norm_c_beta_h = np.linalg.norm(c_beta_coord - h_cord)
                    norm_n_h = np.linalg.norm(n_cord - h_cord)
                    if len(residue) == 4:
                        original_cb = c_beta_coord
                    else:
                        original_cb = residue[4].coord
                    vertices = [n_cord, n_cord, n_cord, co_cord, co_cord, co_cord,
                                c_beta_coord, c_beta_coord, c_beta_coord, h_cord, h_cord, h_cord]
                    original_vertices = np.array([residue[0].coord, residue[2].coord, original_cb, vertices[-1]])
                    edges = np.array([norm_co_n, norm_c_beta_n, norm_co_c_beta, norm_co_h, norm_c_beta_h, norm_n_h])

                    chain_vertex.append(vertices)
                    c_alpha_vertex.append(c_alpha_cord)
                    all_original_vertex.extend(original_vertices)
                    original_c_alpha_vertex.append((n_cord + co_cord + c_beta_coord + h_cord) / 4)
                    amino_count += 1

                chain_vertex = np.array(chain_vertex, np.float32)
                self.va.append(chain_vertex)

    # Function generating the Drawing with all the matrix points in self.va and self.ta
    def tetrahedron_model(self, chains=False, pdb_name='1dn3', reg=True, seq=False):
        if not chains:
            i = 0
            chains = []
            for model in self.model_list:
                for x in model.chains:
                    chains.append(i)
                    i += 1

        self.avg_length()
        self.provide_model(reg)

        # Remove the Protein Chain Model for the part needed to massed
        va = []
        for i in chains:
            ta = []
            amino = 0
            va = np.array(self.va[i], np.float32)

            for a in self.va[i]:
                e = amino * 12
                ta.extend([[e, e + 3, e + 6], [e + 1, e + 7, e + 9], [e + 2, e + 4, e + 10], [e + 5, e + 8, e + 11]])
                amino += 1

            sub_model = Model("chain" + chr(65 + i), self.session)
            va = np.reshape(va, (va.shape[0] * va.shape[1], va.shape[2]))
            ta = np.array(ta, np.int32)
            va_norm = calculate_vertex_normals(va, ta)

            sub_model.set_geometry(va, va_norm, ta)
            self.t.add(sub_model)

        # Feed the matrix coordinates into the Drawing "t"
        self.session.models.add([self.t])

    # Generate Massing
    # Unit is the time of edge length Tetrahedron to be used in reference to what is used in Protein Chain
    def massing(self, chains=False, unit=1, refinement=1):
        if not chains:
            self.tetrahedron_model()
            return

        i = 0
        tetra_chains = []
        massing_chains = []
        mass_chain_objects = []
        for model in self.model_list:
            for ch in model.chains:
                if (model not in chains.keys()) and (ch.chain_id not in chains[model.name]):
                    tetra_chains.append(i)
                else:
                    massing_chains.append(i)
                    mass_chain_objects.append(ch)

                i += 1

        print(tetra_chains)
        print(massing_chains)
        self.tetrahedron_model(chains=tetra_chains)

        model_mass = Model("model_mass", self.session)                    # Drawing for massing Tetrahedron Model
        for (index, ch) in zip(massing_chains, mass_chain_objects):
            v = []
            for amino in ch.residues:
                if amino:
                    v.extend(amino.atoms.coords)

            # Mesh Boundary Outer for Massing used for cases when to stop creating massing Tetrahedrons
            mesh = alphashape.alphashape(v, refinement * 0.1)

            #  Function which becomes negative if the given coordinate is outside the Massing Boundary
            inside = lambda ms: trimesh.proximity.ProximityQuery(ms).signed_distance
            count = 0

            # Create the first tetrahedron
            edge_length = self.avg_edge_length * unit
            pt = self.va[index][0]

            pt1 = pt[0]
            pt2 = pt[0] + (pt[3] - pt[0]) * edge_length / np.linalg.norm(pt[3] - pt[0])
            pt3 = pt[0] + (pt[6] - pt[0]) * edge_length / np.linalg.norm(pt[6] - pt[0])
            pt4 = pt[0] + (pt[9] - pt[0]) * edge_length / np.linalg.norm(pt[9] - pt[0])
            centroid = (pt1 + pt2 + pt3 + pt4) / 4

            self.massing_vertices = [[pt1, pt1, pt1, pt2, pt2, pt2, pt3, pt3, pt3, pt4, pt4, pt4]]
            e = count * 12
            faces = [[e, e + 3, e + 6], [e + 1, e + 7, e + 9], [e + 2, e + 4, e + 10], [e + 5, e + 8, e + 11]]
            count += 1

            q = [[tuple(pt1), tuple(pt2), tuple(pt3), tuple(pt4)]]
            visited = {tuple((int(centroid[0]), int(centroid[1]), int(centroid[2])))}

            # Grow the Tetrahedrons using fixed set of equations to get the required pattern
            depth = 10**5
            while q:
                if depth < 0:
                    print("DEPTH EXCEEDED", len(q))
                    break
                depth -= 1

                # Create new four tetrahedrons
                prev_tetra = list(q.pop())
                combine = combinations(prev_tetra, 3)
                for p in combine:
                    for x in prev_tetra:
                        if x not in p:
                            p += (x,)
                            break

                    pt1, pt2, pt3, pt4 = p
                    pt1, pt2, pt3, pt4 = np.array(pt1), np.array(pt2), np.array(pt3), np.array(pt4)

                    # Set of equations to Generate the new tetrahedron from the previous one
                    p1 = (pt1 + pt2) / 2
                    p3 = p1 + (pt4 - pt3)
                    p2 = pt4 + (pt2 - pt1) / 2
                    p4 = pt4 + (pt1 - pt2) / 2
                    centroid = (p1 + p2 + p3 + p4) / 4

                    #  Check if it's out of boundary
                    if inside(mesh)((centroid,)) < -3 * unit:
                        continue

                    # Visited or Not
                    t = tuple((int(centroid[0]), int(centroid[1]), int(centroid[2])))
                    if t not in visited:
                        self.massing_vertices.append([p1, p1, p1, p2, p2, p2, p3, p3, p3, p4, p4, p4])
                        e = count * 12
                        faces.extend([[e, e + 3, e + 6], [e + 1, e + 7, e + 9], [e + 2, e + 4, e + 10], [e + 5, e + 8, e + 11]])
                        count += 1

                        q.append([tuple(p1), tuple(p2), tuple(p3), tuple(p4)])
                        visited.add(t)

            # Refine the massing Tetras within the Boundary
            x = 0
            while x < len(self.massing_vertices):
                pt1, pt2, pt3, pt4 = self.massing_vertices[x][0], self.massing_vertices[x][3], \
                                     self.massing_vertices[x][6], self.massing_vertices[x][9]
                centroid = (pt1 + pt2 + pt3 + pt4) / 4
                if inside(mesh)((centroid,)) < 0:
                    self.massing_vertices.pop(x)
                else:
                    x += 1

            # Refine the massing Tetras within the Boundary
            mass_v = np.array(self.massing_vertices)
            faces = np.array(faces[:len(mass_v) * 4], np.int32)
            self.massing_vertices = np.reshape(mass_v, (mass_v.shape[0] * mass_v.shape[1], mass_v.shape[2]))

            chainModel = Model("chain" + chr(65 + index), self.session)
            '''
            va_norm = calculate_vertex_normals(va, self.ta)
            self.t.set_geometry(va, va_norm, self.ta)
            self.session.models.add([self.t])
            '''

            mass_v_norm = calculate_vertex_normals(self.massing_vertices, faces)
            chainModel.set_geometry(self.massing_vertices, mass_v_norm, faces)
            model_mass.add([chainModel])

        self.session.models.add([model_mass])
