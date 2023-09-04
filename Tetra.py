import bisect
import os
import numpy as np
from itertools import permutations, combinations
from pathlib import Path
import shutil
import math

from chimerax.core.commands import CmdDesc
from chimerax.core.models import Model
from chimerax.core.commands import run
from chimerax.graphics import Drawing
from chimerax.model_panel import tool
from chimerax.surface import calculate_vertex_normals

from chimerax.core.commands import ListOf, SetOf, TupleOf, Or, RepeatOf, BoolArg, IntArg

# Constants
ht3 = 0.81649658092772603273242802490196379732198249355222337614423086
ht2 = 0.86602540378443864676372317075293618347140262690519031402790349

class Tetra:

    def __init__(self, session):

        """
        Initialize the Session, List of all the models in use, Edge-length of Tetrahedrons
        and the arrays to store all the required vertices and faces.
        """

        self.session = session
        self.model_list = []
        self.edge_length = 0
        self.vertices = []
        self.faces = []

        # Remove the PseudoModels from the Model list as they have no Chain components.
        for model in self.session.models.list():
            try:
                model.chains
            except AttributeError:
                print("PesudoModels Found !!!")
            else:
                self.model_list.append(model)

    """
    Calculation of the required Edge-length of tetrahedrons to use as the average length of CO-N bonds
    over the whole model to minimize the deviation from original vertices.
    """
    def calculate_edge_length(self):

        amino_count = 0
        for model in self.model_list:
            for chain in model.chains:
                for amino_index in range(len(chain.residues)):
                    # If the residues is empty then continue. Likely the case of metal or water components.
                    if chain.residues[amino_index] is None:
                        continue

                    residue = chain.residues[amino_index]

                    # If the residue is not an amino acid skip it.
                    if residue.polymer_type != residue.PT_AMINO:
                        continue

                    # Take the reference of original CO and N coordinates and deviate them to form a regular tetrahedron.
                    n_atom = residue.find_atom('N')
                    co_atom = residue.find_atom('C')
                    if n_atom is None or co_atom is None:
                        continue

                    mid_N_point = n_atom.coord
                    mid_CO_point = co_atom.coord

                    # Consider the mid-points of CO-N bonds as a vertex to form a continuous joined chains of tetrahedrons.
                    # Condition check for edge cases of first and last amino acids in a chain.
                    if amino_index != 0 and chain.residues[amino_index - 1] is not None:
                        mid_N_point = (mid_N_point + chain.residues[amino_index - 1].atoms[2].coord) * 0.5

                    if amino_index != len(chain.residues) - 1 and chain.residues[amino_index + 1] is not None:
                        mid_CO_point = (mid_CO_point + chain.residues[amino_index + 1].atoms[0].coord) * 0.5

                    e = np.linalg.norm(mid_N_point - mid_CO_point)
                    self.edge_length += e
                    amino_count += 1

            self.edge_length /= amino_count

    """
    Calculation of vertices that represents the complete Tetrahedron Structure.
    The original coordinates were deviated from to form a regular Tetrahedron where the four edges of Tetrahedrons
    represents the N, CO, CB and H respectively.
    """
    def calculate_vertices(self):

        amino_count = 0
        for model in self.model_list:
            for chain in model.chains:
                chain_vertex = []
                prev_CO_coordinate = None

                for index in range(len(chain.residues)):
                    # If the residues is empty then continue. Likely the case of metal or water components.
                    if chain.residues[index] is None:
                        continue

                    residue = chain.residues[index]
                    resatoms = residue.atoms

                    n_atom = residue.find_atom('N')
                    co_atom = residue.find_atom('C')
                    if n_atom is None or co_atom is None:
                        continue

                    N_coordinate = n_atom.coord
                    CO_coordinate = co_atom.coord

                    # Consider the mid-points of CO-N bonds as a vertex to form a continuous joined chains of tetrahedrons.
                    # Condition check for edge cases of first and last amino acids in a chain.
                    if index != 0 and chain.residues[index - 1] is not None:
                        if prev_CO_coordinate is not None:
                            N_coordinate = prev_CO_coordinate
                        else:
                            N_coordinate = (N_coordinate + chain.residues[index - 1].atoms[2].coord) * 0.5

                    if index != len(chain.residues) - 1 and chain.residues[index + 1] is not None:
                        CO_coordinate = (CO_coordinate + chain.residues[index + 1].atoms[0].coord) * 0.5

                    CO_N_vector = N_coordinate - CO_coordinate
                    CO_N_normal = np.linalg.norm(CO_N_vector)

                    CB_coordinate = None
                    CA_coordinate = None

                    # Coordinates for a regular Tetrahedron.
                    CO_N_unit_vec = CO_N_vector / CO_N_normal
                    CO_N_vector = CO_N_unit_vec * self.edge_length
                    CO_coordinate = N_coordinate - CO_N_vector
                    CO_N_normal = self.edge_length
                    prev_CO_coordinate = CO_coordinate

                    # If the residue is not an amino acid skip it.
                    if residue.polymer_type != residue.PT_AMINO:
                        prev_CO_coordinate = None
                        continue

                    # Case of Glycine, with no CB coordinate.
                    ca_atom = residue.find_atom('CA')
                    cb_atom = residue.find_atom('CB')
                    if ca_atom is not None and cb_atom is None:
                        CA_coordinate = ca_atom.coord
                        vector = N_coordinate - CO_coordinate
                        move_vertical_CO_CB = np.array([-1 / vector[0], 1 / vector[1], 0])

                    elif ca_atom is not None and cb_atom is not None:
                        CB_coordinate = cb_atom.coord
                        CA_coordinate = ca_atom.coord
                        CO_CB_vector = CB_coordinate - CO_coordinate
                        CO_CB_normal = np.linalg.norm(CO_CB_vector)

                        move_along_CO_CB = (0.5 * CO_N_normal - (np.dot(CO_N_vector, CO_CB_vector) / CO_N_normal)) * (
                                    CO_N_vector / CO_N_normal)

                        move_vertical_CO_CB = CB_coordinate + move_along_CO_CB - (CO_coordinate + N_coordinate) * 0.5

                    move_vertical_CO_CB *= ht2 * CO_N_normal / np.linalg.norm(move_vertical_CO_CB)
                    CB_coordinate = (CO_coordinate + N_coordinate) * 0.5 + move_vertical_CO_CB

                    centroid_CO_CB_N = (CO_coordinate + CB_coordinate + N_coordinate) / 3
                    H_direction = np.cross((N_coordinate - CO_coordinate), (CB_coordinate - CO_coordinate))

                    H_unit_direction = H_direction / np.linalg.norm(H_direction)
                    H_vector = ht3 * CO_N_normal * H_unit_direction
                    H_coordinate = centroid_CO_CB_N + H_vector

                    vertices = [N_coordinate, N_coordinate, N_coordinate, CO_coordinate, CO_coordinate, CO_coordinate,
                                CB_coordinate, CB_coordinate, CB_coordinate, H_coordinate, H_coordinate, H_coordinate]

                    chain_vertex.append(vertices)
                    amino_count += 1

                chain_vertex = np.array(chain_vertex, np.float32)
                self.vertices.append(chain_vertex)

    """
    A Function to create the Tetrahedron Models for each Chain components using the calculated vertices and faces.
    This takes the coordinates of currently opened session and modify accordingly.
    """
    def tetrahedron(self, chains=False):

        # If chains are not given then the whole model will be a Tetrahedron Model.
        if not chains:
            i = 0
            chains = []
            for model in self.model_list:
                for ch in model.chains:
                    chains.append((i, ch))
                    i += 1

        self.calculate_edge_length()
        self.calculate_vertices()

        # Remove the Protein Chain Model for the part needed to massed.
        tetrahedron_model = Model('Tetrahedrons', self.session)
        for (index, obj) in chains:
            ta = []
            amino = 0
            va = np.array(self.vertices[index], np.float32)

            for a in self.vertices[index]:
                e = amino * 12
                ta.extend([[e, e + 3, e + 6], [e + 1, e + 7, e + 9], [e + 2, e + 4, e + 10], [e + 5, e + 8, e + 11]])
                amino += 1

            if (0 in va.shape):
                continue

            # Create Sub-Models for each chain and add them to parent Tetrahedron Model.
            sub_model = Model("Chain " + obj.chain_id, self.session)
            va = np.reshape(va, (va.shape[0] * va.shape[1], va.shape[2]))
            ta = np.array(ta, np.int32)
            va_norm = calculate_vertex_normals(va, ta)

            sub_model.set_geometry(va, va_norm, ta)
            tetrahedron_model.add([sub_model])

        # Add the Tetrahedron Model to the running session.
        self.session.models.add([tetrahedron_model])

    """
    A Function to create the Massing Models for each Chain components using regular Tetrahedrons in a compact structure.
    This takes the coordinates of currently opened session and modify accordingly.
    
    Chain takes the map of models and list of it's chains to be massed,
    Unit defines the timesX the size of tetrahedrons edge length for massing tetrahedrons,
    Alpha defines the compactness of the massing volume.
    """
    def massing(self, chains=False, unit=1, alpha=2):

        # If chains not provided then create a Tetrahedron model of the whole session end the function.
        if not chains:
            self.tetrahedron()
            return

        # Creating the list of indices of chains along with the chain objects to be and not to be massed.
        i = 0
        tetra_chains, massing_chains, model = [], [], self.model_list[0]
        for ch in model.chains:
            if ch.chain_id not in chains:
                tetra_chains.append((i, ch))
            else:
                massing_chains.append((i, ch))
            i += 1

        # Models and corresponding chains not to be massed will be represented in the Tetrahedron Model.
        self.tetrahedron(chains=tetra_chains)
        massing_model = Model("Massing", self.session)

        for (index, ch) in massing_chains:

            # Collect all the coordinates to be considered inside massing volume.
            mesh_vertices = []
            for amino in ch.residues:
                if amino:
                    mesh_vertices.extend(amino.atoms.coords)

            # A mesh of covering the volume needed to be massed.
            import alphashape
            mesh = alphashape.alphashape(mesh_vertices, alpha * 0.1)

            # Function to define the position of a coordinate in respect to the created mesh.
            import trimesh
            inside = lambda ms: trimesh.proximity.ProximityQuery(ms).signed_distance
            tetrahedron_count = 0

            # Creating the First Tetrahedron for each Chain.
            edge_length = self.edge_length * unit
            pt = self.vertices[index][0]

            pt1 = pt[0]
            pt2 = pt[0] + (pt[3] - pt[0]) * edge_length / np.linalg.norm(pt[3] - pt[0])
            pt3 = pt[0] + (pt[6] - pt[0]) * edge_length / np.linalg.norm(pt[6] - pt[0])
            pt4 = pt[0] + (pt[9] - pt[0]) * edge_length / np.linalg.norm(pt[9] - pt[0])
            centroid = (pt1 + pt2 + pt3 + pt4) / 4

            idx = tetrahedron_count * 12
            massing_vertices = [[pt1, pt1, pt1, pt2, pt2, pt2, pt3, pt3, pt3, pt4, pt4, pt4]]
            faces = [[idx, idx + 3, idx + 6], [idx + 1, idx + 7, idx + 9], [idx + 2, idx + 4, idx + 10], [idx + 5, idx + 8, idx + 11]]
            tetrahedron_count += 1

            # A queue to take instance of all the tetrahedrons to be grown outwards to fill the massing volume.
            queue = [[tuple(pt1), tuple(pt2), tuple(pt3), tuple(pt4)]]

            # A set to keep instance of all the tetrahedrons already created.
            visited = {tuple((int(centroid[0]), int(centroid[1]), int(centroid[2])))}

            # Grow the Tetrahedrons using a fixed set of equations to get the required pattern.
            depth = 10**5
            while queue:

                # A Forceful loop termination if exceeded the iteration limit.
                if depth < 0:
                    raise Exception("DEPTH EXCEEDED: " + len(queue))
                depth -= 1

                # Create new four tetrahedrons from the already created ones.
                prev_tetrahedron = list(queue.pop())
                combine = permutations(prev_tetrahedron)

                for p in combine:
                    pt1, pt2, pt3, pt4 = p
                    pt1, pt2, pt3, pt4 = np.array(pt1), np.array(pt2), np.array(pt3), np.array(pt4)

                    # Set of equations to Generate the new tetrahedron from the previous one.
                    p1 = (pt1 + pt2) / 2
                    p3 = p1 + (pt4 - pt3)
                    p2 = pt4 + (pt2 - pt1) / 2
                    p4 = pt4 + (pt1 - pt2) / 2
                    centroid = (p1 + p2 + p3 + p4) / 4

                    # Check if Visited or Not.
                    t = tuple((round(centroid[0], 1), round(centroid[1], 1), round(centroid[2], 1)))
                    c = t not in visited

                    condition = c
                    # and c1_a and c2_a and c3_a and c4_a and c5_a and c6_a and c1_b and c2_b and c3_b and c4_b and c5_b and c6_b

                    # Check if it's out of boundary.
                    if condition and (inside(mesh)((centroid,)) > -3 * unit):
                        idx = tetrahedron_count * 12
                        massing_vertices.append([p1, p1, p1, p2, p2, p2, p3, p3, p3, p4, p4, p4])
                        faces.extend([[idx, idx + 3, idx + 6], [idx + 1, idx + 7, idx + 9], [idx + 2, idx + 4, idx + 10], [idx + 5, idx + 8, idx + 11]])
                        tetrahedron_count += 1

                        queue.append([tuple(p1), tuple(p2), tuple(p3), tuple(p4)])
                        visited.add(t)

            # Refine the massing Tetrahedrons within the tight boundary.
            x = 0
            visited = set()
            while x < len(massing_vertices):
                p1, p2, p3, p4 = massing_vertices[x][0], massing_vertices[x][3], massing_vertices[x][6], massing_vertices[x][9]
                centroid = (p1 + p2 + p3 + p4) / 4

                pt12, pt23, pt34, pt13, pt24, pt14 = (p1 + p2) / 2, (p2 + p3) / 2, (p3 + p4) / 2, (p1 + p3) / 2, (p2 + p4) / 2, (p1 + p4) / 2
                pt12_a, pt12_b = pt12 + edge_length * (p1 - p2) / (4 * np.linalg.norm((p1 - p2))), pt12 - edge_length * (p1 - p2) / (4 * np.linalg.norm((p1 - p2)));
                pt23_a, pt23_b = pt23 + edge_length * (p2 - p3) / (4 * np.linalg.norm((p2 - p3))), pt23 - edge_length * (p2 - p3) / (4 * np.linalg.norm((p2 - p3)));
                pt34_a, pt34_b = pt34 + edge_length * (p3 - p4) / (4 * np.linalg.norm((p3 - p4))), pt34 - edge_length * (p3 - p4) / (4 * np.linalg.norm((p3 - p4)));
                pt13_a, pt13_b = pt13 + edge_length * (p1 - p3) / (4 * np.linalg.norm((p1 - p3))), pt13 - edge_length * (p1 - p3) / (4 * np.linalg.norm((p1 - p3)));
                pt24_a, pt24_b = pt24 + edge_length * (p2 - p4) / (4 * np.linalg.norm((p2 - p4))), pt24 - edge_length * (p2 - p4) / (4 * np.linalg.norm((p2 - p4)));
                pt14_a, pt14_b = pt14 + edge_length * (p1 - p4) / (4 * np.linalg.norm((p1 - p4))), pt14 - edge_length * (p1 - p4) / (4 * np.linalg.norm((p1 - p4)));

                t1_a, t1_b = tuple((round(pt12_a[0], 2), round(pt12_a[1], 2), round(pt12_a[2], 2))), tuple((round(pt12_b[0], 2), round(pt12_b[1], 2), round(pt12_b[2], 2)))
                t2_a, t2_b = tuple((round(pt23_a[0], 2), round(pt23_a[1], 2), round(pt23_a[2], 2))), tuple((round(pt23_b[0], 2), round(pt23_b[1], 2), round(pt23_b[2], 2)))
                t3_a, t3_b = tuple((round(pt34_a[0], 2), round(pt34_a[1], 2), round(pt34_a[2], 2))), tuple((round(pt34_b[0], 2), round(pt34_b[1], 2), round(pt34_b[2], 2)))
                t4_a, t4_b = tuple((round(pt13_a[0], 2), round(pt13_a[1], 2), round(pt13_a[2], 2))), tuple((round(pt13_b[0], 2), round(pt13_b[1], 2), round(pt13_b[2], 2)))
                t5_a, t5_b = tuple((round(pt24_a[0], 2), round(pt24_a[1], 2), round(pt24_a[2], 2))), tuple((round(pt24_b[0], 2), round(pt24_b[1], 2), round(pt24_b[2], 2)))
                t6_a, t6_b = tuple((round(pt14_a[0], 2), round(pt14_a[1], 2), round(pt14_a[2], 2))), tuple((round(pt14_b[0], 2), round(pt14_b[1], 2), round(pt14_b[2], 2)))

                c1_a, c2_a, c3_a, c4_a, c5_a, c6_a = t1_a not in visited, t2_a not in visited, t3_a not in visited, t4_a not in visited, t5_a not in visited, t6_a not in visited
                c1_b, c2_b, c3_b, c4_b, c5_b, c6_b = t1_b not in visited, t2_b not in visited, t3_b not in visited, t4_b not in visited, t5_b not in visited, t6_b not in visited

                condition = c1_a and c2_a and c3_a and c4_a and c5_a and c6_a and c1_b and c2_b and c3_b and c4_b and c5_b and c6_b

                if condition and (inside(mesh)((centroid,)) > -0.8 * unit):
                    x += 1
                    visited.add(t1_a), visited.add(t2_a), visited.add(t3_a), visited.add(t4_a), visited.add(t5_a), visited.add(t6_a)
                    visited.add(t1_b), visited.add(t2_b), visited.add(t3_b), visited.add(t4_b), visited.add(t5_b), visited.add(t6_b)        

                else:
                    massing_vertices.pop(x)

            mass_v = np.array(massing_vertices)
            faces = np.array(faces[:len(mass_v) * 4], np.int32)
            massing_vertices = np.reshape(mass_v, (mass_v.shape[0] * mass_v.shape[1], mass_v.shape[2]))

            # Create Sub-Models for each chain and add them to parent Massing Model.
            chain_model = Model("Chain " + ch.chain_id, self.session)

            mass_v_norm = calculate_vertex_normals(massing_vertices, faces)
            chain_model.set_geometry(massing_vertices, mass_v_norm, faces)
            massing_model.add([chain_model])

        # Add the Massing Model to the running session.
        self.session.models.add([massing_model])


def tetrahedral_model(session, chains=False):
    #from Tetra import Tetra
    t = Tetra(session)
    if chains:
        chains = list(enumerate(chains))
    t.tetrahedron(chains=chains)

def massing_model(session, chains=None, unit=1, alpha=2):
    #from Tetra import Tetra
    t = Tetra(session)
    chain_ids = [c.chain_id for c in chains]
    t.massing(chains = chain_ids, unit = unit, alpha = alpha)

def register_command(session):
    from chimerax.core.commands import CmdDesc, register, ListOf, SetOf, TupleOf, RepeatOf, BoolArg, IntArg
    from chimerax.atomic import UniqueChainsArg
    t_desc = CmdDesc(required = [],
                     optional=[("chains", UniqueChainsArg)],
                     synopsis = 'creates tetrahedral model')
    register('tetra', t_desc, tetrahedral_model, logger=session.logger)

    m_desc = CmdDesc(required = [],
                     optional=[("chains", UniqueChainsArg)],
                     keyword=[("unit", IntArg), ("alpha", IntArg)],
                     synopsis = 'create tetrahedral massing model')
    register('massing', m_desc, massing_model, logger=session.logger)

register_command(session)
