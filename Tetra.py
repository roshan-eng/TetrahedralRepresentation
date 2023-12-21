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

class Amino:
    def __init__(self, coords, obj):
        self.nh, self.c_alpha, self.co, self.c_beta, self.h = coords
        self._model_coords = [self.nh, self.nh, self.nh, self.co, self.co, self.co, self.c_beta, self.c_beta, self.c_beta, self.h, self.h, self.h]
        self._rmsd_calpha, self._rmsd = None, None
        self._e_len_og, _e_len = None, None
        self.obj, self.coords = obj, coords

    @property
    def model_coords(self):
        return self._model_coords
    
    @model_coords.setter
    def model_coords(self, coords):
        self._model_coords = coords

    @property
    def rmsd_calpha(self):
        if not self._rmsd_calpha: 
            self._rmsd_calpha = np.sqrt(((np.linalg.norm(self.obj.atoms[1].coord - self.coords[1])) ** 2).mean())

        return self._rmsd_calpha

    @rmsd_calpha.setter
    def rmsd_calpha(self, val):
        self._rmsd_calpha = val

    @property
    def rmsd(self):
        if not self._rmsd:
            og_coords = [self.obj.atoms[x].coord for x in [0, 1, 2]]
            if (self.obj.name == 'GLY'):
                og_coords.append(self.c_beta)
            else:
                og_coords.append(self.obj.atoms[4].coord)

            og_coords.append(self.h)
            self._rmsd = np.sqrt((np.array([np.linalg.norm(p1 - p2) ** 2 for (p1, p2) in zip(og_coords, self.coords)])).mean())

        return self._rmsd

    @rmsd.setter
    def rmsd(self, val):
        self._rmsd = val

    @property
    def e_len_og(self):
        if not self._e_len_og:
            og_coords = [self.obj.atoms[x].coord for x in [0, 2]]
            if (self.obj.name == 'GLY'):
                og_coords.append(self.c_beta)
            else:
                og_coords.append(self.obj.atoms[4].coord)

            og_coords.append(self.h)
            x = itertools.combinations(og_coords, 2)
            self._e_len_og = np.array([np.linalg.norm(p1 - p2) for (p1, p2) in x]).mean()

        return self._e_len_og

    @e_len_og.setter
    def e_len_og(self, val):
        self._e_len_og = val

    @property
    def e_len(self):
        if not self._e_len:
            x = itertools.combinations(self.coords[:1] + self.coords[2:], 2)
            self._e_len = np.array([np.linalg.norm(p1 - p2) for (p1, p2) in x]).mean()

        return self._e_len

    @e_len.setter
    def e_len(self, val):
        self._e_len = val

class Tetra:

    def __init__(self, session):

        """
        Initialize the Session, List of all the models in use, Edge-length of Tetrahedrons
        and the arrays to store all the required vertices and faces.
        """

        self.session = session
        self.vertices = []
        self.faces = []

        self.all_edge_lengths = []
        self.chain_elements = []
        self.model_list = {}
        self.protein = {}

        self.edge_length = None

        # Remove the PseudoModels from the Model list as they have no Chain components.
        for model in self.session.models.list():
            try:
                model.chains
            except AttributeError:
                print("PesudoModels Found !!!")
            else:
                self.model_list[model.id] = model

    def regularize_egde_length(self, chain, res_index):

        mid_N_point = chain.residues[res_index].atoms[0].coord
        mid_CO_point = chain.residues[res_index].atoms[2].coord

        # Consider the mid-points of CO-N bonds as a vertex to form a continuous joined chains of tetrahedrons.
        # Condition check for edge cases of first and last amino acids in a chain.
        if res_index != 0 and chain.residues[res_index - 1] is not None:
            mid_N_point = (mid_N_point + chain.residues[res_index - 1].atoms[2].coord) * 0.5

        if res_index != len(chain.residues) - 1 and chain.residues[res_index + 1] is not None:
            mid_CO_point = (mid_CO_point + chain.residues[res_index + 1].atoms[0].coord) * 0.5

        e = np.linalg.norm(mid_N_point - mid_CO_point)
        self.all_edge_lengths.append(e)
        self.edge_length = sum(self.all_edge_lengths) / len(self.all_edge_lengths)

    def process_coordinates(self, chain, res_index, is_continuous_chain):

        N_coordinate = chain.residues[res_index].atoms[0].coord
        CO_coordinate = chain.residues[res_index].atoms[2].coord

        # Consider the mid-points of CO-N bonds as a vertex to form a continuous joined chains of tetrahedrons.
        # Condition check for edge cases of first and last amino acids in a chain.
        if res_index != 0 and chain.residues[res_index - 1] is not None:
            if is_continuous_chain:
                N_coordinate = self.chain_elements[-1].co
            else:
                N_coordinate = (N_coordinate + chain.residues[res_index - 1].atoms[2].coord) * 0.5

        if res_index < len(chain.residues) - 1 and chain.residues[res_index + 1]:
            CO_coordinate = (CO_coordinate + chain.residues[res_index + 1].atoms[0].coord) * 0.5

        # Coordinates for a regular Tetrahedron.
        CO_coordinate = N_coordinate - (N_coordinate - CO_coordinate) * self.edge_length / np.linalg.norm(N_coordinate - CO_coordinate)
        CO_N_vector = N_coordinate - CO_coordinate

        # Case of Glycine, with no CB coordinate.
        if len(chain.residues[res_index].atoms) <= 4:
            CA_coordinate = chain.residues[res_index].atoms[1].coord
            vector = N_coordinate - CO_coordinate
            move_vertical_CO_CB = np.array([-1 / vector[0], 1 / vector[1], 0])
        else:
            CA_coordinate = chain.residues[res_index].atoms[1].coord
            CB_coordinate = chain.residues[res_index].atoms[4].coord
            move_along_CO_CB = (0.5 * self.edge_length - (np.dot(CO_N_vector, (CB_coordinate - CO_coordinate)) / self.edge_length)) * (CO_N_vector / self.edge_length)
            move_vertical_CO_CB = CB_coordinate + move_along_CO_CB - (CO_coordinate + N_coordinate) * 0.5

        move_vertical_CO_CB *= ht2 * self.edge_length / np.linalg.norm(move_vertical_CO_CB)
        CB_coordinate = (CO_coordinate + N_coordinate) * 0.5 + move_vertical_CO_CB

        H_direction = np.cross((N_coordinate - CO_coordinate), (CB_coordinate - CO_coordinate))
        H_vector = ht3 * self.edge_length * H_direction / np.linalg.norm(H_direction)
        H_coordinate = (CO_coordinate + CB_coordinate + N_coordinate) / 3 + H_vector

        vertices = [N_coordinate, (N_coordinate + CO_coordinate + CB_coordinate + H_coordinate) / 4, CO_coordinate, CB_coordinate, H_coordinate]

        self.chain_elements.append(Amino(vertices, chain.residues[res_index]))

    """
    Calculation of the required Edge-length of tetrahedrons to use as the average length of CO-N bonds
    over the whole model to minimize the deviation from original vertices.
    """
    def iterate_aminos(self, execute=False):

        for model in self.model_list.values():
            for chain in model.chains:
                is_continuous_chain = False
                for res_index in range(len(chain.residues)):

                    residue = chain.residues[res_index]
                    if not residue or residue.polymer_type != residue.PT_AMINO or not residue.find_atom('CA'):
                        is_continuous_chain = False
                        continue

                    if execute:
                        self.process_coordinates(chain, res_index, is_continuous_chain)
                        is_continuous_chain = True
                    else:
                        self.regularize_egde_length(chain, res_index)

                if execute:
                    self.protein[chain.chain_id] = self.chain_elements
                    self.chain_elements = []
    """
    Calculation of vertices that represents the complete Tetrahedron Structure.
    The original coordinates were deviated from to form a regular Tetrahedron where the four edges of Tetrahedrons
    represents the N, CO, CB and H respectively.
    """
    def grow(self, massing_vertices, faces, queue, visited, mesh, unit, tetrahedron_count, edge_length):
        # Function to define the position of a coordinate in respect to the created mesh.
        import trimesh
        inside = lambda ms: trimesh.proximity.ProximityQuery(ms).signed_distance

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

        return massing_vertices, faces

    """
    A Function to create the Tetrahedron Models for each Chain components using the calculated vertices and faces.
    This takes the coordinates of currently opened session and modify accordingly.
    """
    def tetrahedron(self, sequence=False, chains=False):
        tetrahedron_model = Model('Tetrahedrons', self.session)
        self.iterate_aminos()
        self.iterate_aminos(execute=True)

        if sequence:
            for chain in self.protein.values():
                va, ta, x = [], [], 0
                for am in chain:
                    if sequence[0] <= am.obj.number and am.obj.number < sequence[1]:
                        ta.extend([[12*x, 12*x + 3, 12*x + 6], [12*x + 1, 12*x + 7, 12*x + 9], [12*x + 2, 12*x + 4, 12*x + 10], [12*x + 5, 12*x + 8, 12*x + 11]])
                        va.append(am.model_coords)
                        x += 1

                va = np.array(va, np.float32)
                if 0 not in va.shape:
                    # Create Sub-Models for each chain and add them to parent Tetrahedron Model.
                    sub_model = Model("Chain " + chain[0].obj.chain_id + " (SEQ)", self.session)
                    va = np.reshape(va, (va.shape[0] * va.shape[1], va.shape[2]))
                    ta = np.array(ta, np.int32)
                    va_norm = calculate_vertex_normals(va, ta)

                    sub_model.set_geometry(va, va_norm, ta)
                    tetrahedron_model.add([sub_model])

        else:
            # If chains are not given then the whole model will be a Tetrahedron Model.
            if not chains:
                chains = []
                for model in self.model_list.values():
                    for ch in model.chains:
                        chains.append(ch.chain_id)

            # Remove the Protein Chain Model for the part needed to massed.
            for ids in chains:
                va, ta = np.array([am.model_coords for am in self.protein[ids]], np.float32), []

                for x in range(len(self.protein[ids])):
                    ta.extend([[12*x, 12*x + 3, 12*x + 6], [12*x + 1, 12*x + 7, 12*x + 9], [12*x + 2, 12*x + 4, 12*x + 10], [12*x + 5, 12*x + 8, 12*x + 11]])

                if 0 not in va.shape:
                    # Create Sub-Models for each chain and add them to parent Tetrahedron Model.
                    sub_model = Model("Chain " + ids, self.session)
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
    def massing(self, sequence=False, chains=False, unit=1, alpha=2):

        massing_model = Model("Massing", self.session)
        self.tetrahedron(sequence)

        if sequence:
            cnt = 0
            mesh_vert_lt, mesh_vert, pts = [], [], []
            for lt in self.vertices:
                for i in range(len(lt)):
                    cnt += 1

                    if sequence[0] <= cnt and cnt <= sequence[1]:
                        mesh_vert.extend([lt[i][0], lt[i][3], lt[i][6], lt[i][9]])
                        
                        if i == 0 or cnt == sequence[0]:
                            pts.append(lt[i])

                        if i == len(lt) - 1 or cnt == sequence[1]:
                            mesh_vert_lt.append(mesh_vert)
                            mesh_vert = []

                    elif cnt > sequence[1]:
                        break

            # Creating the First Tetrahedron for each Chain.
            edge_length = self.edge_length * unit

            mass_id = 1
            for pt, mv in zip(pts, mesh_vert_lt):
                if not mv:
                    print("Wrong Input Format !!!")
                    continue

                # A mesh of covering the volume needed to be massed.
                import alphashape
                mesh = alphashape.alphashape(mv, alpha * 0.1)
                tetrahedron_count = 0

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

                massing_vertices, faces = self.grow(massing_vertices, faces, queue, visited, mesh, unit, tetrahedron_count, edge_length)

                mass_v = np.array(massing_vertices)
                if not np.all(mass_v.shape) : continue

                faces = np.array(faces[:len(mass_v) * 4], np.int32)
                massing_vertices = np.reshape(mass_v, (mass_v.shape[0] * mass_v.shape[1], mass_v.shape[2]))

                # Create Sub-Models for each chain and add them to parent Massing Model.
                chain_model = Model("Chain" + str(mass_id), self.session)
                mass_id += 1

                mass_v_norm = calculate_vertex_normals(massing_vertices, faces)
                chain_model.set_geometry(massing_vertices, mass_v_norm, faces)
                massing_model.add([chain_model])

        else:
            # If chains not provided then create a Tetrahedron model of the whole session end the function.
            if not chains:
                self.tetrahedron()
                return

            # Creating the list of indices of chains along with the chain objects to be and not to be massed.
            i = 0
            tetra_chains, massing_chains, model = [], [], self.model_list.values()[0]
            for ch in model.chains:
                if ch.chain_id not in chains:
                    tetra_chains.append((i, ch))
                else:
                    massing_chains.append((i, ch))
                i += 1

            # Models and corresponding chains not to be massed will be represented in the Tetrahedron Model.
            self.tetrahedron(chains=tetra_chains)

            for (index, ch) in massing_chains:

                # Collect all the coordinates to be considered inside massing volume.
                mesh_vertices = []
                for amino in ch.residues:
                    if amino:
                        mesh_vertices.extend(amino.atoms.coords)

                # A mesh of covering the volume needed to be massed.
                import alphashape
                mesh = alphashape.alphashape(mesh_vertices, alpha * 0.1)

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

                massing_vertices, faces = self.grow(massing_vertices, faces, queue, visited, mesh, unit, tetrahedron_count, edge_length)

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


# def tetrahedral_model(session, chains=False):
#     #from Tetra import Tetra
#     t = Tetra(session)
#     if chains:
#         chains = list(enumerate(chains))
#     t.tetrahedron(chains=chains)

# def massing_model(session, chains=None, unit=1, alpha=2):
#     #from Tetra import Tetra
#     t = Tetra(session)
#     chain_ids = [c.chain_id for c in chains]
#     t.massing(chains = chain_ids, unit = unit, alpha = alpha)

# def register_command(session):
#     from chimerax.core.commands import CmdDesc, register, ListOf, SetOf, TupleOf, RepeatOf, BoolArg, IntArg
#     from chimerax.atomic import UniqueChainsArg
#     t_desc = CmdDesc(required = [],
#                      optional=[("chains", UniqueChainsArg)],
#                      synopsis = 'creates tetrahedral model')
#     register('tetra', t_desc, tetrahedral_model, logger=session.logger)

#     m_desc = CmdDesc(required = [],
#                      optional=[("chains", UniqueChainsArg)],
#                      keyword=[("unit", IntArg), ("alpha", IntArg)],
#                      synopsis = 'create tetrahedral massing model')
#     register('massing', m_desc, massing_model, logger=session.logger)

# register_command(session)
