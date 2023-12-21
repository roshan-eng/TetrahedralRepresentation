import alphashape
import numpy as np
from itertools import permutations, combinations

from chimerax.core.commands import CmdDesc
from chimerax.core.models import Model
from chimerax.surface import calculate_vertex_normals
from chimerax.core.commands import ListOf, SetOf, TupleOf, Or, RepeatOf, BoolArg, IntArg

ht3 = 0.81649658092772603273242802490196379732198249355222337614423086
ht2 = 0.86602540378443864676372317075293618347140262690519031402790349

# TODO: Validate the RMSD calculation methods
class Amino:
    def __init__(self, coords, obj):
        self.nh, self.co, self.c_beta, self.h, self.c_alpha = coords
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
            if (len(self.obj.atoms) <= 4):
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
            og_coords = [self.obj.atoms[x].coord for x in [0, 1]]
            if (len(self.obj.atoms) <= 4):
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

        self.all_edge_lengths, self.chain_elements = [], []
        self.session, self.edge_length = session, None
        self.model_list, self.protein = {}, {}

        for model in self.session.models.list():
            try:
                model.chains
            except AttributeError:
                print("TODO: Handle Pseudo-Models !")
            else:
                self.model_list[model.id] = model

    def regularize_egde_length(self, chain, res_index):

        mid_N_point = chain.residues[res_index].atoms[0].coord
        mid_CO_point = chain.residues[res_index].atoms[2].coord

        if res_index != 0 and chain.residues[res_index - 1] is not None:
            mid_N_point = (mid_N_point + chain.residues[res_index - 1].atoms[2].coord) * 0.5

        if res_index != len(chain.residues) - 1 and chain.residues[res_index + 1] is not None:
            mid_CO_point = (mid_CO_point + chain.residues[res_index + 1].atoms[0].coord) * 0.5

        self.all_edge_lengths.append(np.linalg.norm(mid_N_point - mid_CO_point))
        self.edge_length = sum(self.all_edge_lengths) / len(self.all_edge_lengths)

    def process_coordinates(self, chain, res_index, is_continuous_chain):

        N_coordinate = chain.residues[res_index].atoms[0].coord
        CO_coordinate = chain.residues[res_index].atoms[2].coord

        if res_index != 0 and chain.residues[res_index - 1]:
            if is_continuous_chain:
                N_coordinate = self.chain_elements[-1].co
            else:
                N_coordinate = (N_coordinate + chain.residues[res_index - 1].atoms[2].coord) * 0.5

        if res_index < len(chain.residues) - 1 and chain.residues[res_index + 1]:
            CO_coordinate = (CO_coordinate + chain.residues[res_index + 1].atoms[0].coord) * 0.5

        CO_coordinate = N_coordinate - (N_coordinate - CO_coordinate) * self.edge_length / np.linalg.norm(N_coordinate - CO_coordinate)
        CA_coordinate = chain.residues[res_index].atoms[1].coord
        CO_N_vector = N_coordinate - CO_coordinate

        if len(chain.residues[res_index].atoms) <= 4:
            vector = N_coordinate - CO_coordinate
            move_vertical_CO_CB = np.array([-1 / vector[0], 1 / vector[1], 0])
        else:
            CB_coordinate = chain.residues[res_index].atoms[4].coord
            move_along_CO_CB = (0.5 * self.edge_length - (np.dot(CO_N_vector, (CB_coordinate - CO_coordinate)) / self.edge_length)) * (CO_N_vector / self.edge_length)
            move_vertical_CO_CB = CB_coordinate + move_along_CO_CB - (CO_coordinate + N_coordinate) * 0.5

        move_vertical_CO_CB *= ht2 * self.edge_length / np.linalg.norm(move_vertical_CO_CB)
        CB_coordinate = (CO_coordinate + N_coordinate) * 0.5 + move_vertical_CO_CB

        # TODO: Check for directionality in cross product
        H_direction = np.cross((N_coordinate - CO_coordinate), (CB_coordinate - CO_coordinate))
        H_vector = ht3 * self.edge_length * H_direction / np.linalg.norm(H_direction)
        H_coordinate = (CO_coordinate + CB_coordinate + N_coordinate) / 3 + H_vector

        vertices = [N_coordinate, CO_coordinate, CB_coordinate, H_coordinate, (N_coordinate + CO_coordinate + CB_coordinate + H_coordinate) / 4]
        self.chain_elements.append(Amino(vertices, chain.residues[res_index]))

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

    def grow(self, ms, unit):

        mesh = alphashape.alphashape(ms, alpha * 0.1)
        inside = lambda ms: trimesh.proximity.ProximityQuery(ms).signed_distance
        el = self.edge_length * unit

        pt1 = ms[0]
        pt2 = ms[0] + (ms[1] - ms[0]) * el / np.linalg.norm(ms[1] - ms[0])
        pt3 = ms[0] + (ms[2] - ms[0]) * el / np.linalg.norm(ms[2] - ms[0])
        pt4 = ms[0] + (ms[3] - ms[0]) * el / np.linalg.norm(ms[3] - ms[0])
        centroid = (pt1 + pt2 + pt3 + pt4) / 4

        massing_coords = [[pt1, pt1, pt1, pt2, pt2, pt2, pt3, pt3, pt3, pt4, pt4, pt4]]
        visited = {tuple((int(centroid[0]), int(centroid[1]), int(centroid[2])))}
        queue = [[tuple(pt1), tuple(pt2), tuple(pt3), tuple(pt4)]]
        faces = [[0, 3, 6], [1, 7, 9], [2, 4, 10], [5, 8, 11]]

        tcnt, depth = 1, 10**5
        while queue:

            if depth < 0:
                raise Exception("DEPTH EXCEEDED: " + len(queue))
            depth -= 1

            prev_tetra = list(queue.pop())
            combine = permutations(prev_tetra)

            for p in combine:
                pt1, pt2, pt3, pt4 = [np.array(x) for x in p]
                p1, p2, p3, p4 = (pt1 + pt2) / 2, pt4 + (pt2 - pt1) / 2, p1 + (pt4 - pt3), pt4 + (pt1 - pt2) / 2
                centroid = (p1 + p2 + p3 + p4) / 4

                t = tuple((round(centroid[0], 1), round(centroid[1], 1), round(centroid[2], 1)))
                is_visited = t in visited

                if not is_visited and (inside(mesh)((centroid,)) > -3 * unit):
                    massing_coords.append([p1, p1, p1, p2, p2, p2, p3, p3, p3, p4, p4, p4])
                    faces.extend([[12*tcnt, 12*tcnt + 3, 12*tcnt + 6], [12*tcnt + 1, 12*tcnt + 7, 12*tcnt + 9], [12*tcnt + 2, 12*tcnt + 4, 12*tcnt + 10], [12*tcnt + 5, 12*tcnt + 8, 12*tcnt + 11]])
                    tcnt += 1

                    queue.append([tuple(p1), tuple(p2), tuple(p3), tuple(p4)])
                    visited.add(t)

        x = 0
        visited = set()
        while x < len(massing_coords):
            p1, p2, p3, p4 = massing_coords[x][0], massing_coords[x][3], massing_coords[x][6], massing_coords[x][9]
            centroid = (p1 + p2 + p3 + p4) / 4

            def checking_coords(pt, a, b):
                return pt + el * (a - b) / (4 * np.linalg.norm((a - b))), pt - el * (a - b) / (4 * np.linalg.norm((a - b)));

            pt12, pt23, pt34, pt13, pt24, pt14 = (p1 + p2) / 2, (p2 + p3) / 2, (p3 + p4) / 2, (p1 + p3) / 2, (p2 + p4) / 2, (p1 + p4) / 2
            pt12_a, pt12_b = checking_coords(pt12, p1, p2)
            pt23_a, pt23_b = checking_coords(pt23, p2, p3)
            pt34_a, pt34_b = checking_coords(pt34, p3, p4)
            pt13_a, pt13_b = checking_coords(pt13, p1, p3)
            pt24_a, pt24_b = checking_coords(pt24, p2, p4)
            pt14_a, pt14_b = checking_coords(pt14, p1, p4)

            def convert_to_tpl(a, b):
                return tuple((round(pt12_a[0], 2), round(pt12_a[1], 2), round(pt12_a[2], 2))), tuple((round(pt12_b[0], 2), round(pt12_b[1], 2), round(pt12_b[2], 2)))

            t1_a, t1_b = convert_to_tpl(pt12_a, pt12_b)
            t2_a, t2_b = convert_to_tpl(pt23_a, pt23_b)
            t3_a, t3_b = convert_to_tpl(pt34_a, pt34_b)
            t4_a, t4_b = convert_to_tpl(pt13_a, pt13_b)
            t5_a, t5_b = convert_to_tpl(pt24_a, pt24_b)
            t6_a, t6_b = convert_to_tpl(pt14_a, p14_b)

            condition = t1_a in visited or t2_a in visited or t3_a in visited or t4_a in visited or t5_a in visited or t6_a in visited or 
                        t1_b in visited or t2_b in visited or t3_b in visited or t4_b in visited or t5_b in visited or t6_b in visited

            if not condition and (inside(mesh)((centroid,)) > -0.8 * unit):
                x += 1
                visited.update([t1_a, t1_b, t2_a, t2_b, t3_a, t3_b, t4_a, t4_b, t5_a, t5_b, t6_a, t6_b])
            else:
                massing_coords.pop(x)

        return np.array(massing_coords), np.array(faces[:len(massing_coords) * 4], np.int32)

    def tetrahedron(self, sequence=False, chains=False):
        def add_to_sub_model(va, ta, cid):
            sub_model = Model("Chain " + cid, self.session)
            va = np.reshape(va, (va.shape[0] * va.shape[1], va.shape[2]))
            ta = np.array(ta, np.int32)

            sub_model.set_geometry(va, calculate_vertex_normals(va, ta), ta)
            tetrahedron_model.add([sub_model])

        tetrahedron_model = Model('Tetrahedrons', self.session)
        self.iterate_aminos()
        self.iterate_aminos(execute=True)

        if sequence:
            for chain in self.protein.values():
                va, ta, x = [], [], 0
                for am in chain:
                    in_seq = False
                    for seq in sequence:
                        if len(seq) < 1 or len(seq) > 2:
                            print("INVALID SEQUENCE !")
                            return
                        else if (len(seq) == 1 and seq[0] <= am.obj.number) or (seq[0] <= am.obj.number and am.obj.number < seq[1]):
                            in_seq = True

                        if in_seq:
                            ta.extend([[12*x, 12*x + 3, 12*x + 6], [12*x + 1, 12*x + 7, 12*x + 9], [12*x + 2, 12*x + 4, 12*x + 10], [12*x + 5, 12*x + 8, 12*x + 11]])
                            va.append(am.model_coords)
                            x += 1
                            break

                va = np.array(va, np.float32)
                if 0 not in va.shape:
                    add_to_sub_model(va, ta, chain[0].obj.chain_id)

        else:
            if not chains:
                chains = []
                for model in self.model_list.values():
                    for ch in model.chains:
                        chains.append(ch.chain_id)

            for ids in chains:
                va, ta = np.array([am.model_coords for am in self.protein[ids]], np.float32), []
                for x in range(len(self.protein[ids])):
                    ta.extend([[12*x, 12*x + 3, 12*x + 6], [12*x + 1, 12*x + 7, 12*x + 9], [12*x + 2, 12*x + 4, 12*x + 10], [12*x + 5, 12*x + 8, 12*x + 11]])

                if 0 not in va.shape:
                    add_to_sub_model(va, ta, ids)

        self.session.models.add([tetrahedron_model])

    def massing(self, sequence=False, chains=False, unit=1, alpha=2):

        def add_to_sub_model(ms, mass_id):
            massing_coords, faces = self.grow(ms, unit)
            if not np.all(massing_coords.shape):
                continue

            massing_coords = np.reshape(massing_coords, (massing_coords.shape[0] * massing_coords.shape[1], massing_coords.shape[2]))

            sub_model = Model("Chain " + mass_id, self.session)
            sub_model.set_geometry(massing_vertices, calculate_vertex_normals(massing_vertices, faces), faces)
            self.massing_model.add([sub_model])
            mass_id = chr(ord(mass_id) + 1)

        mass_id = "A"
        self.massing_model = Model("Massing", self.session)

        if sequence:
            tetra_seq = []
            for i in range(len(sequence)):
                if i == 0 and sequence[0][0] != 0:
                    tetra_seq.append((0, sequence[0][0]))
                else if i == len(sequence) - 1 and sequence[-1][1] != self.protein[-1][-1].number + 1:
                    tetra_seq.append((sequence[-1][1], self.protein[-1][-1].number + 1))

            self.tetrahedron(sequence=tetra_seq)

            for seq in sequence:
                for chain in self.protein.values():
                    ms = []
                    for am in chain:
                        if seq[0] <= am.obj.number and am.obj.number < seq[1]:
                            ms.extend(am.coords)

                    add_to_sub_model(ms, mass_id)

        else:
            chain_ids = []
            for ids in self.protein.keys():
                if ids not in chains:
                    chain_ids.append(ids)

            self.tetrahedron(chains=chain_ids)

            for chain in self.protein.values():
                ms = [v for v in x.coords for x in chain]
                add_to_sub_model(ms, mass_id)

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
