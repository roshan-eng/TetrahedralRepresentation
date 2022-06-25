# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
from Polyhedron import Polyhedron, VertexCalc
import numpy as np
from Bio import PDB

if __name__ == "__main__":

    parser = PDB.PDBParser()
    io = PDB.PDBIO()
    struct = parser.get_structure('1dn3', '/home/roshan/PycharmProjects/Chimera_Projects/1dn3.pdb')

    all_vertex = []
    all_face_indices = []
    all_outline_indices = []

    amino_count = 0
    for model in struct:
        for chain in model:
            for amino_index in range(len(chain)):
                vertex_points = []
                max_bond_length = 1
                residue = list(list(chain)[amino_index])

                if amino_index <= 0:
                    vertex_points.append(residue[0].get_coord())
                else:
                    elem = list(list(chain)[amino_index - 1])[2].get_coord() + residue[0].get_coord()
                    vertex_points.append(elem / 2)
                    max_bond_length = np.linalg.norm(residue[1].get_coord() - elem / 2)

                if amino_index == len(chain) - 1:
                    vertex_points.append(residue[2].get_coord())
                else:
                    elem = residue[2].get_coord() + list(list(chain)[amino_index + 1])[0].get_coord()
                    vertex_points.append(elem / 2)
                    max_bond_length = np.linalg.norm(residue[1].get_coord() - elem / 2)

                center = residue[1].get_coord()
                c_beta_coord = residue[4].get_coord() - center
                factor = max_bond_length / np.linalg.norm(c_beta_coord)
                c_beta_coord *= factor
                c_beta_coord = c_beta_coord + center

                vertex_points.append(c_beta_coord)
                vertex_points.append(center)

                vertex = VertexCalc(*vertex_points, 10)
                vertices = vertex.vertices()
                face_index = vertex.face_indices(amino_count * 4)
                outline_index = vertex.outline_indices(amino_count * 4)

                amino_count += 1
                for vertex_elem in vertices:
                    all_vertex.append(list(vertex_elem))

                for face_elem in face_index:
                    all_face_indices.append(face_elem)

                for outline_elem in outline_index:
                    all_outline_indices.append(outline_elem)

    poly = Polyhedron()
    poly.vertices = np.zeros(amino_count * 4, [("a_position", np.float32, 3), ("a_color", np.float32, 4)])
    poly.vertices["a_position"] = all_vertex
    poly.vertices["a_color"] = [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]] * amino_count
    poly.f_indices = np.array(all_face_indices, dtype=np.uint32)
    poly.o_indices = np.array(all_outline_indices, dtype=np.uint32)

    poly.main()
