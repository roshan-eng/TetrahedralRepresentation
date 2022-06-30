"""
Imported a Custom created class Polyhedron
Import Numpy for handling arrays
Import Bio that will be used to parse the pdb file to extract original coordinates
"""

from Polyhedron import Polyhedron
from itertools import combinations
import numpy as np
from Bio import PDB


class VertexCalc:

    def __init__(self, point1=None, point2=None, point3=None, center_=None, reduce=1):
        self.point1 = np.array(point1)
        self.point2 = np.array(point2)
        self.point3 = np.array(point3)
        self.center = np.array(center_)
        self.reduce = reduce
        self.points = None

    # Provide the fourth H vertex along with all provided ones
    def vertices(self):
        self.center /= self.reduce
        self.points = [self.point1 / self.reduce, self.point2 / self.reduce, self.point3 / self.reduce,
                       4 * self.center - (self.point1 + self.point2 + self.point3) / self.reduce]

        # Return all the  four points needed for a tetrahedron
        return self.points

    # Create a method to list all vertices in order that needed to create a face
    @staticmethod
    def face_indices(start_idx):
        return np.ravel(list(combinations(np.arange(start_idx, start_idx + 4), 3)))

    # Create a method to list all vertices in order that needed to create an outline
    @staticmethod
    def outline_indices(start_idx):
        return np.ravel(list(combinations(np.arange(start_idx, start_idx + 4), 2)))


if __name__ == "__main__":

    # Parse the PDB file
    parser = PDB.PDBParser()
    io = PDB.PDBIO()
    struct = parser.get_structure('5nx2', 'PDB_DATA/5nx2.pdb')       # Convert it into a struct format

    all_vertex = []                                         # It will contain the viewing coordinates of all the atoms
    all_face_indices = []                                   # It will store the info of vertices to join to form a face
    all_outline_indices = []                                # Stores the info of vertices to join to form an outline
    c_alphas = []                                           # Coordinates of all c_alpha to later calculate c_alpha_RMSD
    dist_from_c_alpha = []                                  # Coordinates of all atoms to calculate RMSD in amino atoms

    amino_count = 0
    amino_skipped_count = 0                                 # The amino acids whose model were skipped

    for model in struct:
        for chain in model:
            for amino_index in range(len(chain)):
                vertex_points = []
                max_bond_length = 1

                residue = list(list(chain)[amino_index])

                # If any residue contains less than or equals 4 atoms then it can't be modeled
                if len(residue) <= 4:
                    amino_skipped_count += 1
                    continue

                # Take the mid-points as two coordinates of tetrahedron
                # Skip for first and last amino acids
                # And the amide link mid-points distance from c_alpha will be taken as to extend other two vertices
                if amino_index <= 0:
                    vertex_points.append(residue[0].get_coord())
                else:
                    elem = list(list(chain)[amino_index - 1])[2].get_coord() + residue[0].get_coord()
                    vertex_points.append(elem / 2)

                    # Mid-pont distance from c_alpha
                    max_bond_length = np.linalg.norm(residue[1].get_coord() - elem / 2)

                if amino_index == len(chain) - 1:
                    vertex_points.append(residue[2].get_coord())
                else:
                    elem = residue[2].get_coord() + list(list(chain)[amino_index + 1])[0].get_coord()
                    vertex_points.append(elem / 2)

                    # Mid-pont distance from c_alpha
                    max_bond_length = np.linalg.norm(residue[1].get_coord() - elem / 2)

                # If max_bond length comes greater than 2 units then skip it cause of ambiguous results
                if max_bond_length > 2:
                    continue

                # c_alpha coordinate respect to frame
                # And now all other amino coordinates will be taken in reference to it
                center = residue[1].get_coord()
                c_alphas.append(center)

                # Extend the c_beta by a factor to make a regular tetrahedron
                c_beta_coord = residue[4].get_coord() - center
                factor = max_bond_length / np.linalg.norm(c_beta_coord)
                c_beta_coord *= factor
                c_beta_coord = c_beta_coord + center

                vertex_points.append(c_beta_coord)
                vertex_points.append(center)

                vertex = VertexCalc(*vertex_points, 30)
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

                # Store the distance between c_alpha and other four atoms
                for idx in range(4):
                    dist = np.linalg.norm(vertices[idx] * vertex.reduce - center)
                    dist_from_c_alpha.append(dist)
                    
    poly = Polyhedron()

    # Provide all the necessary inputs to opengl to draw the model
    poly.vertices = np.zeros(amino_count * 4, [("a_position", np.float32, 3), ("a_color", np.float32, 4)])
    poly.vertices["a_position"] = all_vertex
    poly.vertices["a_color"] = [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]] * amino_count
    poly.f_indices = np.array(all_face_indices, dtype=np.uint32)
    poly.o_indices = np.array(all_outline_indices, dtype=np.uint32)

    c_alpha_dist = []

    # Calculate the distances between c_alphas in structure
    for index in range(len(c_alphas) - 1):
        current_c_alpha = c_alphas[index]
        next_c_alpha = c_alphas[index + 1]
        norm = np.linalg.norm(next_c_alpha - current_c_alpha)
        c_alpha_dist.append(norm)

    # If the distance comes out ambiguously high then more likely a new chain start so skip the distance calc.
    c_alpha_dist = [x for x in c_alpha_dist if x < 2 * np.mean(c_alpha_dist)]
    dist_from_c_alpha = [x for x in dist_from_c_alpha if x < 2 * np.mean(dist_from_c_alpha)]

    # Calculate the RMSD values
    c_alpha_RMSD = (sum(map(lambda x: (x - np.mean(c_alpha_dist)) ** 2, c_alpha_dist)) / len(c_alpha_dist)) ** 0.5
    tetrahedron_atomic_dist_RMSD = (sum(map(lambda x: (x - np.mean(dist_from_c_alpha)) ** 2, dist_from_c_alpha)) /
                                    len(dist_from_c_alpha)) ** 0.5

    print("c_alpha_RMSD: ", c_alpha_RMSD)
    print("tetrahedron_atomic_dist_RMSD: ", tetrahedron_atomic_dist_RMSD)
    print("No. of Amino Acids Skipped: ", amino_skipped_count)

    # start the modelling
    poly.main()

    # Comment out to print all the vertices use to draw tetrahedron
    # Every four sets of coordinates from start will correspond to one tetrahedron in order as from .pdb file
    # print(all_vertex)
