# wiac-tetrahedral-modeling
Generating tetrahedral models of biological molecules,
especially proteins and protein complexes

Note: Used the OpenGL to visualize the structures

Firstly to create a regular / prefect Tetrahedron all joined
together from the vertices the approach taken was like;

1. Take first two coordinates as the mid-points of amide
linkages that will create a vertex-to-vertex joined tetrahedrons.

2. To create a perfect tetrahedron, the C-Beta will be
shifted by some factor farther from C-alpha. This will be
our third coordinate.

3. And the last H coordinate can be approximated from other 3
coordinates to create a perfect tetrahedron.

The RMSD value "c_alpha_RMSD"  were calculated between the C-alphas
The RMSD value "tetrahedron_atomic_dist_RMSD" were also calculated
between the C-alpha to all other four vertices of tetrahedron.
