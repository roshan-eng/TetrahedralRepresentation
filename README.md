Instructions to run.

-> Download the Tetra.py file in ChimeraX sitepackage file. This way we can import the Code in ChimeraX Shell. (D:\PC Softwares\ChimeraX\bin\Lib\site-packages\Tetra.py)
-> GOTO (in ChimeraX): Tools > General > Shell
-> Shell CMD:
[1] from Tetra import Tetra
[2] tetra = Tetra(session)
[3] tetra.massing(sequence={"A":[(0, 100)], "B":[(20, 40), (60, 100)], "R":[(240, 300), (350, 420)]}, unit=1.5)

Here, the massing have following attributes.
Sequence: Give a sequence in dict format where key is the chain id and value is a list of index based range that needed to be massed. If sequence is given then code will only consider the sequence (not the chains even if given).
Chains: Give a tuple of all the chains that needed to be massed.
Unit: The magnifying factor of tetrahedron sizes.
alpha: Defines that how tight the massing needed to be along the model original coordinates. By-deafult its 2 that gives fairly consistent results.

Everything other than given in massing will be created in tetrahedron model. We can run tetrahedron seperately with similar parameters needed.

