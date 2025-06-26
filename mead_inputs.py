mead_lattice_create = 'fcc'
mead_lattice_spacing = 4.405

# The number of species to consider
nattype = 5
kim_model = 'Sim_LAMMPS_ADP_StarikovGordeevLysogorskiy_2020_SiAuAl__SM_113843830602_000'
units = 'metal'
atnames = ['Al', 'Al', 'Al', 'Al', 'Al']
isghost = [False, False, False, True, True]
phantom_to_real = [0, 0, 0, 2, 3]

atmasses = [26.981, 26.981 ,26.981, 26.981, 26.981]

radius_surf = 1
ratio = [0.95, 0.05]
ecutoff = 0.15
LoopMax = 1
Temp = 300.

lat = 0.7

natom_max = 40000
nloop_max = 10000
