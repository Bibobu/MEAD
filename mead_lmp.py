#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
A full Python version of the MEAD procedure.
'''

from mpi4py import MPI

import argparse
import logging
import numpy as np
from ctypes import c_int
from lammps import lammps, LMP_STYLE_ATOM, LMP_TYPE_VECTOR

from mead_inputs import *


def distance(pos1, pos2, lx, ly):
    dx = np.sqrt((pos1[0]-pos2[0])**2)
    dx -= dx*np.rint(dx/lx)*lx
    dy = np.sqrt((pos1[1]-pos2[1])**2)
    dy -= dy*np.rint(dy/ly)*ly
    dz = np.sqrt((pos1[2]-pos2[2])**2)
    dist = np.sqrt(dx*dx+dy*dy+dz*dz)
    return dist


def compute_surface(lmp, radius_surf, isolevel, resolution):
    import vtk
    import ovito
    from ovito.io import export_file
    from ovito.io.lammps import lammps_to_ovito
    from ovito.modifiers import ConstructSurfaceModifier, SliceModifier
    from ovito.data import DataCollection, SimulationCell, ParticleType, Particles

    data = lammps_to_ovito(lmp)
    if data is None:
        return

    slice_dist = 10.
    pbc = data.cell.pbc

    SURFmod = ConstructSurfaceModifier(
        method=ConstructSurfaceModifier.Method.GaussianDensity,
        radius_scaling=radius_surf,
        isolevel=isolevel,
        grid_resolution=resolution,
    )
    SLICmod = SliceModifier(operate_on={"surfaces"},
    normal=(0., 0., -1.),
    distance= -slice_dist,
    )

    data.apply(SURFmod)
    data.apply(SLICmod)
    surface = data.surfaces["surface_"]
    surface.domain_.pbc = (pbc[0], pbc[1], 0)

    outfile = "".join(["surface", ".vtk"])
    export_file(surface, outfile, "vtk/trimesh", include_caps=False)
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(outfile)
    reader.Update()
    filt = vtk.vtkGeometryFilter()
    filt.SetInputConnection(reader.GetOutputPort())
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(filt.GetOutputPort())
    writer.SetFileName("".join(["surface", ".stl"]))
    writer.Write()

    positions = surface.vertices_['Position']

    # Ugly but similar to original version
    positions = positions[positions[:, 2] > slice_dist]
    xlo, xhi = min(positions[:, 0]), max(positions[:, 0])
    ylo, yhi = min(positions[:, 1]), max(positions[:, 1])
    zlo, zhi = min(positions[:, 2]), max(positions[:, 2])

    if np.abs(xhi-xlo) < lat:
        xhi += lat/2.
    if np.abs(yhi-ylo) < lat:
        yhi += lat/2.
    if np.abs(zhi-zlo) < lat:
        zhi = zlo + lat

    return [xlo, xhi, ylo, yhi, zlo, zhi]


def compute_minima(lmp, rng, phantom_coords, phantom_energy, phantom_min, phantom_max, Temp):

    # LAMMPS global parameters
    Kb = lmp.extract_global('boltz')
    xlo = lmp.extract_global('boxxlo')
    xhi = lmp.extract_global('boxxhi')
    ylo = lmp.extract_global('boxylo')
    yhi = lmp.extract_global('boxyhi')
    lx = xhi-xlo
    ly = yhi-ylo

    # MC or regular MEAD
    KbT = Kb*Temp
    thresh = 0.15
    rc = 3.15
    keep = []
    nphantoms = phantom_coords.shape[0]
    nadatoms = 100
    nstep = 10
    if nphantoms < nadatoms:
        logging.warning("Number of deposited atoms {:d} higher than phantom positions {:d}. Switching to old MEAD.".format(nphantoms,nadatoms))
        KbT = 0

    # print("KBT = {:f}, Kb = {:f}, T = {:f}".format(KbT, Kb, Temp))
    if KbT:
        print("Using MC moves")
        actual_atoms = np.zeros(nphantoms, dtype=int)
        rand_init_ids = rng.choice(nphantoms, size=nadatoms, replace=False)
        for n, i in enumerate(rand_init_ids):
            actual_atoms[i] = n+1

        for myatom in range(n):
            print("Added {:d} atoms".format(myatom))
            myatom_id = np.argwhere(actual_atoms == myatom+1)[0]
            myatom_pos = phantom_coords[myatom_id]
            myatom_pos = np.reshape(myatom_pos, 3)

            # Testing to see if atom typs is switched
            test_ratio = rng.uniform()
            switch = 0
            for i, n in enumerate(ratio):
                if test_ratio > np.sum(ratio[:i]):
                    switch = i
            myatom_type = phantom_min + switch
            energy = phantom_energy[myatom_type]
            myatom_energy = energy[myatom_id]

            # MC moves
            for step in range(nstep):
                rand_atom_id = rng.choice(nphantoms, size=1)[0]
                if (actual_atoms[rand_atom_id] != 0):
                    continue

                rand_atom_pos = phantom_coords[rand_atom_id]
                rand_atom_pos = np.reshape(rand_atom_pos, 3)
                too_close = False
                for o, other in enumerate(actual_atoms):
                    if other == myatom_id or other == rand_atom_id:
                        continue
                    if other != 0:
                        check_pos = phantom_coords[o]
                        check_pos = np.reshape(check_pos, 3)
                        if (distance(check_pos, rand_atom_pos, lx, ly) < rc):
                            too_close = True
                            break
                if too_close:
                    continue

                # MC check
                rand_atom_energy = energy[rand_atom_id]
                delta_E = myatom_energy - rand_atom_energy
                proba = np.exp(delta_E/KbT)
                # print(proba, myatom_energy, rand_atom_energy)
                if (proba > 1.) or (rng.uniform() < proba):
                    actual_atoms[myatom_id] = 0
                    myatom_id = rand_atom_id
                    myatom_pos = np.reshape(rand_atom_pos, 3)
                    myatom_energy = rand_atom_energy
                    actual_atoms[myatom_id] = n+1
            actual_atoms[myatom_id] = -1
            myatom_pos = np.reshape(myatom_pos, 3)
            keep.append([myatom_type, myatom_pos[0], myatom_pos[1], myatom_pos[2]])
    # Regular MEAD
    else:
        print("Using old MEAD")
        actual_atoms = np.zeros(nphantoms)
        for p in range(phantom_min, phantom_max):
            energy = phantom_energy[p]
            minen = np.min(energy)
            for i, at in enumerate(phantom_coords):
                # print(p, energy[i], minen, np.abs((energy[i]-minen))/minen)
                if np.abs((energy[i] - minen)/minen) < thresh:
                    keep.append([p, at[0], at[1], at[2]])

    with open('data.min.lmp', 'w') as f:
        f.write("Minimum energy position from MEAD output\n")
        f.write("\n")
        f.write("{} atoms\n".format(len(keep)))
        f.write("{} atom types\n".format(phantom_max))
        f.write("\n")
        f.write("0 40.49 xlo xhi\n")
        f.write("0 40.49 ylo yhi\n")
        f.write("0 40.49 zlo zhi\n")
        f.write("\n")
        f.write("Atoms\n")
        f.write("\n")
        for i, at in enumerate(keep):
            f.write("{:d} {:d} {:d} {:f} {:f} {:f} {:f} 0 0 0\n".format(
                i+1, 0, at[0], 0, at[1], at[2], at[3]
                )
            )

    return


def create_system(lmp, nattype):
    # An example of system creation
    # lmp.command('lattice diamond 5.43  orient z 1 1 1 orient x 1 -1 0 orient y 1 1 -2 origin 0.01 0.01 0.01')
    lmp.command('lattice {:s} {:f} origin 0.01 0.01 0.01'.format(mead_lattice_create, mead_lattice_spacing))
    lmp.command('region mybox block 0 10 0 10 0 10')
    lmp.command('create_box {:d} mybox'.format(nattype))
    lmp.command('region mysurf block 0 10 0 10 0 5')
    lmp.command('create_atoms 1 region mysurf')

    # Loop over atom types
    for i in range(nattype):
        lmp.command('mass {:d} {:f}'.format(i+1, atmasses[i]))

    lmp.command('displace_atoms all move 0 0 0.5')
    lmp.command('write_data {:s}'.format('data.in.lmp'))

    return


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=None,
        dest='data',
        help='A default data file to start the simulation from'
    )
    parser.add_argument(
        "-r",
        "--rseed",
        dest="rseed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "-p",
        "--phantom",
        dest='phantom',
        type=str,
        default='vfm',
        help="Method to use for phantom insertion, vfm (default) or surface"
    )
    parser.add_argument(
        "-m",
        "--minimize",
        dest='min',
        type=str,
        default='fire',
        help="Method to use for minimization, fire (default) or tfmc"
    )
    parser.add_argument(
        "-w",
        "--write_peph",
        dest='write_peph',
        action='store_true',
        help="Writes dump files containing phantom atoms positions and energy."
    )
    # Temperature is in the mead_inputs.py file
    # parser.add_argument(
    #     "-t",
    #     "--temp",
    #     dest="temp",
    #     type=float,
    #     default=0.,
    #     help="Temperature for MC if any (default 0K = No MC)",
    # )


    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng()
    rseed = args.rseed if args.rseed else rng.integers(1000000)

    comm = MPI.COMM_WORLD
    me = comm.Get_rank()

    if me == 0:
        logging.info("Using seed {:d}".format(rseed))

    if args.phantom in ['vfm', 'surface']:
        if me == 0:
            logging.info("Using {:s} mode".format(args.phantom))
    else:
        sys.exit('Unknown method provided')

    lmp = lammps()

    # me = lmp.get_mpi_comm().rank

    # This is the previous potential model we used. An ADP model for Si A land Au
    # The use of KIM is very convenient here as it allows to define atoms by name
    # Only the model and units are critical to define
    lmp.command('kim init {:s} {:s}'.format(kim_model, units))
    lmp.command('processors * * 1')
    lmp.command('atom_style      full')
    lmp.command('atom_modify     map yes')
    lmp.command('boundary        p p m')
    lmp.command('comm_style      tiled')
    lmp.command('neigh_modify    page 500000 one 50000 check no')

    # System creation
    if args.data:
        if me == 0:
            logging.info("Reading data file {:s}".format(args.data))
        lmp.command('read_data {:s} nocoeff'.format(args.data))
    else:
        if me == 0:
            logging.info("Creating new surface from mead_create.py")
        create_system(lmp, nattype)

    lmp.command(' '.join(['kim', 'interactions']+atnames))

    # Preparing computation to increase box height
    lmp.command('compute         zmax all reduce max z')
    lmp.command('compute         Peall all pe/atom')

    # This is where the main MEAD parameters are read
    lmp.command('variable        radius_surf equal {:f}'.format(radius_surf))
    # lmp.command('variable        ratio equal {:f}'.format(ratio))
    lmp.command('variable        ecutoff index {:f}'.format(ecutoff))
    lmp.command('variable        RSeed equal {:d}'.format(rseed))
    lmp.command('variable        LoopMax index {:d}'.format(LoopMax))
    lmp.command('variable        Temp index {:f}'.format(Temp))

    lmp.command('thermo 1')
    lmp.command('reset_timestep  0')
    lmp.command('group       new empty')
    lmp.command('variable    newcount equal count(new)')
    lmp.command('print "# Step    Ntot       Nnew" file step.dat')

    # create_atoms still needs a lattice
    lmp.command('variable        lat index {:15.13f}'.format(lat))
    lmp.command('variable        dz equal v_lat')
    lmp.command('variable        dispx equal v_lat/2.')
    lmp.command('variable        dispy equal v_lat/2.')
    lmp.command('variable        dispz equal v_lat/2.')
    lmp.command('min_style fire')
    lmp.command('lattice         sc ${lat}')

    phantom_energy = {}
    phantom_min = isghost.index(True) + 1
    phantom_max = len(isghost)
    if me == 0:
        print("Phantom min/max = {:d} {:d}".format(phantom_min, phantom_max))
    nloop = 0

    while True:
        nloop += 1
        if me == 0:
            logging.info("Starting loop iteration number {:d}".format(nloop))
        # Construct surface and write the data.
        surface_box = compute_surface(lmp, radius_surf, 0.1, 100)
        surface_box = comm.bcast(surface_box, root=0)

        nreal_atoms = lmp.get_natoms()
        if me == 0:
            logging.info('NReal atoms = {:d}'.format(nreal_atoms))
        lmp.command('balance         1. rcb')
        lmp.command('run 0')

        # FORMER SURFACE USE
        if args.phantom == 'vfm':
            (xlo, xhi, ylo, yhi, zlo, zhi) = surface_box
            lmp.command('variable        dx1 equal {:f}'.format(xlo))
            lmp.command('variable        dx2 equal {:f}'.format(xhi))
            lmp.command('variable        dy1 equal {:f}'.format(ylo))
            lmp.command('variable        dy2 equal {:f}'.format(yhi))
            lmp.command('variable        dz1 equal {:f}'.format(zlo))
            lmp.command('variable        dz2 equal {:f}'.format(zhi))

            # Filling the region
            lmp.command('region          VFM block ${dx1} ${dx2} ${dy1} ${dy2} ${dz1} ${dz2} units box')
            lmp.command('create_atoms    {:d} region VFM'.format(phantom_min))
            lmp.command('region          VFM delete')
        elif args.phantom == 'surface':
            lmp.command('create_atoms {:d} mesh surface.stl units box'.format(phantom_min))
            lmp.command('run 0')
            lmp.command('write_data data.mesh.lmp')

        # This is where we get the atom infos from LAMMPS Only the total number
        # of atoms is needed, we get the info of which is phantom which is not
        # from the arrays using the atom types. They should be initialised to
        # the first phantom type.
        #
        # Note that we need to convert the ctypes array to numpy arrays using
        # np.ctypeslib. That's where the ntotal number if useful.

        ntotal_atoms = lmp.get_natoms()
        # Ids
        atom_ids = lmp.gather("id", 0, 1)
        atom_ids = np.ctypeslib.as_array(atom_ids)
        # Types
        atom_types = lmp.gather("type", 0, 1)
        atom_types = np.ctypeslib.as_array(atom_types)
        # Position
        atom_coords = lmp.gather("x", 1, 3)
        atom_coords = np.ctypeslib.as_array(atom_coords)
        atom_coords = atom_coords.reshape(ntotal_atoms, 3)

        phantom_ids = atom_ids[atom_types == phantom_min]
        phantom_coords = atom_coords[atom_types == phantom_min][:]
        if me == 0:
            print("Retrieved {:d} atoms out of {:d}".format(phantom_coords.shape[0], atom_coords.shape[0]))

        for i in range(phantom_min, phantom_max+1):
            lmp.command('neigh_modify    exclude type {:d} {:d}'.format(i, i))

        lmp.command('group           Ph type {:d}'.format(phantom_min))
        lmp.command('compute         PePh Ph pe/atom')
        lmp.command('thermo          100')
        lmp.command('thermo_style    custom step temp epair etotal press pxx pyy v_newcount')

        # Displace atoms randomly
        # If surface method, there is little need to move atoms 
        lmp.command('displace_atoms Ph random ${dispx} ${dispy} ${dispz} ${RSeed} units box')


        for p in range(phantom_min, phantom_max+1):
            lmp.command('set group Ph type {:d}'.format(p))
            if args.write_peph:
                lmp.command('dump  1 Ph custom 1 dump.peph.lammpstrj.type.{:d}.{:d} id type x y z c_PePh'.format(p, nloop))
            lmp.command('run 0')
            if args.write_peph:
                lmp.command('undump  1')
            energy = lmp.gather("c_PePh", LMP_STYLE_ATOM, LMP_TYPE_VECTOR)
            energy = np.ctypeslib.as_array(energy)

            # The size of the array won't change with phantom type number as
            # the number of atoms is constant.
            phantom_energy[p] = energy[atom_types >= phantom_min]

        if me == 0:
            compute_minima(lmp, rng, phantom_coords, phantom_energy, phantom_min, phantom_max, Temp)

        lmp.command('delete_atoms    group Ph')
        lmp.command('uncompute       PePh')

        # This is just a sanyty check
        lmp.command('run 0')

        # This is where the new atoms are definitely added to the system
        # (safer than manipulating the data in LAMMPS since it uses pointers)
        lmp.command('read_data       data.min.lmp add append group NEW')
        lmp.command('delete_atoms    overlap 1. NEW NEW')
        lmp.command('reset_atoms id sort yes')

        lmp.command('group           new union new NEW')

        lmp.command('neigh_modify    exclude none')
        lmp.command('neigh_modify    delay 1 every 1 check no')

        for p in range(phantom_min, phantom_max+1):
            lmp.command('group TY{:d} type {:d}'.format(p, p))
            newtype = phantom_to_real[p-1]
            lmp.command('set             group TY{:d} type {:d}'.format(p, newtype))
            lmp.command('group           TY{:d} clear'.format(p))

        lmp.command('neigh_modify    delay 10 every 1 check yes')
        lmp.command('thermo          100')
        # Minimization/tfMC
        if args.min == 'tfmc':
            lmp.command('fix TFMC all tfmc 0.15 ${Temp} ${RSeed} com 1 1 1 rot')
            lmp.command('dump   1 all custom 10 dump.tfmc.lammpstrj.{:d} id type x y z c_Peall'.format(nloop))
            lmp.command('run 1000')
            lmp.command('unfix TFMC')
            lmp.command('undump 1')
        elif args.min == 'fire':
            lmp.command('minimize 1e-10 1e-12 10000 10000')

        lmp.command('group           NEW clear')

        # Compute zmax coordinate and writes intermediate configuration
        lmp.command('variable        zmax equal c_zmax')
        lmp.command('variable        d equal lz-v_zmax')
        lmp.command('thermo_style    custom step c_zmax')

        lmp.command('dump            1 all custom 1 dump.end.lammpstrj.{:d} id type x y z c_Peall'.format(nloop))
        lmp.command('dump_modify 1 sort id')
        lmp.command('run             0')
        lmp.command('undump          1')

        lmp.command('if "$d < 3." then "change_box all z delta 0. 10. units box"')

        totatoms = lmp.get_natoms()
        if me == 0:
            logging.info("Currently there are {:d} atoms in the simulation.".format(totatoms))
        if (totatoms > natom_max) or (nloop > nloop_max):
            if me == 0:
                logging.info("Finishing.")
            break

    lmp.command('write_data       data.end.lmp')

    return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("User interruption.")
