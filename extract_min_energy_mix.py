#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import numpy as np


class Box:
    def __init__(self, xlo=0., xhi=1., ylo=0., yhi=1., zlo=0, zhi=1., xy=0., xz=0., yz=0.):
        self.xlo = xlo
        self.xhi = xhi
        self.ylo = ylo
        self.yhi = yhi
        self.zlo = zlo
        self.zhi= zhi
        self.xy = xy
        self.xz = xz
        self.yz = yz


class Conf:
    def __init__(self, timestep=0, atoms=[], box = Box()):
        self.timestep = timestep
        self.atoms = atoms
        self.box = box


class Dump:
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        self.f = open(self.filename, 'r')
        return self

    def __next__(self):
        line = self.f.readline().strip()
        if not 'TIMESTEP' in line.split():
            raise StopIteration
        else:
            atoms = []
            timestep = int(self.f.readline().strip())
            line = self.f.readline().strip()
            nat = int(self.f.readline().strip())
            line = self.f.readline().strip()
            xlo, xhi = list(map(float, self.f.readline().strip().split()))
            ylo, yhi = list(map(float, self.f.readline().strip().split()))
            zlo, zhi = list(map(float, self.f.readline().strip().split()))
            at_attributes = self.f.readline().strip().split()[2:]
            for _ in range(nat):
                at = {}
                line = self.f.readline().strip().split()
                for attr, val in zip(at_attributes, line):
                    if attr in ['id', 'type']:
                        at[attr] = int(val)
                    else:
                        at[attr] = float(val)
                atoms.append(at)
            return Conf(timestep, atoms, Box(xlo, xhi, ylo, yhi, zlo, zhi))

def compute_dist(at1, at2, lx, ly, lz):
    dx = at1['x']-at2['x']
    dx = dx - int(dx/lx)*lx
    dy = at1['y']-at2['y']
    dy = dy - int(dy/ly)*ly
    dz = at1['z']-at2['z']
    dz = dz - int(dz/lz)*lz
    dr = dx**2 + dy**2 + dz**2
    return dr**0.5


def find_closest(at, list, lx, ly, lz):
    nbs = sorted(list, key=lambda x: compute_dist(at, x, lx, ly, lz))
    return nbs[0]


def main():

    parser = argparse.ArgumentParser(
        description="Extract min energy info"
    )
    parser.add_argument(
        "-i",
        "--inputfile",
        nargs='+',
        dest="inputfile",
        help="File to read",
    )
    parser.add_argument(
        "-o",
        "--outputfile",
        dest="outputfile",
        help="File to write",
    )
    parser.add_argument(
        "-n",
        "--natoms",
        dest="nat",
        default=0,
        type=int,
        help="Number of atoms to keep [default 0]",
    )
    parser.add_argument(
        "-p",
        "--prop",
        dest="prop",
        default=0.,
        type=float,
        help="Proportion of atoms to keep (< 1.) [default 0.]",
    )
    parser.add_argument(
        "-r",
        "--rc",
        dest="rc",
        default=0.,
        type=float,
        help="Minimal distance between atoms (skipped if 0.) [default 0.]",
    )
    parser.add_argument(
        "--ratio",
        dest="ratio",
        default=0.,
        type=float,
        help="Max number of atoms of the 2nd species wrt the 1st one [default 0.]",
    )
    parser.add_argument(
        "-e",
        "--etol",
        dest="etol",
        default=0.01,
        type=float,
        help="Maximal relative energy tolerance wrt. to the minima [default 0.01]",
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="format",
        default="atomic",
        type=str,
        help="Format to use for data file writing",
    )
    args = parser.parse_args()
    if len(args.inputfile) == 1 and args.ratio:
        sys.exit("Cannot have ratio with only one dumpe file.")

    rc = args.rc
    n = args.nat
    etol = args.etol
    rng = np.random.default_rng()

    # A placeholder for final atom selection of all dumpfiles
    myfinalatoms = []
    for filenumber, inputfile in enumerate(args.inputfile):
        mydump = Dump(inputfile)
        myatoms = []
        for c in mydump:
            print("Reading TIMESTEP {}".format(c.timestep), end='\r')
            for at in c.atoms:
                myatoms.append(at)
        print()
        totatoms = len(myatoms)
        print("{} initial atoms.".format(totatoms))

        if not filenumber:
            lx = c.box.xhi - c.box.xlo
            ly = c.box.yhi - c.box.ylo
            lz = c.box.zhi - c.box.zlo
            print("1st file input")
            # Atoms to keep
            if n:
                print("prop arg ignored")
            else:
                if args.prop > 1.:
                    sys.exit("args.prop > 1.")
                n = int(totatoms*args.prop)
            if myatoms:
                myatoms = sorted(myatoms, key=lambda x: x['c_PePh'])[0:n]
            else:
                sys.exit("No atom in selection.")
            nat = len(myatoms)
            reftype = myatoms[0]['type']
            print("{} atoms after 1st cleanup.".format(nat))
            print("Max energy = {}".format(max([x['c_PePh'] for x in myatoms])))
            print("Min energy = {}".format(min([x['c_PePh'] for x in myatoms])))
            minen = min([x['c_PePh'] for x in myatoms])

            # Sticking around the global minima
            myatoms = [x for x in myatoms if abs((x['c_PePh']-minen)/minen) < etol]
            nat = len(myatoms)
            print("{} atoms after 2st cleanup.".format(nat))
            print("Max energy = {}".format(max([x['c_PePh'] for x in myatoms])))
            print("Min energy = {}".format(min([x['c_PePh'] for x in myatoms])))

            if rc:
                myatoms = sorted(myatoms, key=lambda x: x['c_PePh'])
                for i, at1 in enumerate(myatoms):
                    for at2 in myatoms[i:]:
                        if at1 is at2:
                            continue
                        else:
                            d = compute_dist(at1, at2, lx, ly, lz)
                            if d < rc:
                                myatoms.remove(at2)
                                # print(i, len(myfinalatoms), d)

            nat = len(myatoms)
            print("{} atoms after 3nd cleanup.".format(nat))
            maxenref = max([x['c_PePh'] for x in myatoms])
            minenref = min([x['c_PePh'] for x in myatoms])
            print("Max energy = {}".format(maxenref))
            print("Min energy = {}".format(minenref))
            natref = nat
            myfinalatoms = myatoms
        else:
            print("File {:d} input.".format(filenumber+1))
            # addtot = int(np.rint(natref/(1-args.ratio)))
            # addmax = max(0, addtot-nat)
            # if not addmax:
            #     break

            if myatoms:
                myatoms = [x for x in myatoms if x['c_PePh'] < maxenref]
                myatoms = sorted(myatoms, key=lambda x: x['c_PePh'])  # [0:addmax]
            else:
                sys.exit("No atom in selection.")

            # Since we expect the second species to be in much lower proportion
            # wrt the first one, the cleaning of atoms is much simpler.
            # This might need some change in the future.
            nattemp = len(myatoms)
            print("{} atoms after 1st cleanup.".format(nattemp))
            print("Max energy = {}".format(max([x['c_PePh'] for x in myatoms])))
            print("Min energy = {}".format(min([x['c_PePh'] for x in myatoms])))
            minen = min([x['c_PePh'] for x in myatoms])

            # myfinalatoms.extend(myatoms)
            # Finding reftype atoms nearest neighbor and giving it 'ratio' chance
            # to be replaced by its nearest neighbor of another type
            for n, at in enumerate(myfinalatoms):
                if at['type'] == reftype:
                    if rng.random() < args.ratio:
                        new_at = find_closest(at, myatoms, lx, ly, lz)
                        myfinalatoms[n] = new_at
                        print('Changed atom {:d}'.format(n))


    # if rc:
    #     myfinalatoms = sorted(myfinalatoms, key=lambda x: x['c_PePh'])
    #     for i, at1 in enumerate(myfinalatoms):
    #         for at2 in myfinalatoms[i:]:
    #             if at1 is at2:
    #                 continue
    #             else:
    #                 d = compute_dist(at1, at2, lx, ly, lz)
    #                 if d < rc:
    #                     myfinalatoms.remove(at2)
    #                     # print(i, len(myfinalatoms), d)


    # We remove secondary to stick to the desired ratio
    myatoms = sorted(myfinalatoms, key=lambda x: x['c_PePh'])
    # curref = len([x for x in myatoms if x['type'] == reftype])
    # addtot = int(np.rint(curref/(1-args.ratio)))
    # while len(myatoms) > addtot:
    #     topop = max([x for x, y in enumerate(myatoms) if y['type'] != reftype])
    #     myatoms.pop(topop)

    ntype = max([at['type'] for at in myatoms])
    myatoms = sorted(myatoms, key=lambda x: x['id'])
    nat=len(myatoms)
    zhi = max(c.box.zhi, max([at['z'] for at in myatoms]))

    # Writing
    with open(args.outputfile, 'w') as f:
        f.write("Some stuff\n")
        f.write("\n")
        f.write("{} atoms\n".format(nat))
        f.write("{} atom types\n".format(ntype))
        f.write("\n")
        f.write("{} {} xlo xhi\n".format(c.box.xlo, c.box.xhi))
        f.write("{} {} ylo yhi\n".format(c.box.ylo, c.box.yhi))
        f.write("{} {} zlo zhi\n".format(c.box.zlo, zhi))
        f.write("\n")
        f.write("Atoms\n")
        f.write("\n")
        if args.format == "full":
            try:
                for i, at in enumerate(myatoms):
                    f.write("{:d} {:d} {:d} {:f} {:f} {:f} {:f} 0 0 0\n".format(
                        i+1, 0, at['type'], at['q'], at['x'], at['y'], at['z']
                        )
                    )
            except Exception:
                for i, at in enumerate(myatoms):
                    f.write("{:d} {:d} {:d} {:f} {:f} {:f} {:f} 0 0 0\n".format(
                        i+1, 0, at['type'], 0, at['x'], at['y'], at['z']
                        )
                    )
        else:
            for i, at in enumerate(myatoms):
                f.write("{:d} {:d} {:f} {:f} {:f} 0 0 0\n".format(
                    i+1, at['type'], at['x'], at['y'], at['z']
                    )
                )


    return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("User interruption.")
