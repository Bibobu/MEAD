#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A script to compute a surface using Ovito.
"""

import argparse
import vtk
import math
from ovito.io import import_file, export_file
from ovito.modifiers import ConstructSurfaceModifier, SliceModifier


def mymod(frame, data):
    print("Evaluating frame number {:<d}".format(frame))


def main():
    parser = argparse.ArgumentParser(description="Makes a surface mesh.")
    parser.add_argument(
        "-f", "--filename", default="in.lmp", help="Data file to be read."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="MySurface",
        help="Name of the ouput files (Index indicates smoothness interation)."
    )
    parser.add_argument(
        "-r",
        "--radius",
        default=1,
        type=float,
        help="Radius parameter for Alpha probing.",
    )
    parser.add_argument(
        "-s",
        "--smoothness",
        default=0,
        type=int,
        help="Number of smoothness iteration (alpha probing only).",
    )
    parser.add_argument(
        "--scaling",
        default=1.,
        type=float,
        dest='scaling',
        help="Radius scaling for gaussian method, default=1. (gaussian only).",
    )
    parser.add_argument(
        "--iso",
        default=0.6,
        type=float,
        dest='isolevel',
        help="Isolevel, default=0.6 (gaussian only).",
    )
    parser.add_argument(
        "--grid-resolution",
        default=100,
        type=int,
        dest='resolution',
        help="grid resolution, default=100 (gaussian only).",
    )
    parser.add_argument(
        "--meshformat",
        default="vtk",
        type=str,
        help="File format for the mesh (vtk or stl).",
    )
    parser.add_argument(
        "-g",
        "--gaussian",
        action='store_true',
        help="Use gaussian method for surface making."
    )
    parser.add_argument(
        "--pbc",
        default="",
        type=str,
        help="Dimensions with PBC (default None, provide 1 string containing x, y and/or z)"
    )
    parser.add_argument(
        "--dz",
        default=0.,
        type=float,
        help="Altitude to add to the surface (default 0.).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='count',
        default=0,
        help="Display more informations..."
        )
    args = parser.parse_args()
    datafile = args.filename
    output = args.output
    verbose = args.verbose
    pbc = args.pbc
    # Probe method arguments
    aradius = args.radius
    smoothness = args.smoothness
    # Gaussien method arguments
    gaussian = args.gaussian
    scaling = args.scaling
    isolevel = args.isolevel
    resolution = args.resolution
    dz = args.dz

    if verbose:
        print("Launching!")
        print("Reading data!")
    pipeline = import_file(datafile)

    if verbose:
        print("Making Surface mod!")
    if gaussian:
        SURFmod = ConstructSurfaceModifier(
                method=ConstructSurfaceModifier.Method.GaussianDensity, radius_scaling=scaling, isolevel=isolevel, grid_resolution=resolution,
                )
        # SURFmod = ConstructSurfaceModifier(
        #         method=ConstructSurfaceModifier.Method.GaussianDensity,
        #         )
    else:
        SURFmod = ConstructSurfaceModifier(
            radius=aradius,
            smoothing_level=smoothness,
        )
    pbcx = 'x' in pbc
    pbcy = 'y' in pbc
    pbcz = 'z' in pbc

    print("PBC ARE {} {} {}".format(pbcx, pbcy, pbcz))
    pipeline.modifiers.append(SURFmod)
    data = pipeline.compute()

    # This pbc setting is needed else the surface is completed with
    # new edges on the pbc during export.
    surface = data.surfaces["surface_"]
    surface.domain_.pbc = (pbcx, pbcy, pbcz)
    coords = data.particles.position
    zmax = max(coords[:, 2])

    SLICEmod = SliceModifier(
            normal=(0, 0, 1),
            inverse=True,
            distance=0.5*zmax
            )

    pipeline.modifiers.append(SLICEmod)
    data = pipeline.compute()
    surface = data.surfaces["surface_"]
    surface.domain_.pbc = (pbcx, pbcy, pbcz)
    vertex_coords = surface.vertices['Position']
    keep = vertex_coords[vertex_coords[:, 2] > 0.5*zmax]

    if verbose:
        print("Saving in {:}.vtk".format(output))
    outfile = ''.join([output, '.vtk'])
    export_file(surface, outfile, "vtk/trimesh", include_caps=False)
    if args.meshformat == 'stl':
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(outfile)
        reader.Update()
        filt = vtk.vtkGeometryFilter()
        filt.SetInputConnection(reader.GetOutputPort())
        writer = vtk.vtkSTLWriter()
        writer.SetInputConnection(filt.GetOutputPort())
        writer.SetFileName(''.join([output, '.stl']))
        writer.Write()

    xlo, xhi = min(keep[:, 0]), max(keep[:, 0])
    ylo, yhi = min(keep[:, 1]), max(keep[:, 1])
    zlo, zhi = min(keep[:, 2]), max(keep[:, 2])
    if zhi-zlo < dz:
        zhi = zlo+dz
    # zhi = zhi + dz

    with open(''.join([output, '.geom']), 'w') as f:
        f.write("{:18f}\n".format(xlo))
        f.write("{:18f}\n".format(xhi))
        f.write("{:18f}\n".format(ylo))
        f.write("{:18f}\n".format(yhi))
        f.write("{:18f}\n".format(zlo))
        f.write("{:18f}\n".format(zhi))

    return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("User interruption.")
