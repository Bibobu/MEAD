# MEAD

This repository contains a minimum working example for the MEAD procedure described in [Karewar et al. (2024)](https://doi.org/10.1016/j.surfcoat.2024.131462).

To run the scripts you will need:

* The LAMMPS software
* A Python interpreter accessible through the shell
* The Ovito Python library and the numpy Python library
* The KIM package compiled in your LAMMPS executable (optional for general cases)
* The kim-api-collection-management and models that you want to use (optional for general cases)

The way the example work is first by creating an simple FCC surface through the `in.create.lmp` script. That is:
```
lmp -i in.create.lmp
```

And then execute the main MEAD script:
```
lmp -i in_MEAD.lmp
```

The script calls the two python scripts from LAMMPS instance and dumps the growing surface in successive files `dump.N.lammpstrj` with `N` being the loop index.

These examples can be edited to be compatible with any model you would like to use.
