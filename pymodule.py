#
# @BEGIN LICENSE
#
# paralleldf by Psi4 Developer, a plugin to:
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2016 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

import psi4
import re
import os
import inputparser
import math
import warnings
import driver
from molutil import *
import p4util
from p4util.exceptions import *


def run_paralleldf(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    paralleldf can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('paralleldf')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    #psi4.set_global_option('BASIS', 'sto-3g')
    psi4.set_local_option('MYPLUGIN', 'PRINT', 1)

    # Compute a SCF reference, a wavefunction is return which holds the molecule used, orbitals
    # Fock matrices, and more
    print('Attention! This SCF may be density-fitted.')
    ref_wfn = kwargs.get('ref_wfn', None)
    if ref_wfn is None:
        ref_wfn = driver.scf_helper(name, **kwargs)

    # Call the Psi4 plugin
    # Please note that setting the reference wavefunction in this way is ONLY for plugins
    paralleldf_wfn = psi4.plugin('paralleldf.so', ref_wfn)

    return paralleldf_wfn


# Integration with driver routines
driver.procedures['energy']['paralleldf'] = run_paralleldf


def exampleFN():
    # Your Python code goes here
    pass
