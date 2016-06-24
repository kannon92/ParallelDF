/*
 * @BEGIN LICENSE
 *
 * paralleldf by Psi4 Developer, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2016 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include <psi4-dec.h>
#include <libparallel/parallel.h>
#include <liboptions/liboptions.h>
#include <libmints/mints.h>
#include <libpsio/psio.hpp>
#include <libscf_solver/rhf.h>
#include <libfock/jk.h>
#include "paralleldfjk.h"

using namespace boost;

namespace psi{ namespace paralleldf {

extern "C"
int read_options(std::string name, Options& options)
{
    if (name == "PARALLELDF"|| options.read_globals()) {
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 1);
    }

    return true;
}

extern "C"
SharedWavefunction paralleldf(SharedWavefunction ref_wfn, Options& options)
{

    int print = options.get_int("PRINT");
    boost::shared_ptr<BasisSet> auxiliary = BasisSet::pyconstruct_orbital(ref_wfn->molecule(), "DF_BASIS_SCF",options.get_str("DF_BASIS_SCF"));

    /// Compute the PSI4 DFJK
    /// Compare against this code in all stages
    boost::shared_ptr<JK> JK_DFJK(new DFJK(ref_wfn->basisset(), auxiliary));
    JK_DFJK->set_memory(Process::environment.get_memory() * 0.5);
    JK_DFJK->initialize();
    std::vector<boost::shared_ptr<Matrix> >&Cl = JK_DFJK->C_left();
    Cl.clear();
    Cl.push_back(ref_wfn->Ca_subset("SO", "OCC"));
    JK_DFJK->compute();
    SharedMatrix F_target = JK_DFJK->J()[0];

    /// Compute the ParallelDFJK
    boost::shared_ptr<JK> JK_Parallel(new ParallelDFJK(ref_wfn->basisset(), auxiliary));
    JK_Parallel->set_memory(Process::environment.get_memory() * 0.5);
    JK_Parallel->initialize();
    std::vector<boost::shared_ptr<Matrix> >&C_parallel = JK_Parallel->C_left();
    C_parallel.push_back(ref_wfn->Ca_subset("SO", "OCC"));
    JK_Parallel->compute();


    SharedMatrix F_mine = JK_Parallel->J()[0];
    //G->scale(2.0);
    //G->subtract(JK_Parallel->K()[0]);
    F_mine->subtract(F_target);

    outfile->Printf("\n F_mine %8.8f", F_mine->rms());

    




    return ref_wfn;
}

}} // End namespaces

