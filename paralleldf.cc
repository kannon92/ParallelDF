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
    boost::shared_ptr<Wavefunction> scf(new scf::RHF(ref_wfn, options, PSIO::shared_object()));
    double scf_energy = scf->compute_energy();

    int print = options.get_int("PRINT");
    SharedMatrix F_target = ref_wfn->Fa();
    boost::shared_ptr<BasisSet> auxiliary = BasisSet::pyconstruct_orbital(ref_wfn->molecule(), "DF_BASIS_SCF",options.get_str("DF_BASIS_SCF"));
    boost::shared_ptr<JK> JK(new ParallelDFJK(ref_wfn->basisset(), auxiliary));

    JK->set_memory(Process::environment.get_memory() * 0.5);
    JK->initialize();
    std::vector<boost::shared_ptr<Matrix> >&Cl = JK->C_left();
    Cl.clear();
    Cl.push_back(ref_wfn->Ca_subset("SO", "OCC"));
    JK->compute();

    SharedMatrix F_mine = F_target->clone();

    F_mine->zero();
    F_mine->copy(ref_wfn->H());
    SharedMatrix G = JK->J()[0];
    G->scale(2.0);
    G->subtract(JK->K()[0]);
    F_mine->add(G);
    F_mine->subtract(F_target);

    outfile->Printf("\n F_mine %8.8f", F_mine->rms());

    




    return ref_wfn;
}

}} // End namespaces

