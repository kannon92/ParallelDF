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
#include <libthce/thce.h>
#include <libthce/lreri.h>
#include "paralleldfjk.h"
#include "paralleldfmo.h"
#include <ga.h>
#include <macdecls.h>

using namespace boost;
void* replace_malloc(size_t bytes, int align, char *name)
{
    return malloc(bytes);
}
void replace_free(void *ptr)
{
    free(ptr);
}

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
    GA_Initialize();
    GA_Register_stack_memory(replace_malloc, replace_free);
        
    boost::shared_ptr<BasisSet> auxiliary = BasisSet::pyconstruct_orbital(ref_wfn->molecule(), "DF_BASIS_SCF",options.get_str("DF_BASIS_SCF"));
    int naux = auxiliary->nbf();
    SharedMatrix Ca = ref_wfn->Ca();
    SharedMatrix Ca_ao(new Matrix("CA_AO", ref_wfn->nso(), ref_wfn->nmo()));
    int nso = ref_wfn->nso();
    int nmo = ref_wfn->nmo();
    for (size_t h = 0, index = 0; h < ref_wfn->nirrep(); ++h){
        for (size_t i = 0; i < ref_wfn->nmopi()[h]; ++i){
            size_t nao = ref_wfn->nso();
            size_t nso = ref_wfn->nsopi()[h];
    
            if (!nso) continue;
    
            C_DGEMV('N',nao,nso,1.0,ref_wfn->aotoso()->pointer(h)[0],nso,&Ca->pointer(h)[0][i],ref_wfn->nmopi()[h],0.0,&Ca_ao->pointer()[0][index],ref_wfn->nmopi().sum());

            index += 1;
        }
    }
    boost::shared_ptr<DFERI> df = DFERI::build(ref_wfn->basisset(), auxiliary, options);

    df->set_C(Ca_ao);
    df->add_space("ALL", 0, nmo);
    df->add_pair_space("B", "ALL", "ALL");
    df->set_memory(Process::environment.get_memory() / 8L);
    df->compute();
    boost::shared_ptr<psi::Tensor> B = df->ints()["B"];
    FILE* Bf = B->file_pointer();
    SharedMatrix Bpq(new Matrix("Bpq", nmo * nmo, naux));
    fseek(Bf, 0L, SEEK_SET);
    fread(&Bpq->pointer()[0][0], sizeof(double), naux * nmo * nmo, Bf);
    //Bpq->print();

    ParallelDFMO DFMO = ParallelDFMO(ref_wfn->basisset(), auxiliary);
    DFMO.set_C(Ca_ao);
    DFMO.compute_integrals();
    int MY_DF = DFMO.Q_PQ();
    //GA_Print(MY_DF);
    //Bpq->print();


    ///// Compute the PSI4 DFJK
    ///// Compare against this code in all stages
    boost::shared_ptr<JK> JK_DFJK(new DFJK(ref_wfn->basisset(), auxiliary));
    JK_DFJK->set_memory(Process::environment.get_memory() * 0.5);
    JK_DFJK->initialize();
    std::vector<boost::shared_ptr<Matrix> >&Cl = JK_DFJK->C_left();
    Cl.clear();
    Cl.push_back(ref_wfn->Ca_subset("SO", "OCC"));
    JK_DFJK->compute();
    SharedMatrix F_target = JK_DFJK->J()[0];
    F_target->scale(2.0);
    F_target->print();
    //F_target->subtract(JK_DFJK->K()[0]);

    /// Compute the ParallelDFJK
    boost::shared_ptr<JK> JK_Parallel(new ParallelDFJK(ref_wfn->basisset(), auxiliary));
    JK_Parallel->set_memory(Process::environment.get_memory() * 0.5);
    JK_Parallel->initialize();
    std::vector<boost::shared_ptr<Matrix> >&C_parallel = JK_Parallel->C_left();
    C_parallel.push_back(ref_wfn->Ca_subset("SO", "OCC"));
    JK_Parallel->compute();


    SharedMatrix F_mine = JK_Parallel->J()[0];
    F_mine->scale(2.0);
    F_mine->print();
    //F_mine->subtract(JK_Parallel->K()[0]);
    F_mine->subtract(F_target);

    outfile->Printf("\n F_mine %8.8f", F_mine->rms());
    if(F_mine->rms() > 1e-4)
        throw PSIEXCEPTION("Serial DF and Parallel DF do not agree on F");

    GA_Terminate();
    return ref_wfn;
}

}} // End namespaces

