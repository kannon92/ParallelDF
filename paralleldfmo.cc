#include <libfock/jk.h>
#include <libmints/mints.h>
#include <libmints/sieve.h>
#include <psifiles.h>
#include "paralleldfjk.h"
#include <psi4-dec.h>
#include <lib3index/3index.h>
#include <libqt/qt.h>
#include <omp.h>
//#include <mpi.h>
#include "paralleldfmo.h"
#include <ga.h>
#include <macdecls.h>
namespace psi { namespace paralleldf {

ParallelDFMO::ParallelDFMO(boost::shared_ptr<BasisSet> primary, boost::shared_ptr<BasisSet> auxiliary) : primary_(primary), auxiliary_(auxiliary)
{
    outfile->Printf("\n ParallelDFJK");
}
int ParallelDFMO::J_one_half()
{
    // Everybody likes them some inverse square root metric, eh?

    int nthread = 1;
    #ifdef _OPENMP
        nthread = omp_get_max_threads();
    #endif

    int naux = auxiliary_->nbf();

    boost::shared_ptr<Matrix> J(new Matrix("J", naux, naux));
    double** Jp = J->pointer();

    if(GA_Nodeid() == 0)
    {
        boost::shared_ptr<IntegralFactory> Jfactory(new IntegralFactory(auxiliary_, BasisSet::zero_ao_basis_set(), auxiliary_, BasisSet::zero_ao_basis_set()));
        std::vector<boost::shared_ptr<TwoBodyAOInt> > Jeri;
        for (int thread = 0; thread < nthread; thread++) {
            Jeri.push_back(boost::shared_ptr<TwoBodyAOInt>(Jfactory->eri()));
        }

        std::vector<std::pair<int, int> > Jpairs;
        for (int M = 0; M < auxiliary_->nshell(); M++) {
            for (int N = 0; N <= M; N++) {
                Jpairs.push_back(std::pair<int,int>(M,N));
            }
        }
        long int num_Jpairs = Jpairs.size();

        #pragma omp parallel for schedule(dynamic) num_threads(nthread)
        for (long int PQ = 0L; PQ < num_Jpairs; PQ++) {

            int thread = 0;
            #ifdef _OPENMP
                thread = omp_get_thread_num();
            #endif

            std::pair<int,int> pair = Jpairs[PQ];
            int P = pair.first;
            int Q = pair.second;

            Jeri[thread]->compute_shell(P,0,Q,0);

            int np = auxiliary_->shell(P).nfunction();
            int op = auxiliary_->shell(P).function_index();
            int nq = auxiliary_->shell(Q).nfunction();
            int oq = auxiliary_->shell(Q).function_index();

            const double* buffer = Jeri[thread]->buffer();

            for (int p = 0; p < np; p++) {
            for (int q = 0; q < nq; q++) {
                Jp[p + op][q + oq] =
                Jp[q + oq][p + op] =
                    (*buffer++);
            }}
        }
        Jfactory.reset();
        Jeri.clear();

        // > Invert J < //

        J->power(-1.0/2.0, 1e-10);
        int dims[2];
        int chunk[2];
        dims[0] = naux;
        dims[1] = naux;
        chunk[0] = -1;
        chunk[1] = -1;
        GA_J_onehalf_ = NGA_Create(C_DBL, 2, dims, (char *)"J_1/2", chunk);
        if(not GA_J_onehalf_)
            throw PSIEXCEPTION("Failure in creating J_^(-1/2) in GA");
        for(int me = 0; me < GA_Nnodes(); me++)
        {
            int begin_offset[2];
            int end_offset[2];
            NGA_Distribution(GA_J_onehalf_, me, begin_offset, end_offset);
            NGA_Put(GA_J_onehalf_, begin_offset, end_offset, J->pointer()[0], &naux);
        }
    }
    GA_Print(GA_J_onehalf_);
    GA_Print_distribution(GA_J_onehalf_);

    return GA_J_onehalf_;
}
}}
