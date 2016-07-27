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
    memory_ = Process::environment.get_memory();
}
void ParallelDFMO::compute_integrals()
{
    transform_integrals();
}
void ParallelDFMO::transform_integrals()
{
    // > Sizing < //

    int nso = primary_->nbf();
    int naux = auxiliary_->nbf();
    nmo_ = Ca_->colspi()[0];

    // > Threading < //

    int nthread = 1;
    #ifdef _OPENMP
        nthread = omp_get_max_threads();
    #endif

    // > Maximum orbital sizes < //

    size_t max1 = nmo_;
    size_t max12 = max1 * max1;

    //for (int i = 0; i < pair_spaces_order_.size(); i++) {
    //    std::string name = pair_spaces_order_[i];
    //    std::string space1 = pair_spaces_[name].first;
    //    std::string space2 = pair_spaces_[name].second;

    //    int size1 = spaces_[space1].second - spaces_[space1].first;
    //    int size2 = spaces_[space2].second - spaces_[space2].first;

    //    size_t size12 = size1 * (size_t) size2;

    //    max1 = (max1 < size1 ? size1 : max1);
    //    max12 = (max12 < size12 ? size12 : max12);
    //}

    // > Row requirements < //

    unsigned long int per_row = 0L;
    // (Q|mn)
    per_row += nso * (unsigned long int) nso;
    // (Q|mi)
    per_row += max1 * (unsigned long int) nso;
    // (Q|ia)
    per_row += max12;

    // > Maximum number of rows < //

    unsigned long int max_rows = (memory_ / per_row);
    //max_rows = 3L * auxiliary_->max_function_per_shell(); // Debug
    if (max_rows < auxiliary_->max_function_per_shell()) {
        throw PSIEXCEPTION("Out of memory in DFERI.");
    }
    max_rows = (max_rows > auxiliary_->nbf() ? auxiliary_->nbf() : max_rows);

    // > Shell block assignments < //

    std::vector<int> shell_starts;
    shell_starts.push_back(0);
    int fcount = auxiliary_->shell(0).nfunction();
    for (int Q = 1; Q < auxiliary_->nshell(); Q++) {
        if (fcount + auxiliary_->shell(Q).nfunction() > max_rows) {
            shell_starts.push_back(Q);
            fcount = auxiliary_->shell(Q).nfunction();
        } else {
            fcount += auxiliary_->shell(Q).nfunction();
        }
    }
    shell_starts.push_back(auxiliary_->nshell());

    // > Task printing (Debug) < //

    //outfile->Printf("Auxiliary Composition:\n\n");
    //for (int Q = 0; Q < auxiliary_->nshell(); Q++) {
    //    outfile->Printf("%3d: %2d\n", Q, auxiliary_->shell(Q).nfunction());
    //}
    //outfile->Printf("\n");

    //outfile->Printf("Max Rows: %zu\n\n", max_rows);

    //outfile->Printf("Task Starts:\n\n");
    //for (int task = 0; task < shell_starts.size() - 1; task++) {
    //    outfile->Printf("%3d: %3d\n", task, shell_starts[task]);
    //}
    //outfile->Printf("\n");

    // => ERI Objects <= //

    boost::shared_ptr<IntegralFactory> factory(new IntegralFactory(auxiliary_, BasisSet::zero_ao_basis_set(), primary_, primary_));
    std::vector<boost::shared_ptr<TwoBodyAOInt> > eri;
    for (int thread = 0; thread < nthread; thread++) {
            eri.push_back(boost::shared_ptr<TwoBodyAOInt>(factory->eri()));
    }

    // => ERI Sieve <= //

    boost::shared_ptr<ERISieve> sieve(new ERISieve(primary_, 1e-10));
    const std::vector<std::pair<int,int> >& shell_pairs = sieve->shell_pairs();
    long int nshell_pairs = (long int) shell_pairs.size();

    // => Temporary Tensors <= //

    // > Three-index buffers < //
    boost::shared_ptr<Matrix> Amn(new Matrix("(A|mn)", max_rows, nso * (unsigned long int) nso));
    boost::shared_ptr<Matrix> Ami(new Matrix("(A|mi)", max_rows, nso * (unsigned long int) max1));
    boost::shared_ptr<Matrix> Aia(new Matrix("(A|ia)", naux, max12));
    double** Amnp = Amn->pointer();
    double** Amip = Ami->pointer();
    double** Aiap = Aia->pointer();
    int dims[2];
    int chunk[2];
    dims[0] = naux;
    dims[1] = nso * nso;
    chunk[0] = GA_Nnodes();
    chunk[1] = nso * nso;
    int Aia_ga = NGA_Create(C_DBL, 2, dims, (char *)"Aia_temp", chunk);
    GA_Q_PQ_ = GA_Duplicate(Aia_ga, (char *)"Q|PQ");
    GA_Print_distribution(Aia_ga);

    for (int block = 0; block < shell_starts.size() - 1; block++) {
        outfile->Printf("\n Pstart: %d Pstop: %d", shell_starts[block], shell_starts[block+1]);
    }
    // > C-matrix weirdness < //

    double** Cp = Ca_->pointer();
    int lda = nmo_;

    //// ==> Master Loop <== //

    int Aia_begin[2];
    int Aia_end[2];
    for (int block = 0; block < shell_starts.size() - 1; block++) {
    //    // > Block characteristics < //
        int Pstart = shell_starts[block];
        int Pstop  = shell_starts[block+1];
        int nPshell = Pstop - Pstart;
        int pstart = auxiliary_->shell(Pstart).function_index();
        int pstop = (Pstop == auxiliary_->nshell() ? auxiliary_->nbf() : auxiliary_->shell(Pstop).function_index());
        int rows = pstop - pstart;

        // > (Q|mn) ERIs < //

        ::memset((void*) Amnp[0], '\0', sizeof(double) * rows * nso * nso);

        #pragma omp parallel for schedule(dynamic) num_threads(nthread)
        for (long int PMN = 0L; PMN < nPshell * nshell_pairs; PMN++) {

            int thread = 0;
            #ifdef _OPENMP
                thread = omp_get_thread_num();
            #endif

            int P  = PMN / nshell_pairs + Pstart;
            int MN = PMN % nshell_pairs;
            std::pair<int,int> pair = shell_pairs[MN];
            int M = pair.first;
            int N = pair.second;

            eri[thread]->compute_shell(P,0,M,N);

            int nm = primary_->shell(M).nfunction();
            int nn = primary_->shell(N).nfunction();
            int np = auxiliary_->shell(P).nfunction();
            int om = primary_->shell(M).function_index();
            int on = primary_->shell(N).function_index();
            int op = auxiliary_->shell(P).function_index();

            const double* buffer = eri[thread]->buffer();

            for (int p = 0; p < np; p++) {
            for (int m = 0; m < nm; m++) {
            for (int n = 0; n < nn; n++) {
                Amnp[p + op - pstart][(m + om) * nso + (n + on)] =
                Amnp[p + op - pstart][(n + on) * nso + (m + om)] =
                (*buffer++);
            }}}
        }

        //for (int ind1 = 0; ind1 < tasks.size(); ind1++) {

        //    std::string space1 = pair_spaces_[tasks[ind1][0]].first;
        //    int start1 = spaces_[space1].first;
        //    int end1   = spaces_[space1].second;
        //    int n1      = end1 - start1;
        //    double* C1p = &Cp[0][start1];
        int start1 = 0;
        int end1   = nmo_;
        int n1     = end1 - start1;
        double* C1p = &Cp[0][start1];


        C_DGEMM('N','N',rows*nso,n1,nso,1.0,Amnp[0],nso,C1p,lda,0.0,Amip[0],n1);

        //for (int ind2 = 0; ind2 < tasks[ind1].size(); ind2++) {
        //std::string space2 = pair_spaces_[tasks[ind1][ind2]].second;
        //int start2 = spaces_[space2].first;
        //int end2   = spaces_[space2].second;
        //int n2      = end2 - start2;
        //double* C2p = &Cp[0][start2];
        int n2 = nmo_;
        double* C2p = &Cp[0][0];

        size_t n12 = n1 * (size_t) n2;
        size_t no1 = nso * (size_t) n1;

        //std::string name = tasks[ind1][ind2];
        bool transpose12 = false;

        if (transpose12) {
            #pragma omp parallel for num_threads(nthread)
            for (int Q = 0; Q < rows; Q++) {
                C_DGEMM('T','N',n2,n1,nso,1.0,C2p,lda,Amip[0] + Q*no1,n1,0.0,Aiap[0] + Q*n12,n1);
            }
        } else {
            #pragma omp parallel for num_threads(nthread)
            for (int Q = 0; Q < rows; Q++) {
                C_DGEMM('T','N',n1,n2,nso,1.0,Amip[0] + Q*no1,n1,C2p,lda,0.0,Aiap[0] + Q*n12,n2);
            }
        }

        //Amn->print();
        //Ami->print();
        //Aia->print();

        //boost::shared_ptr<Tensor> A = ints_[name + "_temp"];
        //FILE* fh = A->file_pointer();
        //fwrite(Aiap[0],sizeof(double),rows*n12,fh);
        int ld = nso * nso;
        
        NGA_Distribution(Aia_ga, GA_Nodeid(), Aia_begin, Aia_end);
        NGA_Put(Aia_ga, Aia_begin, Aia_end, Aiap[0], &(ld));
            //}
        //}
    }
    J_one_half();
    GA_Dgemm('T', 'N', naux, nso * nso, naux, 1.0, GA_J_onehalf_, Aia_ga, 0.0, GA_Q_PQ_);
    GA_Print(GA_Q_PQ_);
    
}
void ParallelDFMO::J_one_half()
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
}
}}
