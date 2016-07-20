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
namespace psi { namespace paralleldf {

ParallelDFJK::ParallelDFJK(boost::shared_ptr<BasisSet> primary, boost::shared_ptr<BasisSet> auxiliary) : DFJK(primary, auxiliary)
{
    common_init();
}
void ParallelDFJK::common_init()
{
    unit_ = PSIF_DFSCF_BJ;
    psio_ = PSIO::shared_object();
    outfile->Printf("\n ParallelDFJK");
}
void ParallelDFJK::preiterations()
{
    /// Compute the sieveing object on all nodes
    if (!sieve_)
    {
        sieve_ = boost::shared_ptr<ERISieve>(new ERISieve(primary_, cutoff_));    
    }
    ///Will compute J^{-(1/2)} (on all nodes.  Fuck it!)
    Timer Jm12_time;
    outfile->Printf("\n Jm12 takes %8.8f s.", Jm12_time.get());
}
void ParallelDFJK::compute_JK()
{
    /// This function will compute J in parallel for the number of Shell Pairs of auxiliary
    ///Gives the upper triangular size of the significant pairs after screening
    size_t ntri = sieve_->function_pairs().size();
    ULI three_memory = ((ULI)auxiliary_->nbf())*ntri;
    ULI two_memory = ((ULI)auxiliary_->nbf())*auxiliary_->nbf();
    //int nproc = MPI::COMM_WORLD.Get_size();

    //std::vector<std::pair<int, int> > my_aux_tasks = auxiliary_tasks(auxiliary_->nbf(), nproc);

    int nthread = 1;
    #ifdef _OPENMP
        nthread = df_ints_num_threads_;
    #endif
    int rank = 0;

    ///Parallel code only needs to store block of auxiliary indices
    Qmn_ = SharedMatrix(new Matrix("Qmn (Fitted Integrals)",
        auxiliary_->nbf(), ntri));
    double** Qmnp = Qmn_->pointer();

    //Get a TEI for each thread
    boost::shared_ptr<BasisSet> zero = BasisSet::zero_ao_basis_set();
    boost::shared_ptr<IntegralFactory> rifactory(new IntegralFactory(auxiliary_, zero, primary_, primary_));
    const double **buffer = new const double*[nthread];
    boost::shared_ptr<TwoBodyAOInt> *eri = new boost::shared_ptr<TwoBodyAOInt>[nthread];
    for (int Q = 0; Q<nthread; Q++) {
        eri[Q] = boost::shared_ptr<TwoBodyAOInt>(rifactory->eri());
        buffer[Q] = eri[Q]->buffer();
    }

    const std::vector<long int>& schwarz_shell_pairs = sieve_->shell_pairs_reverse();
    const std::vector<long int>& schwarz_fun_pairs = sieve_->function_pairs_reverse();

    int numP,Pshell,MU,NU,P,PHI,mu,nu,nummu,numnu,omu,onu;

    //timer_on("JK: (A|mn)");
    outfile->Printf("\n Nshell: %d", auxiliary_->nshell());

    for (Pshell=0; Pshell < auxiliary_->nshell(); ++Pshell) {
        outfile->Printf("\n Shell: %d NumP: %d\n\n", auxiliary_->nshell(), auxiliary_->shell(Pshell).nfunction());
        outfile->Printf("\n Fnct Index \n");
        for(int P = 0; P < auxiliary_->shell(Pshell).nfunction(); P++)
        {
            int function_index = auxiliary_->shell(Pshell).function_index() + P;
            outfile->Printf(" %d", function_index);
        }
    }
        
    //The integrals (A|mn)
    for (Pshell=0; Pshell < auxiliary_->nshell(); ++Pshell) {
    #pragma omp parallel for private (numP, Pshell, MU, NU, P, PHI, mu, nu, nummu, numnu, omu, onu, rank) schedule (dynamic) num_threads(nthread)
        for (MU=0; MU < primary_->nshell(); ++MU) {
            #ifdef _OPENMP
                rank = omp_get_thread_num();
            #endif
            nummu = primary_->shell(MU).nfunction();
            for (NU=0; NU <= MU; ++NU) {
                numnu = primary_->shell(NU).nfunction();
                if (schwarz_shell_pairs[MU*(MU+1)/2+NU] > -1) {
                    numP = auxiliary_->shell(Pshell).nfunction();
                    eri[rank]->compute_shell(Pshell, 0, MU, NU);
                    for (mu=0 ; mu < nummu; ++mu) {
                        omu = primary_->shell(MU).function_index() + mu;
                        for (nu=0; nu < numnu; ++nu) {
                            onu = primary_->shell(NU).function_index() + nu;
                            if(omu>=onu && schwarz_fun_pairs[omu*(omu+1)/2+onu] > -1) {
                                for (P=0; P < numP; ++P) {
                                    PHI = auxiliary_->shell(Pshell).function_index() + P;
                                    Qmnp[PHI][schwarz_fun_pairs[omu*(omu+1)/2+onu]] = buffer[rank][P*nummu*numnu + mu*numnu + nu];
                                }
                            } 
                        }
                    }
                }
            }
        }
    }
    boost::shared_ptr<Matrix> Jm12_m = Jm12();

    double** Jinvp = Jm12_m->pointer();

    ULI max_cols = (memory_-three_memory-two_memory) / auxiliary_->nbf();
    if (max_cols < 1)
        max_cols = 1;
    if (max_cols > ntri)
        max_cols = ntri;
    SharedMatrix temp(new Matrix("Qmn buffer", auxiliary_->nbf(), max_cols));
    double** tempp = temp->pointer();

    size_t nblocks = ntri / max_cols;
    if ((ULI)nblocks*max_cols != ntri) nblocks++;

    size_t ncol = 0;
    size_t col = 0;

    timer_on("JK: (Q|mn)");

    for (size_t block = 0; block < nblocks; block++) {

        ncol = max_cols;
        if (col + ncol > ntri)
            ncol = ntri - col;

        C_DGEMM('N','N',auxiliary_->nbf(), ncol, auxiliary_->nbf(), 1.0,
            Jinvp[0], auxiliary_->nbf(), &Qmnp[0][col], ntri, 0.0,
            tempp[0], max_cols);

        for (int Q = 0; Q < auxiliary_->nbf(); Q++) {
            C_DCOPY(ncol, tempp[Q], 1, &Qmnp[Q][col], 1);
        }

        col += ncol;
    }
    max_nocc_ = max_nocc();
    max_rows_ = max_rows();
    outfile->Printf("\n max_rows: %d max_nocc: %d", max_rows_, max_nocc_);
    initialize_temps();

    timer_off("JK: (Q|mn)");
    for (int Q = 0 ; Q < auxiliary_->nbf(); Q += max_rows_) {
    int naux = (auxiliary_->nbf() - Q <= max_rows_ ? auxiliary_->nbf() - Q : max_rows_);
        if (do_J_) {
            timer_on("JK: J");
            block_J(&Qmn_->pointer()[Q],naux);
            timer_off("JK: J");
        }
        if(do_K_) {
            timer_on("JK: K");
            block_K(&Qmn_->pointer()[Q], naux);
            timer_off("JK: K");
        }
    }
}
boost::shared_ptr<Matrix> ParallelDFJK::Jm12()
{
    // Everybody likes them some inverse square root metric, eh?

    int nthread = 1;
    #ifdef _OPENMP
        nthread = omp_get_max_threads();
    #endif

    int naux = auxiliary_->nbf();

    boost::shared_ptr<Matrix> J(new Matrix("J", naux, naux));
    double** Jp = J->pointer();

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

    return J;
}
void ParallelDFJK::block_J(double** Qmnp, int naux)
{
    const std::vector<std::pair<int, int> >& function_pairs = sieve_->function_pairs();
    unsigned long int num_nm = function_pairs.size();

    for (size_t N = 0; N < J_ao_.size(); N++) {

        double** Dp   = D_ao_[N]->pointer();
        double** Jp   = J_ao_[N]->pointer();
        double*  J2p  = J_temp_->pointer();
        double*  D2p  = D_temp_->pointer();
        double*  dp   = d_temp_->pointer();
        for (unsigned long int mn = 0; mn < num_nm; ++mn) {
            int m = function_pairs[mn].first;
            int n = function_pairs[mn].second;
            D2p[mn] = (m == n ? Dp[m][n] : Dp[m][n] + Dp[n][m]);
        }

        timer_on("JK: J1");
        C_DGEMV('N',naux,num_nm,1.0,Qmnp[0],num_nm,D2p,1,0.0,dp,1);
        timer_off("JK: J1");

        timer_on("JK: J2");
        C_DGEMV('T',naux,num_nm,1.0,Qmnp[0],num_nm,dp,1,0.0,J2p,1);
        timer_off("JK: J2");
        for (unsigned long int mn = 0; mn < num_nm; ++mn) {
            int m = function_pairs[mn].first;
            int n = function_pairs[mn].second;
            Jp[m][n] += J2p[mn];
            Jp[n][m] += (m == n ? 0.0 : J2p[mn]);
        }
    }
}

}}
