#include <libfock/jk.h>
#include <libmints/mints.h>
#include <libmints/sieve.h>
#include <psifiles.h>
#include "paralleldfjk.h"
#include <psi4-dec.h>
#include <libqt/qt.h>
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
    boost::shared_ptr<Matrix> Jm12_m = Jm12(auxiliary_, 1e-12);
    outfile->Printf("\n Jm12 takes %8.8f s.", Jm12_time.get());
}
void ParallelDFJK::compute_JK()
{
    if(do_J_)
    {
        compute_J();
    }
    //if(do_K_)
    //{
    //    compute_K();
    //}
}
void ParallelDFJK::compute_J()
{
    /// This function will compute J in parallel for the number of Shell Pairs of auxiliary
    ///Gives the upper triangular size of the significant pairs after screening
    size_t ntri = sieve_->function_pairs().size();
    ULI three_memory = ((ULI)auxiliary_->nbf())*ntri;
    ULI two_memory = ((ULI)auxiliary_->nbf())*auxiliary_->nbf();

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

    //The integrals (A|mn)
    #pragma omp parallel for private (numP, Pshell, MU, NU, P, PHI, mu, nu, nummu, numnu, omu, onu, rank) schedule (dynamic) num_threads(nthread)
    for (Pshell=0; Pshell < auxiliary_->nshell(); ++Pshell) {
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
              
}

}}
