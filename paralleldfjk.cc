#include <libfock/jk.h>
#include <libmints/mints.h>
#include <libmints/sieve.h>
#include <psifiles.h>
#include "paralleldfjk.h"
#include <psi4-dec.h>
#include <lib3index/3index.h>
#include <libqt/qt.h>
#include <omp.h>
#include <ga.h>
#include <macdecls.h>
#include <mpi.h>
namespace psi { namespace paralleldf {

ParallelDFJK::ParallelDFJK(boost::shared_ptr<BasisSet> primary, boost::shared_ptr<BasisSet> auxiliary) : JK(primary), auxiliary_(auxiliary)
{
    common_init();
}
void ParallelDFJK::common_init()
{
    outfile->Printf("\n ParallelDFJK");
    memory_ = Process::environment.get_memory();
}
void ParallelDFJK::preiterations()
{
    /// Compute the sieveing object on all nodes
    if (!sieve_)
    {
        sieve_ = boost::shared_ptr<ERISieve>(new ERISieve(primary_, cutoff_));    
    }
    //create_ga_arrays();
    ///Will compute J^{-(1/2)} (on all nodes.  Fuck it!)
    Timer Jm12_time;
    outfile->Printf("\n Jm12 takes %8.8f s.", Jm12_time.get());
    compute_qmn();
}
void ParallelDFJK::compute_JK()
{
    if(do_J_)
    {
        Timer compute_local_j;
        compute_J();
        printf("\n P%d computing J", GA_Nodeid(), compute_local_j.get());
    }
    if(do_K_)
    {
        Timer compute_local_k;
        compute_K();
        printf("\n P%d computing K", GA_Nodeid(), compute_local_k.get());
    }
}
void ParallelDFJK::J_one_half()
{
    // Everybody likes them some inverse square root metric, eh?

    int nthread = 1;
    #ifdef _OPENMP
        nthread = omp_get_max_threads();
    #endif

    int naux = auxiliary_->nbf();

    boost::shared_ptr<Matrix> J(new Matrix("J", naux, naux));
    double** Jp = J->pointer();

    int dims[2];
    int chunk[2];
    dims[0] = naux;
    dims[1] = naux;
    chunk[0] = -1;
    chunk[1] = naux;
    J_12_GA_ = NGA_Create(C_DBL, 2, dims, (char *)"J_1/2", chunk);
    if(not J_12_GA_)
        throw PSIEXCEPTION("Failure in creating J_^(-1/2) in GA");

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
    outfile->Printf("\n JRMS: %8.8f", J->rms());
    if(GA_Nodeid() == 0)
    {
        for(int me = 0; me < GA_Nnodes(); me++)
        {
            int begin_offset[2];
            int end_offset[2];
            NGA_Distribution(J_12_GA_, me, begin_offset, end_offset);
            int offset = begin_offset[0];
            NGA_Put(J_12_GA_, begin_offset, end_offset, J->pointer()[offset], &naux);
        }
    }
}
void ParallelDFJK::block_J(double** Qmnp, int naux)
{
    //const std::vector<std::pair<int, int> >& function_pairs = sieve_->function_pairs();
    //unsigned long int num_nm = function_pairs.size();

    //for (size_t N = 0; N < J_ao_.size(); N++) {

    //    //double** Dp   = D_ao_[N]->pointer();
    //    //double** Jp   = J_ao_[N]->pointer();
    //    //double*  J2p  = J_temp_->pointer();
    //    //double*  D2p  = D_temp_->pointer();
    //    //double*  dp   = d_temp_->pointer();
    //    //for (unsigned long int mn = 0; mn < num_nm; ++mn) {
    //    //    int m = function_pairs[mn].first;
    //    //    int n = function_pairs[mn].second;
    //    //    D2p[mn] = (m == n ? Dp[m][n] : Dp[m][n] + Dp[n][m]);
    //    //}

    //    timer_on("JK: J1");
    //    C_DGEMV('N',naux,num_nm,1.0,Qmnp[0],num_nm,D2p,1,0.0,dp,1);
    //    timer_off("JK: J1");

    //    timer_on("JK: J2");
    //    C_DGEMV('T',naux,num_nm,1.0,Qmnp[0],num_nm,dp,1,0.0,J2p,1);
    //    timer_off("JK: J2");
    //    for (unsigned long int mn = 0; mn < num_nm; ++mn) {
    //        int m = function_pairs[mn].first;
    //        int n = function_pairs[mn].second;
    //        Jp[m][n] += J2p[mn];
    //        Jp[n][m] += (m == n ? 0.0 : J2p[mn]);
    //    }
    //}
}
void ParallelDFJK::compute_qmn()
{
// > Sizing < //

    int nso = primary_->nbf();
    int naux = auxiliary_->nbf();

    // > Threading < //

    int nthread = 1;
    #ifdef _OPENMP
        nthread = omp_get_max_threads();
    #endif

    // > Row requirements < //

    unsigned long int per_row = 0L;
    // (Q|mn)
    per_row += nso * (unsigned long int) nso;

    // > Maximum number of rows < //

    unsigned long int max_rows = (memory_ / per_row);
    //max_rows = 3L * auxiliary_->max_function_per_shell(); // Debug
    if (max_rows < auxiliary_->max_function_per_shell()) {
        throw PSIEXCEPTION("Out of memory in DFERI.");
    }
    max_rows = (max_rows > auxiliary_->nbf() ? auxiliary_->nbf() : max_rows);
    int shell_per_process = 0;
    int shell_start = -1;
    int shell_end = -1;
    /// MPI Environment 
    int my_rank = GA_Nodeid();
    int num_proc = GA_Nnodes();

    if(auxiliary_->nbf() == max_rows)
    {
       shell_per_process = auxiliary_->nshell() / num_proc;
    }
    else {
        throw PSIEXCEPTION("Have not implemented memory bound df integrals");
    }
    ///Have first proc be from 0 to shell_per_process
    ///Last proc is shell_per_process * my_rank to naux
    if(my_rank != (num_proc - 1))
    {
        shell_start = shell_per_process * my_rank;
        shell_end   = shell_per_process * (my_rank + 1);
    }
    else
    {
        shell_start = shell_per_process * my_rank;
        shell_end = (auxiliary_->nshell() % num_proc == 0 ? shell_per_process * (my_rank + 1) : auxiliary_->nshell());
    }

    int function_start = auxiliary_->shell(shell_start).function_index();
    int function_end = (shell_end == auxiliary_->nshell() ? auxiliary_->nbf() : auxiliary_->shell(shell_end).function_index());
    int dims[2];
    int chunk[2];
    dims[0] = naux;
    dims[1] = nso * nso;
    chunk[0] = GA_Nnodes();
    chunk[1] = 1;
    int map[GA_Nnodes() + 1];
    for(int iproc = 0; iproc < GA_Nnodes(); iproc++)
    {
        int shell_start = 0;
        int shell_end = 0;
        if(iproc != (num_proc - 1))
        {
            shell_start = shell_per_process * iproc;
            shell_end   = shell_per_process * (iproc + 1);
        }
        else
        {
            shell_start = shell_per_process * iproc;
            shell_end = (auxiliary_->nshell() % num_proc == 0 ? shell_per_process * (iproc + 1) : auxiliary_->nshell());
        }
        int function_start = auxiliary_->shell(shell_start).function_index();
        int function_end = (shell_end == auxiliary_->nshell() ? auxiliary_->nbf() : auxiliary_->shell(shell_end).function_index());
        map[iproc] = function_start;
        outfile->Printf("\n  P%d shell_start: %d shell_end: %d function_start: %d function_end: %d", iproc, shell_start, shell_end, function_start, function_end);
    }
    map[GA_Nnodes()] = 0;
    int A_UV_GA = NGA_Create_irreg(C_DBL, 2, dims, (char *)"Auv_temp", chunk, map);
    if(not A_UV_GA)
    {
        throw PSIEXCEPTION("GA failed on creating Aia_ga");
    }
    Q_UV_GA_ = GA_Duplicate(A_UV_GA, (char *)"Q|PQ");
    if(not Q_UV_GA_)
    {
        throw PSIEXCEPTION("GA failed on creating GA_Q_PQ");
    }

    // => ERI Objects <= //

    boost::shared_ptr<IntegralFactory> factory(new IntegralFactory(auxiliary_, BasisSet::zero_ao_basis_set(), primary_, primary_));
    std::vector<boost::shared_ptr<TwoBodyAOInt> > eri;
    for (int thread = 0; thread < nthread; thread++) {
            eri.push_back(boost::shared_ptr<TwoBodyAOInt>(factory->eri()));
    }

    // => ERI Sieve <= //

    //boost::shared_ptr<ERISieve> sieve(new ERISieve(primary_, 1e-10));
    const std::vector<std::pair<int,int> >& shell_pairs = sieve_->shell_pairs();
    long int nshell_pairs = (long int) shell_pairs.size();

    // => Temporary Tensors <= //

    // > Three-index buffers < //
    boost::shared_ptr<Matrix> Auv(new Matrix("(Q|mn)", max_rows, nso * (unsigned long int) nso));
    double** Auvp = Auv->pointer();

    //// ==> Master Loop <== //

    int Auv_begin[2];
    int Auv_end[2];
    /// SIMD 
    ///shell_start represents the start of shells for this processor
    ///shell_end represents the end of shells for this processor
    ///NOTE:  This code will have terrible load balance (shells do not correspond to equal number of functions
    Timer compute_Auv;
    {
        int Pstart = shell_start;
        int Pstop  = shell_end;
        int nPshell = Pstop - Pstart;
        int pstart = auxiliary_->shell(Pstart).function_index();
        int pstop = (Pstop == auxiliary_->nshell() ? auxiliary_->nbf() : auxiliary_->shell(Pstop).function_index());
        int rows = pstop - pstart;

        // > (Q|mn) ERIs < //

        ::memset((void*) Auvp[0], '\0', sizeof(double) * rows * nso * nso);

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
                Auvp[p + op - pstart][(m + om) * nso + (n + on)] =
                Auvp[p + op - pstart][(n + on) * nso + (m + om)] =
                (*buffer++);
            }}}
        }

        int ld = nso * nso;
        NGA_Distribution(A_UV_GA, GA_Nodeid(), Auv_begin, Auv_end);
        NGA_Put(A_UV_GA, Auv_begin, Auv_end, Auvp[0], &(ld));
        outfile->Printf(" AUVRMS: %8.8f", Auv->rms());
    }
    printf("\n  P%d Auv took %8.6f s.", GA_Nodeid(), compute_Auv.get());

    Timer J_one_half_time;
    J_one_half();
    printf("\n  P%d J^({-1/2}} took %8.6f s.", GA_Nodeid(), J_one_half_time.get());

    Timer GA_DGEMM;
    GA_Dgemm('T', 'N', naux, nso * nso, naux, 1.0, J_12_GA_, A_UV_GA, 0.0, Q_UV_GA_);
    printf("\n  P%d DGEMM took %8.6f s.", GA_Nodeid(), GA_DGEMM.get());
    GA_Destroy(A_UV_GA);
    GA_Destroy(J_12_GA_);


}
void ParallelDFJK::compute_J()
{
    ///Some basic information (naux -> auxiliary basis set size
    int naux = auxiliary_->nbf();
    ///(nso -> number of basis functions
    int nso = D_ao_[0]->rowspi()[0];
    unsigned long int num_nm = nso * nso;

    SharedVector J_temp(new Vector("Jtemp", num_nm));
    SharedVector D_temp(new Vector("Dtemp", num_nm));
    ///Vector that will be used for AllReduce
    SharedVector Global_J_temp(new Vector("Jtemp", num_nm));
    double* D_tempp = D_temp->pointer();

    ///GA specific information (offset[0] will tell what is distributed on each processor
    int begin_offset[2];
    int end_offset[2];
    int index = 0;
    ///Local q_uv for get and J_V
    std::vector<double> q_uv_temp;
    std::vector<double> J_V;
    Timer Compute_J_all;

    ///Since Q_UV_GA is distributed via NAUX index,
    ///need to get locality information (where data is located)
    ///Since Q never changes via density, no need to be in loop
    NGA_Distribution(Q_UV_GA_,GA_Nodeid(), begin_offset, end_offset);
    size_t q_uv_size = (end_offset[0] - begin_offset[0] + 1) * nso * nso;
    q_uv_temp.resize(q_uv_size);
    J_V.resize(end_offset[0] - begin_offset[0] + 1);
    int stride = nso * nso;
    NGA_Get(Q_UV_GA_, begin_offset, end_offset, &q_uv_temp[0], &stride);
    ///Start a loop over the densities
    for(size_t N = 0; N < J_ao_.size(); N++)
    {
        Timer Compute_J_one;
        ///This loop is parallelized over MPI
        ///Q_UV_GA is distributed with auxiliary_index
        double** Dp = D_ao_[N]->pointer();
        double** Jp = J_ao_[N]->pointer();
        index = 0;
        C_DCOPY(nso * nso, Dp[0], 1,D_temp->pointer(), 1);


        ///J_V = B^Q_{pq} D_{pq}
        Timer v_BD;
        size_t local_naux = end_offset[0] - begin_offset[0] + 1;
        C_DGEMV('N', local_naux, num_nm, 1.0, &q_uv_temp[0], num_nm, D_tempp, 1, 0.0, &J_V[0], 1);
        double J_V_norm = 0.0;
        Timer J_BJ;
        ///J_{uv} = B^{Q}_{uv} J_V^{Q}
        C_DGEMV('T', local_naux, num_nm, 1.0, &q_uv_temp[0], num_nm, &J_V[0], 1, 0.0, J_temp->pointer(), 1);
        index = 0;
        ///Since every processor has a copy of J_temp, sum all the parts and send to every process
        MPI_Allreduce(J_temp->pointer(), Global_J_temp->pointer(), num_nm, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        C_DCOPY(nso * nso, Global_J_temp->pointer(), 1,Jp[0], 1);
        printf("\n P%d Compute_J for %d density takes %8.6f s", GA_Nodeid(), Compute_J_one.get(), N);
    }
    printf("\nP%d Compute_J takes %8.6f s for %d densities", GA_Nodeid(), Compute_J_all.get(), J_ao_.size());
}
void ParallelDFJK::compute_K()
{
    /// K_{uv} = D_{pq} B^{Q}_{up} * B^{Q}_{vq}
    /// Step 1:  Use def of D = \sum_{i} C_{pi}C_{qi}
    /// Step 2:  Perform a one index transform  (N^4)
    ///          B^{Q}_{ui} = C_{pi} B^{Q}_{up}
    ///          B^{Q}_{vi} = C_{qi} B^{Q}_{vq}
    /// Step 3:  Compute K_{uv} = \sum_{Q} \sum_{i} B^{Q}_{ui} B^{Q}_{vi}

    /// The first iteration of this job will assume that B tensor is distributed 
    /// via the Q index.
    /// This means that all of these steps will be performed locally for every process
    /// Only communciation required will be an Allreduce once the K matrix is formed
    /// GA is used, but we will only perform local MM (so we use data on each processor only)

    /// Can have multiple exchange matrices
    /// GA Specific information
    int begin_offset[2];
    int end_offset[2];
    int index = 0;
    ///Local q_uv for get and J_V
    std::vector<double> q_uv_temp;

    ///Since Q_UV_GA is distributed via NAUX index,
    ///need to get locality information (where data is located)
    ///Since Q never changes via density, no need to be in loop
    Timer Get_K_GA;
    NGA_Distribution(Q_UV_GA_,GA_Nodeid(), begin_offset, end_offset);
    int stride = end_offset[1] - begin_offset[1] + 1;
    size_t q_uv_size = (end_offset[0] - begin_offset[0] + 1) * stride;
    size_t local_naux = (end_offset[0] - begin_offset[0] + 1);
    q_uv_temp.resize(q_uv_size);
    NGA_Get(Q_UV_GA_, begin_offset, end_offset, &q_uv_temp[0], &stride);
    printf("\n P%d GET_K takes %8.6f", GA_Nodeid(), Get_K_GA.get());

 

    size_t K_size = K_ao_.size();
    Timer Compute_K_all;
    for(size_t N = 0; N < K_size; N++)
    {
        int nbf = C_left_ao_[N]->rowspi()[0];
        int nocc = C_left_ao_[N]->colspi()[0];
        double** Clp = C_left_ao_[N]->pointer();
        double** Crp = C_right_ao_[N]->pointer();
        double** Kp  = K_ao_[N]->pointer();
        SharedMatrix BQ_ui(new Matrix("B^Q_{ui}", local_naux * nbf, nocc));
        SharedMatrix BQ_vi(new Matrix("B^Q_{vi}", local_naux * nbf, nocc));
        SharedMatrix Bm_Qi(new Matrix("B^m_{Qi}", nbf, local_naux * nocc));
        SharedMatrix Bn_Qi(new Matrix("B^n_{Qi}", nbf, local_naux * nocc));

        if(not nocc) continue; ///If no occupied orbitals skip exchange

        if(N == 0 or C_left_[N].get() != C_left_[N-1].get())
        {
            Timer B_C_halftrans;

            C_DGEMM('N', 'N', local_naux * nbf, nocc, nbf, 1.0, &q_uv_temp[0], nbf, Clp[0], nocc, 0.0, BQ_ui->pointer()[0], nocc);
            printf("\n P%d B * C takes %8.4f", GA_Nodeid(), B_C_halftrans.get());

            Timer swap_index;
            #pragma omp parallel for
            for(int n = 0; n < local_naux; n++)
                for(int m = 0; m < nbf; m++)
                    C_DCOPY(nocc, &BQ_ui->pointer()[0][n * nbf * nocc + m * nocc], 1, &Bm_Qi->pointer()[0][m * local_naux * nocc + n * nocc], 1);

            printf("\n P%d Bm_Qi to BQ_ui takes %8.4f s", GA_Nodeid(), swap_index.get());

        if(lr_symmetric_)
            Bn_Qi = Bm_Qi;
             
        }
        if(!lr_symmetric_ && (N == 0 || C_right_[N].get() != C_right_[N-1].get())) {

            if(C_right_[N].get() == C_left_[N].get()) 
            {
                //::memcpy((void*) Bm_Qi->pointer()[0], (void*) Bn_Qi->pointer()[0], sizeof(double) * local_naux * nocc * nbf);
                Bn_Qi = Bm_Qi;
            }
            else {
            C_DGEMM('N', 'N',local_naux * nbf, nocc, nbf, 1.0, &q_uv_temp[0], nbf, Crp[0], nocc, 0.0, BQ_ui->pointer()[0], nocc);

            #pragma omp parallel for
            for(int n = 0; n < local_naux; n++)
                for(int m = 0; m < nbf; m++)
                    C_DCOPY(nocc, &BQ_ui->pointer()[0][n * nbf * nocc + m * nocc], 1, &Bn_Qi->pointer()[0][m * local_naux * nocc + n * nocc], 1);

            }
             
        }
        SharedMatrix local_K(new Matrix("K", nbf, nbf));
        Timer Final_K;
        C_DGEMM('N','T', nbf, nbf, local_naux * nocc, 1.0, Bm_Qi->pointer()[0], local_naux * nocc, Bn_Qi->pointer()[0], local_naux * nocc, 0.0, local_K->pointer()[0], nbf);
        printf("\n P%d Final_K takes %8.4f s", GA_Nodeid(), Final_K.get());
        
        Timer ALLREDUCE;
        MPI_Allreduce(local_K->pointer()[0],Kp[0], nbf * nbf, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        printf("\n P%d ALLREDUCE takes %8.4f s", GA_Nodeid(), ALLREDUCE.get());
        

    }
    printf("\n P%d for Compute K with %d densities %8.4f s", GA_Nodeid(), D_ao_.size(), Compute_K_all.get());
}
void ParallelDFJK::postiterations()
{
    GA_Destroy(Q_UV_GA_);
}
void ParallelDFJK::print_header() const
{
    outfile->Printf("\n Computing DFJK using %d Processes and %d threads", GA_Nnodes(), omp_get_max_threads());
}
//void ParallelDFJK::create_temp()
//{
//}

}}
