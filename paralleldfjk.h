#include <libmints/mints.h>
#include <libfock/jk.h>
#include <psi4-dec.h>
namespace psi { namespace paralleldf {

class ParallelDFJK : public JK {
    public:
        ParallelDFJK(boost::shared_ptr<BasisSet> primary, boost::shared_ptr<BasisSet> auxiliary);
        void common_init();   
        void set_condition(double condition) { condition_ = condition; }
        void set_df_ints_num_threads(int val) { df_ints_num_threads_ = val; }
    protected:
        boost::shared_ptr<BasisSet> auxiliary_; 
        int df_ints_num_threads_;
        double condition_;
        boost::shared_ptr<ERISieve> sieve_;
        ///J^{(-1/2)} fitting metric
        int J_12_GA_  = 0;
        ///(Q|UV) tensor created in initialization of JK
        int Q_UV_GA_  = 0;
        ///A vector of J and K global array matrices
        std::vector<int> J_UV_GA_;
        std::vector<int> K_UV_GA_;
        ///Temporary information
        ///J_{uv} = B_{uv}^{Q}D_{pq}B_{pq}^{Q}
        ///v^{Q} = D_{pq} * B_{pq}^{Q}
        int J_V_GA_ = 0;
        ///K_{uv} = B_{uv}^{Q} D_{vq} B_{pq}^{Q}
        ///K_{uv} = C_{v i} B_{uv}^{Q} * C_{p i} B_{qp}^{Q}
        int K_CB1_ = 0;
        int K_CB2_ = 0;
        bool debug_ = false;


        void preiterations();
        virtual void postiterations();
        virtual void print_header() const;
        void compute_qmn();
        void J_one_half();
        /// Do a direct J(and/or) K build
        void compute_JK();
        void compute_J();
        void compute_K();
        void create_temp_ga();
        void block_J(double** Qmnp, int naux);
        //void block_K(double** Qmnp, int naux);
        virtual bool C1() const { return true; }

};
}}
