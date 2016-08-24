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
        int J_12_GA_ = 0;
        int Q_UV_GA_  = 0;

        void preiterations();
        virtual void postiterations();
        virtual void print_header() const;
        void compute_qmn();
        void J_one_half();
        /// Do a direct J(and/or) K build
        void compute_JK();
        void block_J(double** Qmnp, int naux);
        //void block_K(double** Qmnp, int naux);
        virtual bool C1() const { return true; }

};
}}
