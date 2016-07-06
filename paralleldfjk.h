#include <libmints/mints.h>
#include <libfock/jk.h>
#include <psi4-dec.h>
namespace psi { namespace paralleldf {

class ParallelDFJK : public DFJK {
    public:
        ParallelDFJK(boost::shared_ptr<BasisSet> primary, boost::shared_ptr<BasisSet> auxiliary);
        //virtual void compute_JK();
        void common_init();   
    protected:
        void preiterations();
        boost::shared_ptr<Matrix> Jm12();
        /// Do a direct J(and/or) K build
        void compute_JK();
        void block_J(double** Qmnp, int naux);
        //void block_K(double** Qmnp, int naux);

};
}}
