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
        /// Driver function to compute J and K
        void compute_JK();
        /// Do a Direct DF-J build (generate integrals over batches of P)
        void compute_J();
        //void compute_K();

};
}}
