#include <libmints/basisset.h>
#include <psi4-dec.h>
namespace psi { namespace paralleldf {

class ParallelDFMO {
    public:
        ParallelDFMO(boost::shared_ptr<BasisSet> primary, boost::shared_ptr<BasisSet> auxiliary);
        void set_C(boost::shared_ptr<Matrix> C);
        void compute_integrals();
    protected:
        SharedMatrix C_ao_;
        /// (A | Q)^{-1/2}
        int J_one_half();
        ///Compute (A|mn) integrals (distribute via mn indices)
        void compute_A_mn();
        /// (A_pq) = (A | mu nu) C_{mu p} C_{nu q}
        void compute_A_pq();
        /// (A | pq) (A | Q)^{-1/2}
        void compute_Q_pq();

        boost::shared_ptr<BasisSet> primary_;
        boost::shared_ptr<BasisSet> auxiliary_;

        /// Distributed DF (Q | pq) integrals
        int GA_Q_PQ_;
        /// GA for J^{-1/2}
        int GA_J_onehalf_;
};
}}
