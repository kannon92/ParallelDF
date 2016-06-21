#include <libfock/jk.h>
#include <psifiles.h>
#include "paralleldfjk.h"
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
}}
