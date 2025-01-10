#include "operators/matmul.h"
#include "core/common.h"
#include <cstdio>
#include <utility>

namespace infini {

MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                     bool transB)
    : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}), transA(transA),
      transB(transB) {
    IT_ASSERT(checkValid(graph));
}

string MatmulObj::toString() const {
    std::ostringstream os;
    os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
       << ",A=" << inputs[0]->getGuid() << ",B=" << inputs[1]->getGuid()
       << ",C=" << outputs[0]->getGuid() << ",mnk=[" << m << "," << n << ","
       << k << "])";
    return os.str();
}

optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs) {
    auto sa = inputs[0]->getDims(), sb = inputs[1]->getDims();
    auto rank = sa.size(), _0 = rank - 1, _1 = rank - 2;
    // m n k
    m = transA ? sa[_0] : sa[_1];
    n = transA ? sa[_1] : sa[_0];
    k = transB ? sb[_1] : sb[_0];
    Shape shape(sa.size());
    // GEMM dimensions: (m,n) x (n,k) = (m,k)
    shape[_0] = k, shape[_1] = m;
    // broadcasting
    for (size_t i = 0; i < _1; i++) {
        shape[i] = std::max(sa[i], sb[i]);
    }
    return optional{vector<Shape>{shape}};
}

} // namespace infini
