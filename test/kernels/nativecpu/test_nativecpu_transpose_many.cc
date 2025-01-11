#include "core/graph.h"
#include "core/runtime.h"
#include "operators/transpose.h"

#include "test.h"

namespace infini {

/*

 import numpy as np
 n = 1 * 2 * 3 * 4
 x = np.arange(n).reshape((1,2,3,4))
 perm = [1,2,3,0]
 x = np.transpose(x, perm); print(x.reshape((n,)))
 x = np.transpose(x, perm); print(x.reshape((n,)))
 x = np.transpose(x, perm); print(x.reshape((n,)))
 x = np.transpose(x, perm); print(x.reshape((n,)))

 */
TEST(TransposeMany, NativeCpu) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    Shape shape{1, 2, 3, 4}, permute{1, 2, 3, 0};
    // t1 (op-x) t2 (op-y) t3 (op-z) t4 (op-w) t5
    auto x0 = g->addTensor(shape, DataType::Float32);
    auto op0 = g->addOp<TransposeObj>(x0, nullptr, permute);
    auto x1 = op0->getOutput(0);
    auto op1 = g->addOp<TransposeObj>(x1, nullptr, permute);
    auto x2 = op1->getOutput(0);
    auto op2 = g->addOp<TransposeObj>(x2, nullptr, permute);
    auto x3 = op2->getOutput(0);
    auto op3 = g->addOp<TransposeObj>(x3, nullptr, permute);
    auto x4 = op3->getOutput(0);

    g->dataMalloc();
    x0->setData(IncrementalGenerator());

    runtime->run(g);
    vector<float> y4{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};

    EXPECT_TRUE(x4->equalData(y4));
}

} // namespace infini
