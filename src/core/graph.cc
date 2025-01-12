#include "core/graph.h"
#include "core/blob.h"
#include "core/common.h"
#include "core/object.h"
#include "core/op_type.h"
#include "core/ref.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <cstdio>
#include <memory>
#include <unordered_map>
#include <utility>

namespace infini {

void GraphObj::addOperatorAndConnect(const Operator &op) {
    sorted = false;
    ops.push_back(op);
    for (auto &input : op->getInputs()) {
        if (input) {
            input->addTarget(op);
            if (auto pred = input->getSource()) {
                pred->addSuccessors(op);
                op->addPredecessors(pred);
            }
        }
    }
    for (auto &output : op->getOutputs()) {
        if (output) {
            output->setSource(op);
            for (auto &succ : output->getTargets()) {
                succ->addPredecessors(op);
                op->addSuccessors(succ);
            }
        }
    }
}

string GraphObj::toString() const {
    std::ostringstream oss;
    oss << "Graph Tensors:\n";
    for (const auto &tensor : tensors)
        oss << tensor << "\n";

    oss << "Graph operators:\n";
    for (const auto &op : ops) {
        vector<UidBaseType> preds, succs;
        for (auto &o : op->getPredecessors())
            preds.emplace_back(o->getGuid());
        for (auto &o : op->getSuccessors())
            succs.emplace_back(o->getGuid());
        oss << "OP " << op->getGuid();
        oss << ", pred " << vecToString(preds);
        oss << ", succ " << vecToString(succs);
        oss << ", " << op << "\n";
    }
    return oss.str();
}

bool GraphObj::topo_sort() {
    if (this->sorted) {
        return true;
    }
    std::vector<Operator> sorted;
    std::unordered_set<OperatorObj *> flags;
    sorted.reserve(ops.size());
    flags.reserve(ops.size());
    while (sorted.size() < ops.size()) {
        // Any node is move to sorted in this loop.
        auto modified = false;
        for (auto const &op : ops) {
            if (auto const &inputs = op->getInputs();
                flags.find(op.get()) == flags.end() &&
                std::all_of(inputs.begin(), inputs.end(),
                            [&flags](auto const &input) {
                                auto ptr = input->getSource().get();
                                return !ptr || flags.find(ptr) != flags.end();
                            })) {
                modified = true;
                sorted.emplace_back(op);
                flags.insert(op.get());
            }
        }
        if (!modified) {
            return false;
        }
    }
    this->ops = std::move(sorted);
    return this->sorted = true;
}

static vector<int> permCompose(const vector<int> p, const vector<int> q) {
    auto n = p.size();
    vector<int> r(n);
    for (auto i = 0u; i < n; i++) {
        r[i] = q[p[i]];
    }
    return r;
}
static bool isMatTrans(const vector<int> p) {
    int n = p.size();
    if (n < 2)
        return false;
    for (auto i = 0; i + 2 < n; i++)
        if (p[i] != i)
            return false;
    return p[n - 1] == n - 2 && p[n - 2] == n - 1;
}
static bool isIdentity(const vector<int> p) {
    int n = p.size();
    for (auto i = 0; i < n; i++)
        if (p[i] != i)
            return false;
    return true;
}

void GraphObj::optimize() {
    // 1. 去除冗余的算子
    // eg two consecutive cancellable transpose -> none
    // 2. 合并算子 eg transpose + matmul transA transB
    bool modified = true;
    do {
        modified = false;
        // rule 1: fuse two transposes
        for (auto op : ops) {
            if (op->getOpType() == OpType::Transpose) {
                const auto &succVec = op->getSuccessors();
                const bool trAll =
                    std::all_of(succVec.begin(), succVec.end(), [](auto x) {
                        return x->getOpType() == OpType::Transpose;
                    });
                auto p = dynamic_cast<TransposeObj *>(op.get())->getPermute();
                if (!succVec.empty() && trAll) {
                    modified = true;

                    for (auto &suc : succVec) {
                        auto in = op->getInputs(0), out = suc->getOutput();
                        auto q = dynamic_cast<TransposeObj *>(suc.get())
                                     ->getPermute();
                        // add a new fused operator
                        auto fusedOp = addOpWithOutputs<TransposeObj>(
                            in, out, permCompose(p, q));
                        // detach the suc operator
                        removeOperator(suc);
                        // NOTE: correct source
                        out->setSource(fusedOp);
                    }
                    // the output buffer will nolonger be used
                    removeTensor(op->getOutput());
                    // finally remove this
                    removeOperator(op);

                    print();
                    break;
                }
            }
        }
        if (modified)
            continue;

        // rule 2: fuse transposes with GEMM
        for (auto op : ops) {
            if (op->getOpType() == OpType::Transpose) {
                auto perm =
                    dynamic_cast<TransposeObj *>(op.get())->getPermute();
                const auto &succVec = op->getSuccessors();
                const bool matmulAll =
                    std::all_of(succVec.begin(), succVec.end(), [](auto x) {
                        return x->getOpType() == OpType::MatMul;
                    });
                if (isMatTrans(perm) && !succVec.empty() && matmulAll) {
                    modified = true;

                    for (auto &suc : succVec) {
                        auto in = op->getInputs(0), out = suc->getOutput();
                        auto mmOp = dynamic_cast<MatmulObj *>(suc.get());

                        std::shared_ptr<MatmulObj> fusedOp = nullptr;
                        // A*B op=A
                        if (suc->getInputs(0) == op->getOutput()) {
                            auto fusedOp = addOpWithOutputs<MatmulObj>(
                                in, mmOp->getInputs(1), out,
                                // transpose
                                !mmOp->getTransA(), mmOp->getTransB());
                        }
                        // A*B op=B
                        else {
                            fusedOp = addOpWithOutputs<MatmulObj>(
                                mmOp->getInputs(0), in, out,
                                // transpose
                                mmOp->getTransA(), !mmOp->getTransB());
                        }
                        // detach the suc operator
                        removeOperator(suc);
                        // NOTE: correct source
                        out->setSource(fusedOp);
                    }
                    // the output buffer will nolonger be used
                    removeTensor(op->getOutput());
                    // finally remove this
                    removeOperator(op);

                    print();
                    break;
                }
            }
        }
        if (modified)
            continue;

        // rule 3: eliminate transposes with GEMM
        for (auto op : ops) {
            if (op->getOpType() == OpType::Transpose) {
                auto perm =
                    dynamic_cast<TransposeObj *>(op.get())->getPermute();
                if (isIdentity(perm)) {
                    modified = true;

                    // (buf) -- id -- (out) -> [op1, op2, op3]
                    // (buf) -> [op1, op2, op3]
                    const auto in = op->getInputs(0);
                    const auto out = op->getOutput();
                    in->removeTarget(op);

                    const auto &succVec = op->getSuccessors();
                    for (auto &suc : succVec) {
                        in->addTarget(suc);
                        suc->removePredecessors(op);

                        *std::find(suc->inputs.begin(), suc->inputs.end(),
                                   out) = in;
                    }
                    // the output buffer will nolonger be used
                    removeTensor(op->getOutput());
                    // finally remove this
                    removeOperator(op);
                    break;
                }
            }
        }
        if (modified)
            continue;

    } while (modified);
}

Tensor GraphObj::getTensor(int fuid) const {
    for (auto tensor : tensors) {
        if (tensor->getFuid() == fuid) {
            return tensor;
        }
    }
    return nullptr;
}

void GraphObj::shape_infer() {
    for (auto &op : ops) {
        auto ans = op->inferShape();
        IT_ASSERT(ans.has_value());
        auto oldOutputs = op->getOutputs();
        IT_ASSERT(ans.value().size() == oldOutputs.size());
        // replace the old outputshape and size with new one
        for (int i = 0; i < (int)ans.value().size(); ++i) {
            auto newShape = ans.value()[i];
            auto oldShape = oldOutputs[i]->getDims();
            auto fuid = oldOutputs[i]->getFuid();
            if (newShape != oldShape) {
                auto tensor = this->getTensor(fuid);
                tensor->setShape(newShape);
            }
        }
    }
}

void GraphObj::dataMalloc() {
    // topological sorting first
    IT_ASSERT(topo_sort() == true);

    // consider this example computation graph:
    // t1 (op-x) t2 (op-y) t3 (op-z) t4 (op-w) t5
    //
    // 1. run op-x, active [t1 t2] (alloc t1 t2)
    // 2. run op-y, active [t2 t3] (free t1, alloc t3)
    // 3. run op-z, active [t3 t4] (free t2, alloc t4)
    // 4. run op-w, active [t4 t5] (free t3, alloc t5)

    std::unordered_map<TensorObj *, size_t> ref;
    std::unordered_map<TensorObj *, size_t> off;
#if 1
    // all input tensors have to be allocated
    for (const auto &in : getInputs()) {
        off[in.get()] = allocator.alloc(in->getBytes());
    }

    // in/out degree counting
    for (const auto &op : ops) {
        for (auto &in : op->getInputs()) {
            ref[in.get()]++;
        }
    }

    // membership testing
    const auto mem = [](const auto &set, const auto &x) {
        return set.find(x) != set.end();
    };

    // execute kernels in topological order
    for (const auto &op : ops) {
        // allocate if need
        for (auto &out : op->getOutputs()) {
            const auto outPtr = out.get();
            if (!mem(off, outPtr)) {
                off[outPtr] = allocator.alloc(out->getBytes());
            }
        }
        // deallocate if ref count = 0
        for (auto &in : op->getInputs()) {
            const auto inPtr = in.get();
            ref[inPtr]--;
            if (ref[inPtr] == 0) {
                allocator.free(off[inPtr], in->getBytes());
            }
        }
    }
#else
    for (auto &t : getTensors()) {
        off[t.get()] = allocator.alloc(t->getBytes());
    }
#endif

    // add offset to pool pointer
    for (auto &t : getTensors()) {
        auto ptr = reinterpret_cast<char *>(allocator.getPtr());
        t->setDataBlob(make_ref<BlobObj>(runtime, ptr + off[t.get()]));
    }

    // print memory usage
    allocator.info();
}

Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
    return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
}

Tensor GraphObj::addTensor(const Tensor &tensor) {
    IT_ASSERT(tensor->getRuntime() == runtime,
              std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                  tensor->getRuntime()->toString() + " to " +
                  runtime->toString());
    tensors.emplace_back(tensor);
    return tensor;
}

TensorVec GraphObj::addTensor(const TensorVec &tensors) {
    for (auto &t : tensors)
        addTensor(t);
    return tensors;
}

// tensor's "source" and "target" must be in "ops".
// tensor has no "source" and no "target" must not exist.
// "inputs" or "outputs" of operators must be in "tensors"
// "predecessors" and "successors" of an operator of "ops" must be in "ops".
bool GraphObj::checkValid() const {
    for (auto tensor : tensors) {
        IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                    nullptr == tensor->getSource()));
        for (auto op : tensor->getTargets()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
        }
        auto op = tensor->getSource();
        IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
    }
    for (auto op : ops) {
        for (auto tensor : op->getInputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                      tensors.end());
        }
        for (auto tensor : op->getOutputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                      tensors.end());
        }
        for (auto pre : op->getPredecessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
        }
        for (auto suc : op->getSuccessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
        }
    }
    std::set<UidBaseType> s;
    // check whether two tensors with the same FUID exist
    for (auto tensor : tensors) {
        int cnt = s.count(tensor->getFuid());
        IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
        s.insert(tensor->getFuid());
    }
    return true;
}

} // namespace infini
