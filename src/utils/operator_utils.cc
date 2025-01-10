#include "utils/operator_utils.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include <algorithm>

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {
    // https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    // 1. exactly the same shape.
    // 2. same rank AND (n:n OR 1:n in every dimension)
    // 3. [1,1,1 ... A] <=> B

    auto n = std::max(A.size(), B.size());
    auto padA = n - A.size();
    auto padB = n - B.size();
    Shape shape(n);
    for (size_t i = 0; i < n; i++) {
        auto la = i < padA ? 1 : A[i - padA];
        auto lb = i < padB ? 1 : B[i - padB];
        shape[i] = std::max(la, lb);
    }
    return shape;
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini
