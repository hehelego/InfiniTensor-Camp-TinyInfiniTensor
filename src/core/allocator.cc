#include "core/allocator.h"

namespace infini {
Allocator::Allocator(Runtime runtime) : runtime(runtime) {
    used = 0;
    peak = 0;
    ptr = nullptr;

    // 'alignment' defaults to sizeof(uint64_t), because it is the length of
    // the longest data type currently supported by the DataType field of
    // the tensor
    alignment = sizeof(uint64_t);
}

Allocator::~Allocator() {
    if (this->ptr != nullptr) {
        runtime->dealloc(this->ptr);
    }
}

size_t Allocator::alloc(size_t size) {
    IT_ASSERT(this->ptr == nullptr);
    // pad the size to the multiple of alignment
    size = this->getAlignedSize(size);

    used += size;

    size_t pos = 0;

    // the first (smallest) block whose size is greater or equal to size
    // v[i]=(pos, sz) < (0, k) := (sz /= k and sz < k) or (sz = k and pos < 0)
    // v[i]=(pos, sz) < (0, k) := sz < k
    // v[i]=(pos, sz) >= (0, k) := sz >= k
    auto i = frees.lower_bound(Block{0, size});
    if (i != frees.end()) {
        auto blk = *i;
        pos = blk.begin;
        frees.erase(i);

        // still got some space available
        if (blk.size < size) {
            blk.begin += size;
            blk.size -= size;
            frees.insert(blk);
        }

    } else if (!frees.empty()) {
        // extend the last free block
        auto k = frees.begin();
        for (auto i = frees.begin(); i != frees.end(); i++) {
            if (k->begin < i->begin) {
                k = i;
            }
        }
        const auto blk = *k;
        frees.erase(k);
        pos = blk.begin;

        peak = std::max(peak, pos + size - blk.size);
    } else {
        // cannot fit in a free block
        // so we allocate a separate consecutive block
        pos = peak;
        peak += size;
    }

    return pos;
}

void Allocator::free(size_t addr, size_t size) {
    IT_ASSERT(this->ptr == nullptr);
    size = getAlignedSize(size);

    used -= size;

    frees.insert(Block{addr, size});
}

void *Allocator::getPtr() {
    if (this->ptr == nullptr) {
        this->ptr = runtime->alloc(this->peak);
    }
    return this->ptr;
}

size_t Allocator::getAlignedSize(size_t size) {
    return ((size - 1) / this->alignment + 1) * this->alignment;
}

void Allocator::info() {
    std::cout << "Used memory: " << this->used
              << ", peak memory: " << this->peak << std::endl;
}
} // namespace infini
