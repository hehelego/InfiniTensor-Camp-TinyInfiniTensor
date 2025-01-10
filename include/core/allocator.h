#pragma once
#include "core/object.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include <set>
#ifdef BUILD_TEST
#include "gtest/gtest.h"
#endif
#include <cstddef>
#include <map>
#include <unordered_set>

namespace infini {
class Allocator {
  private:
    Runtime runtime;

    size_t used;

    size_t peak;

    size_t alignment;

    // pointer to the memory actually allocated
    void *ptr;

    struct Block {
        size_t begin, size;
        inline bool operator<(const Block &rhs) const {
            return size != rhs.size ? size < rhs.size : begin < rhs.begin;
        }
    };
    std::set<Block> frees;

  public:
    Allocator(Runtime runtime);

    virtual ~Allocator();

    // function: simulate memory allocation
    // arguments：
    //     size: size of memory block to be allocated
    // return: head address offset of the allocated memory block
    size_t alloc(size_t size);

    // function: simulate memory free
    // arguments:
    //     addr: head address offset of memory block to be free
    //     size: size of memory block to be freed
    void free(size_t addr, size_t size);

    // function: perform actual memory allocation
    // return: pointer to the head address of the allocated memory
    void *getPtr();

    void info();

  private:
    // function: memory alignment, rouned up
    // return: size of the aligned memory block
    size_t getAlignedSize(size_t size);
};
} // namespace infini
