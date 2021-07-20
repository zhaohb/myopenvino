// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_weights_cache.hpp"

#include <ie_system_conf.h>
#include <memory>

namespace MKLDNNPlugin {

const SimpleDataHash MKLDNNWeightsSharing::simpleCRC;

MKLDNNWeightsSharing::MKLDNNSharedMemory::MKLDNNSharedMemory(
        std::unique_lock<std::mutex> && lock,
        const MKLDNNMemoryInfo::Ptr & memory,
        MKLDNNMemoryPtr newPtr)
    : lock(std::move(lock))
    , memory(memory)
    , newPtr(newPtr)
{}

MKLDNNWeightsSharing::MKLDNNSharedMemory::operator MKLDNNMemoryPtr() const {
    return memory->sharedMemory.lock();
}

bool MKLDNNWeightsSharing::MKLDNNSharedMemory::isValid() const {
    return memory->valid;
}

void MKLDNNWeightsSharing::MKLDNNSharedMemory::valid(bool b) {
    memory->valid = b;
}

MKLDNNWeightsSharing::MKLDNNSharedMemory::Ptr MKLDNNWeightsSharing::findOrCreate(
                            const std::string& key,
                            std::function<MKLDNNMemoryPtr(void)> create,
                            bool valid) {
    std::unique_lock<std::mutex> lock(guard);
    auto found = sharedWeights.find(key);

    MKLDNNMemoryInfo::Ptr ptr;
    MKLDNNMemoryPtr newPtr;

    if (found == sharedWeights.end()
        || !(ptr = found->second)
        || ptr->sharedMemory.expired()) {
        newPtr = create();
        ptr = std::make_shared<MKLDNNMemoryInfo>(newPtr, valid);
        sharedWeights[key] = ptr;
    }

    return std::make_shared<MKLDNNSharedMemory>(ptr->valid
                                                ? std::unique_lock<std::mutex>(ptr->guard, std::defer_lock)
                                                : std::unique_lock<std::mutex>(ptr->guard), ptr, newPtr);
}

MKLDNNWeightsSharing::MKLDNNSharedMemory::Ptr MKLDNNWeightsSharing::get(const std::string& key) const {
    std::unique_lock<std::mutex> lock(guard);
    auto found = sharedWeights.find(key);

    MKLDNNMemoryInfo::Ptr ptr;

    if (found == sharedWeights.end()
        || !(ptr = found->second)
        || ptr->sharedMemory.expired())
        THROW_IE_EXCEPTION << "Unknown shared memory with key " << key;

    return std::make_shared<MKLDNNSharedMemory>(ptr->valid
                                                ? std::unique_lock<std::mutex>(ptr->guard, std::defer_lock)
                                                : std::unique_lock<std::mutex>(ptr->guard), ptr);
}

NumaNodesWeights::NumaNodesWeights() {
    for (auto numa_id : InferenceEngine::getAvailableNUMANodes())
        _cache_map[numa_id] = std::make_shared<MKLDNNWeightsSharing>();
}

MKLDNNWeightsSharing::Ptr& NumaNodesWeights::operator[](int numa_id) {
    auto found = _cache_map.find(numa_id);
    if (found == _cache_map.end())
        THROW_IE_EXCEPTION << "Unknown numa node id " << numa_id;
    return found->second;
}

const MKLDNNWeightsSharing::Ptr& NumaNodesWeights::operator[](int numa_id) const {
    auto found = _cache_map.find(numa_id);
    if (found == _cache_map.end())
        THROW_IE_EXCEPTION << "Unknown numa node id " << numa_id;
    return found->second;
}

}  // namespace MKLDNNPlugin
