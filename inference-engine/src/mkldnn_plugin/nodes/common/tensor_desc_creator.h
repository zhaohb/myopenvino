// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_layouts.h>

namespace MKLDNNPlugin {

enum class TensorDescCreatorTypes : unsigned {
    nspc,       // general per channels format
    ncsp,        // general planar
    nCsp8c,     // general channels blocked by 8
    nCsp16c    // general channels blocked by 16
};

class CreatorsMapFilterConstIterator;

class TensorDescCreator {
public:
    typedef std::shared_ptr<TensorDescCreator> CreatorPtr;
    typedef std::shared_ptr<const TensorDescCreator> CreatorConstPtr;
    typedef std::map<TensorDescCreatorTypes, CreatorConstPtr> CreatorsMap;
    typedef std::function<bool(const CreatorsMap::value_type&)> Predicate;

public:
    static const CreatorsMap& getCommonCreators();
    static std::pair<CreatorsMapFilterConstIterator, CreatorsMapFilterConstIterator>
    makeFilteredRange(const CreatorsMap &map, unsigned rank);
    static std::pair<CreatorsMapFilterConstIterator, CreatorsMapFilterConstIterator>
    makeFilteredRange(const CreatorsMap& map, unsigned rank, const std::vector<TensorDescCreatorTypes>& supportedTypes);
    static std::pair<CreatorsMapFilterConstIterator, CreatorsMapFilterConstIterator>
    makeFilteredRange(const CreatorsMap& map, Predicate predicate);
    virtual InferenceEngine::TensorDesc createDesc(const InferenceEngine::Precision& precision, const InferenceEngine::SizeVector& srcDims) const = 0;
    virtual size_t getMinimalRank() const = 0;
    virtual ~TensorDescCreator() = default;
};

class CreatorsMapFilterConstIterator {
public:
    typedef TensorDescCreator::CreatorsMap::const_iterator Iterator;
    typedef std::iterator_traits<Iterator>::value_type value_type;
    typedef std::iterator_traits<Iterator>::reference reference;
    typedef std::iterator_traits<Iterator>::pointer pointer;
    typedef std::iterator_traits<Iterator>::difference_type difference_type;
    typedef std::forward_iterator_tag iterator_category;
    typedef std::function<bool(const value_type&)> predicate_type;

public:
    CreatorsMapFilterConstIterator(predicate_type filter, Iterator begin, Iterator end) : _filter(std::move(filter)), _iter(begin), _end(end)  {
        while (_iter != _end && !_filter(*_iter)) {
            ++_iter;
        }
    }
    CreatorsMapFilterConstIterator& operator++() {
        do {
            ++_iter;
        } while (_iter != _end && !_filter(*_iter));
        return *this;
    }

    CreatorsMapFilterConstIterator end() const {
        return CreatorsMapFilterConstIterator(predicate_type(), _end, _end);
    }

    CreatorsMapFilterConstIterator operator++(int) {
        CreatorsMapFilterConstIterator temp(*this);
        ++*this;
        return temp;
    }

    reference operator*() const {
        return *_iter;
    }

    pointer operator->() const {
        return std::addressof(*_iter);
    }

    friend bool operator==(const CreatorsMapFilterConstIterator& lhs, const CreatorsMapFilterConstIterator& rhs) {
        return lhs._iter == rhs._iter;
    }

    friend bool operator!=(const CreatorsMapFilterConstIterator& lhs, const CreatorsMapFilterConstIterator& rhs) {
        return !(lhs == rhs);
    }

private:
    Iterator _iter;
    Iterator _end;
    predicate_type _filter;
};
} // namespace MKLDNNPlugin