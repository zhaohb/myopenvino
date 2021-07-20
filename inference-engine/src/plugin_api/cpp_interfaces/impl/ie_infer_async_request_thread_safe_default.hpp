// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <threading/ie_immediate_executor.hpp>
#include <threading/ie_itask_executor.hpp>
#include <threading/ie_istreams_executor.hpp>

#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>
#include <cpp_interfaces/interface/ie_iinfer_async_request_internal.hpp>
#include <cpp_interfaces/exception2status.hpp>
#include <ie_system_conf.h>

#include <exception>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace InferenceEngine {

/**
 * @ingroup ie_dev_api_async_infer_request_api
 * @brief Base class with default implementation of asynchronous multi staged inference request.
 *        To customize pipeline stages derived class should change the content
 *        of AsyncInferRequestThreadSafeDefault::_pipeline member container.
 *        It consists of pairs of tasks and executors which will run the task.
 *        The class is recommended to be used by plugins as a base class for asynchronous inference request implementation.
 * @note  To synchronize derived context with stages
 *        derived class should call AsyncInferRequestThreadSafeDefault::StopAndWait() function in destructor.
 * @par Example
 *        Here is an example of asynchronous inference request implementation for some accelerator device.
 *        It uses 5 different executors to run different stages of a synchronous inference request.
 *
 * @snippet example_async_infer_request.cpp async_infer_request:define_pipeline
 */
class AsyncInferRequestThreadSafeDefault : public IAsyncInferRequestInternal {
    enum InferState {Idle, Busy, Canceled, Stop};
    using Futures = std::vector<std::shared_future<void>>;
    using Promise = std::shared_ptr<std::promise<void>>;
    enum Stage_e : std::uint8_t { executor, task };
    InferRequestInternal::Ptr _syncRequest;

    friend struct DisableCallbackGuard;
    struct DisableCallbackGuard {
        explicit DisableCallbackGuard(AsyncInferRequestThreadSafeDefault* this_)
            : _this{this_} {
                std::lock_guard<std::mutex> lock{_this->_mutex};
                std::swap(_callback, _this->_callback);
            }
        ~DisableCallbackGuard() {
            std::lock_guard<std::mutex> lock{_this->_mutex};
            _this->_callback = _callback;
        }
        AsyncInferRequestThreadSafeDefault* _this = nullptr;
        IInferRequest::CompletionCallback _callback = nullptr;
    };

    struct ImmediateStreamsExecutor : public InferenceEngine::ITaskExecutor {
        explicit ImmediateStreamsExecutor(const IStreamsExecutor::Ptr& streamsExecutor) : _streamsExecutor{streamsExecutor} {}
        void run(InferenceEngine::Task task) override {_streamsExecutor->Execute(std::move(task));}
        IStreamsExecutor::Ptr _streamsExecutor;
    };

    template<typename F>
    void InferImpl(const F& f) {
        _syncRequest->checkBlobs();
        InferState state = InferState::Idle;
        {
            std::lock_guard<std::mutex> lock{_mutex};
            state = _state;
            switch (_state) {
            case InferState::Busy :
                THROW_IE_EXCEPTION_WITH_STATUS(REQUEST_BUSY);
            case InferState::Canceled :
                THROW_IE_EXCEPTION_WITH_STATUS(INFER_CANCELLED);
            case InferState::Idle : {
                _futures.erase(std::remove_if(std::begin(_futures), std::end(_futures),
                                                [](const std::shared_future<void>& future) {
                                                    if (future.valid()) {
                                                        return (std::future_status::ready ==
                                                                future.wait_for(std::chrono::milliseconds {0}));
                                                    } else {
                                                        return true;
                                                    }
                                                }),
                                _futures.end());
                _promise = {};
                _futures.emplace_back(_promise.get_future().share());
            } break;
            case InferState::Stop : break;
            }
            _state = InferState::Busy;
        }
        if (state != InferState::Stop) {
            try {
                f();
            } catch (...) {
                _promise.set_exception(std::current_exception());
                std::lock_guard<std::mutex> lock{_mutex};
                _state = InferState::Idle;
                throw;
            }
        }
    }

protected:
    /**
     * @brief Throws exception if inference request is busy or canceled
     */
    void CheckState() const {
        std::lock_guard<std::mutex> lock {_mutex};
        switch (_state) {
        case InferState::Busy :
            THROW_IE_EXCEPTION_WITH_STATUS(REQUEST_BUSY);
        case InferState::Canceled :
            THROW_IE_EXCEPTION_WITH_STATUS(INFER_CANCELLED);
        default: break;
        }
    }

public:
    /**
     * @brief A shared pointer to AsyncInferRequestThreadSafeDefault
     */
    using Ptr = std::shared_ptr<AsyncInferRequestThreadSafeDefault>;

    /**
     * @brief      Wraps a InferRequestInternal::Ptr implementation and constructs a
     * AsyncInferRequestThreadSafeDefault::_pipeline where `taskExecutor` is used to run InferRequestInternal::Infer
     * asynchronously.
     *
     * @param[in]  request           The synchronous request
     * @param[in]  taskExecutor      The task executor
     * @param[in]  callbackExecutor  The callback executor
     */
    AsyncInferRequestThreadSafeDefault(const InferRequestInternal::Ptr& request,
                                       const ITaskExecutor::Ptr& taskExecutor,
                                       const ITaskExecutor::Ptr& callbackExecutor) :
        _syncRequest {request},
        _requestExecutor {taskExecutor},
        _callbackExecutor {callbackExecutor},
        _pipeline {{taskExecutor, [this] {_syncRequest->InferImpl();}}},
        _syncPipeline {{std::make_shared<ImmediateExecutor>(), [this] {_syncRequest->InferImpl();}}} {
        auto streamsExecutor = std::dynamic_pointer_cast<IStreamsExecutor>(taskExecutor);
        if (streamsExecutor != nullptr) {
            _syncPipeline = {{std::make_shared<ImmediateStreamsExecutor>(std::move(streamsExecutor)), [this] {_syncRequest->InferImpl();}}};
        }
    }

    /**
     * @brief      Destroys the object, stops AsyncInferRequestThreadSafeDefault::_pipeline and waits for a finish.
     */
    ~AsyncInferRequestThreadSafeDefault() {
        StopAndWait();
    }

    /**
     * @brief Waits for completion of all pipeline stages
     *        If the pipeline raises an exception it will be rethrown here
     * @param millis_timeout A timeout is `ms` to wait or special enum value of IInferRequest::WaitMode
     * @return A status code
     */
    StatusCode Wait(int64_t millis_timeout) override {
        if (millis_timeout < IInferRequest::WaitMode::RESULT_READY) {
            THROW_IE_EXCEPTION_WITH_STATUS(PARAMETER_MISMATCH)
                << " Timeout can't be less "
                << IInferRequest::WaitMode::RESULT_READY << " for InferRequest::Wait\n";
        }
        auto status = std::future_status::deferred;

        // Just use the last '_futures' member to wait pipeline completion
        auto future = [&] {
            std::lock_guard<std::mutex> lock {_mutex};
            return _futures.empty() ? std::shared_future<void> {} : _futures.back();
        }();

        if (!future.valid()) {
            return StatusCode::INFER_NOT_STARTED;
        }

        switch (millis_timeout) {
        case IInferRequest::WaitMode::RESULT_READY: {
            future.wait();
            status = std::future_status::ready;
        } break;
        case IInferRequest::WaitMode::STATUS_ONLY: {
            status = future.wait_for(std::chrono::milliseconds {0});
        } break;
        default: {
            status = future.wait_for(std::chrono::milliseconds {millis_timeout});
        } break;
        }

        if (std::future_status::ready == status) {
            future.get();
            return StatusCode::OK;
        } else {
            return StatusCode::RESULT_NOT_READY;
        }
    }

    void StartAsync() override {
        InferImpl([&] {StartAsync_ThreadUnsafe();});
    }

    void Infer() override {
        DisableCallbackGuard disableCallbackGuard{this};
        InferImpl([&] {Infer_ThreadUnsafe();});
        Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    }

    std::map<std::string, InferenceEngineProfileInfo> GetPerformanceCounts() const override {
        CheckState();
        return _syncRequest->GetPerformanceCounts();
    }

    void SetBlob(const std::string& name, const Blob::Ptr& data) override {
        CheckState();
        _syncRequest->SetBlob(name, data);
    }

    void SetBlob(const std::string& name, const Blob::Ptr& data, const PreProcessInfo& info) override {
        CheckState();
        _syncRequest->SetBlob(name, data, info);
    }

    Blob::Ptr GetBlob(const std::string& name) override {
        CheckState();
        return _syncRequest->GetBlob(name);
    }

    const PreProcessInfo& GetPreProcess(const std::string& name) const override {
        return _syncRequest->GetPreProcess(name);
    }

    void SetBatch(int batch) override {
        CheckState();
        _syncRequest->SetBatch(batch);
    };

    void GetUserData(void** data) override {
        CheckState();
        if (data == nullptr) THROW_IE_EXCEPTION << NOT_ALLOCATED_str;
        *data = _userData;
    }

    void SetUserData(void* data) override {
        CheckState();
        _userData = data;
    }

    void SetCompletionCallback(IInferRequest::CompletionCallback callback) override {
        CheckState();
        _callback = callback;
    }

    /**
     * @brief Sets the pointer to public interface.
     * @note Needed to correctly handle ownership between objects
     * @param[in]  ptr A shared pointer to a public IInferRequest interface.
     */
    void SetPointerToPublicInterface(InferenceEngine::IInferRequest::Ptr ptr) {
        _publicInterface = std::shared_ptr<IInferRequest>(ptr.get(), [](IInferRequest*) {});
    }

    std::vector<InferenceEngine::IVariableStateInternal::Ptr> QueryState() override {
        return _syncRequest->QueryState();
    }

    void ThrowIfCanceled() const {
        std::lock_guard<std::mutex> lock{_mutex};
        if (_state == InferState::Canceled) {
            THROW_IE_EXCEPTION_WITH_STATUS(INFER_CANCELLED);
        }
    }

    void Cancel() override {
        std::lock_guard<std::mutex> lock{_mutex};
        if (_state == InferState::Busy) {
            _state = InferState::Canceled;
        }
    }

protected:
    /**
     * @brief Each pipeline stage is a @ref Task that is executed by specified ITaskExecutor implementation
     */
    using Stage = std::pair<ITaskExecutor::Ptr, Task>;
    /**
     * @brief Pipeline is vector of stages
     */
    using Pipeline = std::vector<Stage>;

    /**
     * @brief Creates and run the first stage task. If destructor was not called add a new std::future to the
     * AsyncInferRequestThreadSafeDefault::_futures list that would be used to wait
     * AsyncInferRequestThreadSafeDefault::_pipeline finish
     * @param[in]  itBeginStage Iterator to begin of pipeline
     * @param[in]  itEndStage End pipeline iterator
     * @param[in]  callbackExecutor Final or error stage executor
     */
    void RunFirstStage(const Pipeline::iterator itBeginStage, const Pipeline::iterator itEndStage,
                       const ITaskExecutor::Ptr callbackExecutor = {}) {
        auto& firstStageExecutor = std::get<Stage_e::executor>(*itBeginStage);
        IE_ASSERT(nullptr != firstStageExecutor);
        firstStageExecutor->run(MakeNextStageTask(itBeginStage, itEndStage, std::move(callbackExecutor)));
    }

    /**
     * @brief Forbids pipeline start and wait for all started pipelines.
     * @note Should be called in derived class destructor to wait for completion of usage of derived context captured by
     * pipeline tasks
     */
    void StopAndWait() {
        _callback = nullptr;
        Futures futures;
        InferState state = InferState::Idle;
        {
            std::lock_guard<std::mutex> lock{_mutex};
            state = _state;
            if (state != InferState::Stop) {
                _state = InferState::Stop;
                futures = std::move(_futures);
            }
        }
        if (state != InferState::Stop) {
            for (auto&& future : futures) {
                if (future.valid()) {
                    future.wait();
                }
            }
        }
    }


    ITaskExecutor::Ptr _requestExecutor;  //!< Used to run inference CPU tasks.
    ITaskExecutor::Ptr _callbackExecutor;  //!< Used to run post inference callback in asynchronous pipline
    ITaskExecutor::Ptr _syncCallbackExecutor;  //!< Used to run post inference callback in synchronous pipline
    Pipeline _pipeline;  //!< Pipeline variable that should be filled by inherited class.
    Pipeline _syncPipeline;  //!< Synchronous pipeline variable that should be filled by inherited class.

    /**
     * @brief Starts an asynchronous pipeline thread unsafe.
     * @note Used by StartAsync which ensures thread-safety and calls this method after.
     */
    virtual void StartAsync_ThreadUnsafe() {
        RunFirstStage(_pipeline.begin(), _pipeline.end(), _callbackExecutor);
    }

    /**
     * @brief Performs inference of pipeline in syncronous mode
     * @note Used by Infer which ensures thread-safety and calls this method after.
     */
    virtual void Infer_ThreadUnsafe() {
        RunFirstStage(_syncPipeline.begin(), _syncPipeline.end(), _syncCallbackExecutor);
    }

    /**
     * @brief Implements Infer() using StartAsync() and Wait()
     */
    void InferUsingAsync() {
        StartAsync_ThreadUnsafe();
    }

private:
    /**
     * @brief Create a task with next pipeline stage.
     * Each call to MakeNextStageTask() generates @ref Task objects for each stage.
     * On last stage or if the exception is raised from `_pipeline` task
     * the last stage task is called or passed to callback executor if it is presented. The last stage task call the
     * callback, if it is presented, capture the `_promise` member and use it to forward completion or exception to the
     * one of `_futures` member
     * @param[in]  itStage Iterator to next stage of pipeline
     * @param[in]  itEndStage End pipeline iterator
     * @param[in]  callbackExecutor Executor that will run final stage with callback call
     * @return A next stage task
     */
    Task MakeNextStageTask(const Pipeline::iterator itStage, const Pipeline::iterator itEndStage,
                           const ITaskExecutor::Ptr callbackExecutor) {
        return std::bind([this, itStage, itEndStage](ITaskExecutor::Ptr& callbackExecutor) mutable {
            StatusCode requestStatus = StatusCode::OK;
            std::exception_ptr localCurrentException = nullptr;
            auto& thisStage = *itStage;
            auto itNextStage = itStage + 1;

            try {
                auto& stageTask = std::get<Stage_e::task>(thisStage);
                IE_ASSERT(nullptr != stageTask);
                stageTask();
               if (itEndStage != itNextStage) {
                    auto& nextStage = *itNextStage;
                    auto& nextStageExecutor = std::get<Stage_e::executor>(nextStage);
                    IE_ASSERT(nullptr != nextStageExecutor);
                    nextStageExecutor->run(MakeNextStageTask(itNextStage, itEndStage, std::move(callbackExecutor)));
                }
            } catch (InferenceEngine::details::InferenceEngineException& ie_ex) {
                requestStatus = ie_ex.hasStatus() ? ie_ex.getStatus() : StatusCode::GENERAL_ERROR;
                localCurrentException = std::make_exception_ptr(ie_ex);
            } catch (...) {
                requestStatus = StatusCode::GENERAL_ERROR;
                localCurrentException = std::current_exception();
            }

            if ((itEndStage == itNextStage) || (nullptr != localCurrentException)) {
                auto lastStageTask = [this, requestStatus, localCurrentException]() mutable {
                    auto promise = std::move(_promise);
                    IInferRequest::CompletionCallback callback = nullptr;
                    {
                        std::lock_guard<std::mutex> lock{_mutex};
                        _state = InferState::Idle;
                        callback = _callback;
                    }
                    if (nullptr != callback) {
                        InferenceEngine::CurrentException() = localCurrentException;
                        try {
                            callback(_publicInterface, requestStatus);
                        } catch (...) {
                            localCurrentException = std::current_exception();
                        }
                        InferenceEngine::CurrentException() = nullptr;
                    }
                    if (nullptr == localCurrentException) {
                        promise.set_value();
                    } else {
                        promise.set_exception(localCurrentException);
                    }
                };

                if (nullptr == callbackExecutor) {
                    lastStageTask();
                } else {
                    callbackExecutor->run(std::move(lastStageTask));
                }
            }
        }, std::move(callbackExecutor));
    }

    void* _userData = nullptr;
    IInferRequest::CompletionCallback _callback = nullptr;
    IInferRequest::Ptr _publicInterface;
    std::promise<void> _promise;
    mutable std::mutex _mutex;
    Futures _futures;
    InferState _state = InferState::Idle;
};
}  // namespace InferenceEngine
