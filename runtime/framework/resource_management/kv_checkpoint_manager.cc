#include "runtime/framework/resource_management/threaded_execution_manager.h"
#include "absl/synchronization/mutex.h"
namespace litert::lm {
absl::Status ThreadedExecutionManager::TransferState(SessionId id, bool (*transfer)(void*, size_t, void*), void* user_data, bool reading) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (session_lookup_.size() != 1 || !session_lookup_.contains(id))
    return absl::FailedPreconditionError("KV checkpoint requires one live session");
  auto& info = *session_lookup_.at(id);
  if (!info.active_tasks.empty()) return absl::FailedPreconditionError("session busy");
  auto executor = resource_manager_->AcquireExecutorWithContextHandler(info.context_handler);
  if (!executor.ok()) return executor.status();
  return executor.value()->TransferState(transfer, user_data, reading);
}
}
