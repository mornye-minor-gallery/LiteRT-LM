#include "runtime/core/session_advanced.h"
#include "absl/synchronization/mutex.h"
namespace litert::lm {
absl::Status SessionAdvanced::TransferState(bool (*transfer)(void*, size_t, void*), void* user_data, bool reading) {
  absl::MutexLock lock(mutex_);
  auto manager = execution_manager_.lock();
  if (!manager) return absl::FailedPreconditionError("manager unavailable");
  if (reading && session_state_ != SessionState::kFresh)
    return absl::FailedPreconditionError("restore requires fresh session");
  auto drained = manager->WaitUntilSessionDone(session_id_, Engine::kDefaultTimeout);
  if (!drained.ok()) return drained;
  auto status = manager->TransferState(session_id_, transfer, user_data, reading);
  if (status.ok() && reading) {
    session_state_ = SessionState::kDecoded;
    last_task_ids_.clear();
  }
  return status;
}
}
