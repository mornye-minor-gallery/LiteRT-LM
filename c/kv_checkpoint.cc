#include "c/engine_internal.h"
#include "c/engine.h"

extern "C" int litert_lm_session_transfer_state(
    LiteRtLmSession* session, LiteRtLmStateTransfer transfer, void* user_data,
    bool reading) {
  if (!session || !transfer) return 3;
  auto status = session->session->TransferState(transfer, user_data, reading);
  return static_cast<int>(status.code());
}
