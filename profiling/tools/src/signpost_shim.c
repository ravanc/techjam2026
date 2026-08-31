// Minimal os_signpost bridge for Python.
//
// os_signpost_* are macros that require compile-time format strings placed in
// the image's __TEXT,__os_log section, so they cannot be called through ctypes
// directly. This shim wraps them behind a plain C ABI that ctypes can reach.
//
// Signposts land in the "Points of Interest" category, which Instruments
// graphs next to the Metal/GPU tracks.

#include <os/log.h>
#include <os/signpost.h>
#include <stdint.h>

static os_log_t g_log;

__attribute__((constructor)) static void tj_init(void) {
  g_log = os_log_create("com.techjam.profiling", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
}

int tj_signpost_enabled(void) {
  return g_log != NULL && os_signpost_enabled(g_log);
}

uint64_t tj_signpost_id(void) {
  return g_log ? os_signpost_id_generate(g_log) : 0;
}

// `name` is passed as an argument rather than a literal so callers can label
// regions dynamically; Instruments renders it as the interval's message.
void tj_interval_begin(uint64_t spid, const char *name) {
  if (g_log) os_signpost_interval_begin(g_log, spid, "region", "%{public}s", name);
}

void tj_interval_end(uint64_t spid, const char *name) {
  if (g_log) os_signpost_interval_end(g_log, spid, "region", "%{public}s", name);
}

void tj_event(const char *name) {
  if (g_log)
    os_signpost_event_emit(g_log, OS_SIGNPOST_ID_EXCLUSIVE, "event", "%{public}s", name);
}
