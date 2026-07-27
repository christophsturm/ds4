/* Regression coverage for the machine-readable Metal memory snapshot. */

#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>

static int failures = 0;

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); \
        failures++; \
    } \
} while (0)

int main(void) {
    const uint64_t bytes = 4096;
    const ds4_gpu_memory_snapshot before =
        ds4_gpu_memory_snapshot_current();
    ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(bytes);
    CHECK(tensor != NULL, "allocate tracked Metal tensor");

    const ds4_gpu_memory_snapshot allocated =
        ds4_gpu_memory_snapshot_current();
    CHECK(allocated.tensor_live_bytes == before.tensor_live_bytes + bytes,
          "snapshot includes a live tensor allocation");
    CHECK(allocated.tensor_peak_bytes >= allocated.tensor_live_bytes,
          "snapshot peak covers the current allocation");

    ds4_gpu_tensor_free(tensor);
    const ds4_gpu_memory_snapshot freed =
        ds4_gpu_memory_snapshot_current();
    CHECK(freed.tensor_live_bytes == before.tensor_live_bytes,
          "snapshot removes a freed tensor allocation");
    ds4_gpu_cleanup();

    if (failures != 0) {
        fprintf(stderr, "test_metal_memory_snapshot: %d failure(s)\n",
                failures);
        return 1;
    }
    fprintf(stderr, "test_metal_memory_snapshot: PASS\n");
    return 0;
}
