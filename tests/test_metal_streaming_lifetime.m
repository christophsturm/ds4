/* Compile the Metal implementation here so weak references can observe actual
 * resource destruction, rather than trusting counters reset by cache cleanup.
 * Cache population and invalidation still use the public GPU API. */
#include "../ds4_metal.m"

static int seed(const ds4_gpu_stream_expert_table *table, int mode) {
    const int32_t selected[] = {0, 1, 2, 3};
    const uint32_t priorities[] = {4, 3, 2, 1};
    switch (mode) {
    case 0:
        return ds4_gpu_stream_expert_cache_seed_selected(table, selected, 4);
    case 1:
        return ds4_gpu_stream_expert_cache_seed_experts(table, selected, priorities, 4);
    case 2:
        if (!ds4_gpu_begin_commands()) return 0;
        if (!ds4_gpu_stream_expert_cache_seed_experts_gpu_copy(table, selected, priorities, 4)) return 0;
        return ds4_gpu_end_commands();
    case 3:
        return ds4_gpu_stream_expert_cache_begin_selected_load(table, selected, 4);
    }
    return 0;
}

int main(void) {
    enum { experts = 8, expert_bytes = 4096, bytes = 3 * experts * expert_bytes };
    FILE *file = tmpfile();
    if (!file || ftruncate(fileno(file), bytes)) return 1;
    void *model = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(file), 0);
    if (model == MAP_FAILED) return 1;
    memset(model, 1, bytes);
    int failures = 0;
    for (int mode = 0; mode < 4; mode++) {
      @autoreleasepool {
        if (!ds4_gpu_init()) return 1;
        ds4_gpu_set_ssd_streaming(true);
        ds4_gpu_set_streaming_expert_cache_budget(experts);
        if (!ds4_gpu_set_model_fd(fileno(file))) return 1;
        if (!ds4_gpu_set_model_map_range(model, bytes, 0, bytes, expert_bytes)) return 1;
        const ds4_gpu_stream_expert_table table = {
            .model_map = model, .model_size = bytes, .layer = 0,
            .n_total_expert = experts, .gate_offset = 0,
            .up_offset = experts * expert_bytes, .down_offset = 2 * experts * expert_bytes,
            .gate_expert_bytes = expert_bytes, .down_expert_bytes = expert_bytes,
        };
        if (!seed(&table, mode)) return 1;
        __weak id<MTLBuffer> first = g_stream_expert_cache_slabs[0];
        const bool populated = first != nil;
        ds4_gpu_set_streaming_expert_cache_budget(experts);
        const bool released_on_reset = first == nil;
        if (!seed(&table, mode)) return 1;
        __weak id<MTLBuffer> second = g_stream_expert_cache_slabs[0];
        const bool repopulated = second != nil;
        ds4_gpu_cleanup();
        const bool released_on_cleanup = second == nil;
        if (!populated || !repopulated || !released_on_reset || !released_on_cleanup) failures++;
        fprintf(stderr, "streaming slab lifetime mode %d: populated=%d reset_released=%d repopulated=%d cleanup_released=%d\n",
                mode, populated, released_on_reset, repopulated, released_on_cleanup);
      }
    }
    munmap(model, bytes);
    fclose(file);
    return failures != 0;
}
