#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { INPUT = 256, MID = 512, OUTPUT = 260, EXPERTS = 256, SELECTED = 6 };
typedef struct { uint16_t d; uint8_t qs[64]; } iq2_block;
typedef struct { uint8_t scales[16], qs[64]; uint16_t d, dmin; } q2_block;
typedef struct { uint8_t e, qs[16]; } mxfp4_block;

static uint32_t random_u32(uint32_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

/* The same unfused arithmetic must read only selected expert weights when
 * the resident map deliberately excludes every routed tensor. */
static void check_quality_streaming(bool mxfp4) {
    const uint64_t page = getpagesize();
    const uint64_t gate_row = mxfp4 ? INPUT / 32 * sizeof(mxfp4_block) : sizeof(iq2_block);
    const uint64_t down_row = mxfp4 ? MID / 32 * sizeof(mxfp4_block) : 2 * sizeof(q2_block);
    const uint64_t gate_bytes = MID * gate_row, down_bytes = OUTPUT * down_row;
    const uint64_t up_off = EXPERTS * gate_bytes, down_off = 2 * up_off;
    const uint64_t sentinel = (down_off + EXPERTS * down_bytes + page - 1) / page * page;
    const uint64_t model_size = sentinel + page;
    void *model = NULL;
    assert(posix_memalign(&model, page, model_size) == 0);
    memset(model, 0, model_size);
    uint32_t rng = 1;
    if (mxfp4) {
        mxfp4_block *blocks = model;
        for (uint64_t i = 0; i < (down_off + EXPERTS * down_bytes) / sizeof(*blocks); i++) {
            blocks[i].e = 118 + random_u32(&rng) % 5;
            for (unsigned j = 0; j < sizeof(blocks[i].qs); j++) blocks[i].qs[j] = random_u32(&rng);
        }
    } else {
        iq2_block *gate = model;
        for (uint64_t i = 0; i < down_off / sizeof(*gate); i++) {
            gate[i].d = 0x1400;
            for (unsigned j = 0; j < sizeof(gate[i].qs); j++) gate[i].qs[j] = random_u32(&rng);
        }
        q2_block *down = (q2_block *)((char *)model + down_off);
        for (uint64_t i = 0; i < EXPERTS * down_bytes / sizeof(*down); i++) {
            down[i].d = down[i].dmin = 0x2000;
            for (unsigned j = 0; j < sizeof(down[i].scales); j++) down[i].scales[j] = random_u32(&rng);
            for (unsigned j = 0; j < sizeof(down[i].qs); j++) down[i].qs[j] = random_u32(&rng);
        }
    }
    FILE *model_file = tmpfile();
    assert(model_file);
    assert(fwrite(model, 1, model_size, model_file) == model_size);
    assert(fflush(model_file) == 0);
    const int32_t ids[SELECTED] = {0, 17, 95, 128, 200, 255};
    const float weights[SELECTED] = {0.05f, 0.1f, 0.15f, 0.2f, 0.22f, 0.28f};
    float input[INPUT];
    for (int i = 0; i < INPUT; i++) input[i] = ((int)(random_u32(&rng) % 101) - 50) / 256.0f;
    const uint64_t counts[5] = {SELECTED * MID, SELECTED * MID, SELECTED * MID,
                               SELECTED * OUTPUT, OUTPUT};
    ds4_gpu_tensor *result[5] = {0};
    float *reference[5] = {0};
    float actual[SELECTED * MID];
    for (int i = 0; i < 5; i++) {
        reference[i] = malloc(counts[i] * sizeof(float));
        assert(reference[i]);
    }
    for (int streaming = 0; streaming < 2; streaming++) {
        /* A fresh context prevents the resident reference's view cache from
         * satisfying reads that the streaming map deliberately excludes. */
        assert(ds4_gpu_init());
        ds4_gpu_set_quality(true);
        ds4_gpu_set_ssd_streaming(streaming != 0);
        assert(ds4_gpu_set_model_fd(fileno(model_file)));
        if (streaming) {
            ds4_gpu_set_streaming_expert_cache_budget(SELECTED);
            ds4_gpu_set_streaming_expert_cache_expert_bytes(2 * gate_bytes + down_bytes);
            assert(ds4_gpu_set_model_map_spans(model, model_size, &sentinel, &page, 1, page));
        } else {
            assert(ds4_gpu_set_model_map(model, model_size));
        }
        ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(sizeof(input));
        ds4_gpu_tensor *it = ds4_gpu_tensor_alloc(sizeof(ids));
        ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(sizeof(weights));
        assert(xt && it && wt);
        assert(ds4_gpu_tensor_write(xt, 0, input, sizeof(input)));
        assert(ds4_gpu_tensor_write(it, 0, ids, sizeof(ids)));
        assert(ds4_gpu_tensor_write(wt, 0, weights, sizeof(weights)));
        for (int i = 0; i < 5; i++) {
            result[i] = ds4_gpu_tensor_alloc(counts[i] * sizeof(float));
            assert(result[i]);
        }
        for (int batch = 0; batch <= streaming; batch++) {
            for (int i = 0; i < 5; i++) assert(ds4_gpu_tensor_fill_f32(result[i], NAN, counts[i]));
            if (batch) {
                assert(ds4_gpu_routed_moe_set_selected_override(ids, SELECTED));
                assert(ds4_gpu_begin_commands());
            }
            int decoded = ds4_gpu_routed_moe_one_tensor(
                result[4], result[0], result[1], result[2], result[3],
                model, model_size, 0, up_off, down_off, mxfp4 ? 39 : 16, mxfp4 ? 39 : 10,
                gate_bytes, gate_row, down_bytes, down_row, INPUT, MID, OUTPUT,
                it, wt, EXPERTS, SELECTED, 7.0f, xt, NULL, 0, false);
            int completed = batch ? ds4_gpu_end_commands() : 1;
            assert(decoded && "quality decode must work without mapping unselected experts");
            assert(completed);
            for (int i = 0; i < 5; i++) {
                assert(ds4_gpu_tensor_read(result[i], 0, actual, counts[i] * sizeof(float)));
                for (uint64_t j = 0; j < counts[i]; j++) assert(isfinite(actual[j]));
                if (streaming) assert(memcmp(reference[i], actual, counts[i] * sizeof(float)) == 0);
                else memcpy(reference[i], actual, counts[i] * sizeof(float));
            }
        }
        if (streaming) assert(ds4_gpu_stream_expert_cache_current_count() == SELECTED);
        for (int i = 0; i < 5; i++) ds4_gpu_tensor_free(result[i]);
        ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(it); ds4_gpu_tensor_free(wt);
        ds4_gpu_cleanup();
    }
    for (int i = 0; i < 5; i++) free(reference[i]);
    fclose(model_file);
    free(model);
    printf("quality streaming %s: exact intermediates, %d cached experts: PASS\n",
           mxfp4 ? "MXFP4" : "IQ2/Q2", SELECTED);
}

int main(void) {
    check_quality_streaming(false);
    check_quality_streaming(true);
    return 0;
}
