#include "golden_distribution.h"

#include <math.h>
#include <stdio.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); return 1; \
} } while (0)

int main(void) {
    /* Three of the old top five remain, but all alternatives are negligible.
     * Their combined mass is bounded by 6 * exp(-19.8), below 1e-7. */
    const golden_logit concentrated[] = {
        {0, 20.0f}, {1, 0.0f}, {2, -0.1f}, {3, -0.2f},
        {4, -0.3f}, {5, -0.4f}, {6, -0.5f},
    };
    const float reordered[] = {20.0f, 0.0f, -0.1f, -3.0f, -4.0f, 0.2f, 0.1f};
    CHECK(golden_distribution_distance(concentrated, 7, reordered, 7) < 1e-7);

    /* Greedy output stays the same while 20% of probability moves from the
     * first token to the second: TV((.8,.1,.1), (.6,.3,.1)) = .2. */
    const golden_logit spread[] = {{0, logf(8.0f)}, {1, 0.0f}, {2, 0.0f}};
    const float shifted[] = {logf(6.0f), logf(3.0f), 0.0f};
    const double shifted_distance = golden_distribution_distance(spread, 3, shifted, 3);
    CHECK(fabs(shifted_distance - 0.2) < 1e-7);
    CHECK(shifted_distance > 0.0001);

    /* Common logit offsets do not change a softmax distribution. */
    const golden_logit full[] = {{2, 2.0f}, {0, 1.0f}, {1, 0.0f}};
    const float offset[] = {101.0f, 100.0f, 102.0f};
    CHECK(golden_distribution_distance(full, 3, offset, 3) < 1e-15);

    /* One saved logit cannot certify a two-token uniform distribution: the
     * unknown reference tail can carry half the mass. */
    const golden_logit incomplete[] = {{0, 0.0f}};
    const float uniform[] = {0.0f, 0.0f};
    CHECK(golden_distribution_distance(incomplete, 1, uniform, 2) == 1.0);

    /* The bound includes the tail instead of renormalizing it out of view. */
    const golden_logit head[] = {{0, logf(4.0f)}, {1, logf(2.0f)}};
    const float matching[] = {logf(4.0f), logf(2.0f), 0.0f};
    /* TV(candidate, normalized head) = 1/7; reference tail <= 2/8. */
    CHECK(fabs(golden_distribution_distance(head, 2, matching, 3) - (1.0 / 7.0 + 0.25)) < 1e-7);

    /* Corruption outside the saved slice must be visible even if greedy
     * output and every saved logit stay unchanged. */
    const golden_logit small_tail[] = {{0, 20.0f}, {1, 0.0f}, {2, -1.0f}};
    float outside[100];
    for (int i = 0; i < 100; i++) outside[i] = -100.0f;
    outside[0] = 20.0f; outside[1] = 0.0f; outside[2] = -1.0f;
    CHECK(golden_distribution_distance(small_tail, 3, outside, 100) < 1e-7);
    outside[99] = 19.0f;
    CHECK(golden_distribution_distance(small_tail, 3, outside, 100) > 0.26);

    const float nan_logits[] = {NAN, 1.0f, 2.0f};
    const float positive_infinity[] = {0.0f, INFINITY, 2.0f};
    const float negative_infinity[] = {0.0f, 1.0f, -INFINITY};
    CHECK(isinf(golden_distribution_distance(full, 3, nan_logits, 3)));
    CHECK(isinf(golden_distribution_distance(full, 3, positive_infinity, 3)));
    CHECK(isinf(golden_distribution_distance(full, 3, negative_infinity, 3)));
    const golden_logit duplicate[] = {{0, 2.0f}, {0, 1.0f}};
    const golden_logit unsorted[] = {{0, 1.0f}, {1, 2.0f}};
    const golden_logit invalid_id[] = {{3, 1.0f}};
    const golden_logit invalid_logit[] = {{1, NAN}};
    CHECK(isinf(golden_distribution_distance(duplicate, 2, offset, 3)));
    CHECK(isinf(golden_distribution_distance(unsorted, 2, offset, 3)));
    CHECK(isinf(golden_distribution_distance(invalid_id, 1, offset, 3)));
    CHECK(isinf(golden_distribution_distance(invalid_logit, 1, offset, 3)));
    CHECK(isinf(golden_distribution_distance(full, 0, offset, 3)));
    CHECK(isinf(golden_distribution_distance(full, 3, offset, 2)));
    puts("Golden distribution: analytical distances, rank swaps, missing tails and invalid values PASS");
    return 0;
}
