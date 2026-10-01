#include "golden_distribution.h"

#include <math.h>

/* Keep this calculation independent of inference's fast-math compilation. */
double golden_distribution_distance(const golden_logit *reference, int count,
                                    const float *candidate, int vocab) {
    if (!reference || !candidate || count <= 0 || count > vocab) return INFINITY;
    for (int i = 0; i < count; i++) {
        if (reference[i].id < 0 || reference[i].id >= vocab ||
            !isfinite(reference[i].logit) ||
            (i && reference[i].logit > reference[i - 1].logit)) return INFINITY;
        for (int j = 0; j < i; j++) {
            if (reference[i].id == reference[j].id) return INFINITY;
        }
    }
    double maximum = candidate[0];
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(candidate[i])) return INFINITY;
        if (candidate[i] > maximum) maximum = candidate[i];
    }
    double candidate_sum = 0.0, reference_sum = 0.0;
    for (int i = 0; i < vocab; i++) {
        candidate_sum += exp((double)candidate[i] - maximum);
    }
    for (int i = 0; i < count; i++) {
        reference_sum += exp((double)reference[i].logit - reference[0].logit);
    }
    double difference = 0.0, candidate_slice_mass = 0.0;
    for (int i = 0; i < count; i++) {
        const double p = exp((double)candidate[reference[i].id] - maximum) / candidate_sum;
        const double q = exp((double)reference[i].logit - reference[0].logit) / reference_sum;
        difference += fabs(p - q);
        candidate_slice_mass += p;
    }
    /* Compare against the saved slice normalized in isolation, then use the
     * triangle inequality to account for every possible unrecorded tail. */
    const double unknown_sum = (vocab - count) *
        exp((double)reference[count - 1].logit - reference[0].logit);
    const double tail_bound = unknown_sum / (reference_sum + unknown_sum);
    const double outside_mass = fmax(0.0, 1.0 - candidate_slice_mass);
    return fmin(1.0, 0.5 * (difference + outside_mass) + tail_bound);
}
