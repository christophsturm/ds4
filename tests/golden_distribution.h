#ifndef DS4_GOLDEN_DISTRIBUTION_H
#define DS4_GOLDEN_DISTRIBUTION_H

/* A saved top-logit slice, sorted from greatest to least logit. */
typedef struct {
    int id;
    float logit;
} golden_logit;

/* Upper-bounds total variation from any full softmax distribution consistent
 * with the saved slice. Unrecorded logits cannot exceed its final logit.
 * Returns infinity for malformed references or non-finite candidate logits. */
double golden_distribution_distance(const golden_logit *reference, int count,
                                    const float *candidate, int vocab);

#endif
