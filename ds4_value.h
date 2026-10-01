#ifndef DS4_VALUE_H
#define DS4_VALUE_H

#include <stdbool.h>
#include <stddef.h>

/* A borrowed, typed tree at the embedded API boundary. Object children carry
 * their member names in key/key_length; array children leave those fields zero.
 * Text and keys are UTF-8 byte spans and may contain NUL. Numbers must be finite.
 * Keep the complete tree alive for the consuming call or callback duration.
 * Containers may nest at most 128 levels. No ownership crosses this boundary. */
typedef enum {
    DS4_VALUE_NULL, DS4_VALUE_BOOL, DS4_VALUE_NUMBER, DS4_VALUE_STRING,
    DS4_VALUE_ARRAY, DS4_VALUE_OBJECT,
} ds4_value_kind;

typedef struct ds4_value {
    ds4_value_kind kind;
    const char *key;
    size_t key_length;
    const char *text;
    size_t text_length;
    double number;
    bool boolean;
    const struct ds4_value *children;
    size_t count;
} ds4_value;

#endif
