#ifndef DS4_EVENTS_H
#define DS4_EVENTS_H

#include <stdbool.h>
#include <stddef.h>

/* Semantic output from the native chat pipeline, before HTTP/SSE encoding. */
typedef enum {
    DS4_CHAT_CONTENT,
    DS4_CHAT_REASONING,
    DS4_CHAT_REASONING_STARTED,
    DS4_CHAT_REASONING_COMPLETED,
    DS4_CHAT_TOOL_START,
    DS4_CHAT_TOOL_ARGUMENTS,
    DS4_CHAT_FINISH,
    DS4_CHAT_USAGE,
    DS4_CHAT_PREFILL,
    DS4_CHAT_ERROR,
    DS4_CHAT_DONE,
} ds4_chat_event_kind;

/* All strings and bytes are borrowed for the callback's duration. Text length
 * is explicit; tool arguments remain DS4's native JSON fragments. Unused
 * fields are zero. Finish text is the native stop/length/tool_calls/error
 * reason; error text is an actionable diagnostic. */
typedef struct {
    ds4_chat_event_kind kind;
    const char *text;
    size_t text_length;
    int tool_index;
    const char *tool_id;
    const char *tool_name;
    int prompt_tokens;
    int completion_tokens;
    int cache_read_tokens;
    int cache_write_tokens;
    int current;
    int total;
    int error_status;
} ds4_chat_event;

/* Return false to stop delivery and cancel the native generation. Callbacks
 * run synchronously on the generation thread and must copy retained values. */
typedef bool (*ds4_chat_event_callback)(void *context, const ds4_chat_event *event);

#endif
