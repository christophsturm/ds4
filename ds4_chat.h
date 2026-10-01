#ifndef DS4_CHAT_H
#define DS4_CHAT_H

#include "ds4.h"
#include "ds4_events.h"
#include "ds4_value.h"

/* Owns DS4's native message history, ordered tool schemas and encoded image bytes.
 * Every appended value is copied. A chat belongs to one caller at a time. */
typedef struct ds4_chat ds4_chat;

/* Starts an empty history; release it with ds4_chat_free. */
ds4_chat *ds4_chat_create(void);
/* Releases messages, schemas and image buffers. Accepts NULL. */
void ds4_chat_free(ds4_chat *chat);
/* Appends a role/content/reasoning message and returns its index, or -1 for
 * invalid arguments. Tool results supply their originating tool_call_id. */
int ds4_chat_add_message(ds4_chat *chat, const char *role, const char *content,
                         const char *reasoning, const char *tool_call_id);
/* Appends text to an existing message, preserving image/text part order. */
bool ds4_chat_append_text(ds4_chat *chat, int message, const char *text);
/* Copies a replayed tool call. Arguments must be a typed object. */
bool ds4_chat_add_tool_call(ds4_chat *chat, int message, const char *id,
                           const char *name, const ds4_value *arguments);
/* Copies a complete function-schema object, including arbitrary schema keywords.
 * DwarfStar owns its conversion to each model template. */
bool ds4_chat_add_tool(ds4_chat *chat, const ds4_value *schema);
/* Appends PNG/JPEG bytes at the current end of a user message. */
bool ds4_chat_add_image(ds4_chat *chat, int message, const char *media_type,
                       const uint8_t *bytes, size_t length);
/* Renders the same model-specific prompt used for generation. The caller frees
 * the returned string. A NULL engine selects DeepSeek V4 syntax. */
char *ds4_chat_render(const ds4_chat *chat, ds4_engine *engine,
                      ds4_think_mode thinking, bool tools_enabled);

#endif
