/* Regression coverage for the non-terminating engine API used by embedded
 * hosts. Fatal model-loading failures must return through the checked API. */

#define DS4_TEST_HOOKS
#include "../ds4.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); \
        failures++; \
    } \
} while (0)

static void test_invalid_model_does_not_terminate_host(void) {
    fprintf(stderr, "RUN: invalid model does not terminate host\n");
    char directory[] = "/tmp/ds4-checked-open-XXXXXX";
    CHECK(mkdtemp(directory) != NULL, "create temporary directory");
    if (failures != 0) return;

    char model_path[512];
    char lock_path[512];
    snprintf(model_path, sizeof(model_path), "%s/partial.gguf", directory);
    snprintf(lock_path, sizeof(lock_path), "%s/ds4.lock", directory);

    FILE *model = fopen(model_path, "wb");
    CHECK(model != NULL, "create partial model");
    if (!model) return;
    const unsigned char invalid_gguf[64] = {0};
    CHECK(fwrite(invalid_gguf, sizeof(invalid_gguf), 1, model) == 1,
          "write partial model");
    CHECK(fclose(model) == 0, "close partial model");

    const pid_t child = fork();
    CHECK(child >= 0, "fork checked-open child");
    if (child == 0) {
        setenv("DS4_LOCK_FILE", lock_path, 1);
        ds4_engine_options options = {
            .model_path = model_path,
            .backend = DS4_BACKEND_CPU,
            .n_threads = 1,
        };
        ds4_engine *engine = NULL;
        char error[256] = {0};
        const int result = ds4_engine_open_checked(
                &engine, &options, error, sizeof(error));
        ds4_engine_close(engine);
        _exit(result != 0 && engine == NULL && error[0] != '\0' ? 0 : 2);
    }

    if (child > 0) {
        int status = 0;
        CHECK(waitpid(child, &status, 0) == child, "wait for checked-open child");
        CHECK(WIFEXITED(status), "checked-open child exits normally");
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "checked open returns an error instead of terminating its host");
    }

    unlink(model_path);
    unlink(lock_path);
    rmdir(directory);
}

static void test_failed_open_preserves_existing_lock(void) {
    fprintf(stderr, "RUN: failed open preserves existing lock\n");
    char directory[] = "/tmp/ds4-checked-lock-XXXXXX";
    CHECK(mkdtemp(directory) != NULL, "create lock test directory");
    if (failures != 0) return;

    char model_path[512];
    char lock_path[512];
    snprintf(model_path, sizeof(model_path), "%s/missing.gguf", directory);
    snprintf(lock_path, sizeof(lock_path), "%s/ds4.lock", directory);

    const pid_t child = fork();
    CHECK(child >= 0, "fork lock-ownership child");
    if (child == 0) {
        setenv("DS4_LOCK_FILE", lock_path, 1);
        const bool claimed = ds4_test_instance_lock_claim();
        ds4_engine_options options = {
            .model_path = model_path,
            .backend = DS4_BACKEND_CPU,
            .n_threads = 1,
        };
        ds4_engine *engine = NULL;
        char error[256] = {0};
        const int result = ds4_engine_open_checked(
                &engine, &options, error, sizeof(error));
        const bool original_lock_survived =
            ds4_test_instance_lock_is_held();
        ds4_engine_close(engine);
        ds4_test_instance_lock_release();
        _exit(claimed && result != 0 && engine == NULL &&
              original_lock_survived ? 0 : 3);
    }

    if (child > 0) {
        int status = 0;
        CHECK(waitpid(child, &status, 0) == child, "wait for lock-ownership child");
        CHECK(WIFEXITED(status), "lock-ownership child exits normally");
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "failed checked open preserves the existing engine lock");
    }

    unlink(lock_path);
    rmdir(directory);
}

/* New upstream model validators must honor the same failure boundary used
 * by checked open, including validators that print their own diagnostics. */
static void test_model_validation_does_not_terminate_host(void) {
    fprintf(stderr, "RUN: model validation does not terminate host\n");
    const pid_t child = fork();
    if (child == 0) {
        char error[256] = {0};
        ds4_failure_scope scope = {
            .error = error,
            .error_length = sizeof(error),
        };
        g_ds4_failure_scope = &scope;
        const int result = setjmp(scope.jump);
        if (result == 0) {
            config_expect_epsilon("rms_norm_eps", 0.0f, 1.0e-6f);
            _exit(2);
        }
        g_ds4_failure_scope = NULL;
        _exit(result == 1 && error[0] != '\0' ? 0 : 3);
    }
    int status = 0;
    const pid_t waited = child > 0 ? waitpid(child, &status, 0) : -1;
    CHECK(child > 0, "fork model-validation child");
    CHECK(waited == child, "wait for model-validation child");
    CHECK(WIFEXITED(status), "model-validation child exits normally");
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "model validation returns through the checked-open failure boundary");
}

int main(void) {
    test_invalid_model_does_not_terminate_host();
    test_failed_open_preserves_existing_lock();
    test_model_validation_does_not_terminate_host();
    if (failures != 0) {
        fprintf(stderr, "test_engine_checked_open: %d failure(s)\n", failures);
        return 1;
    }
    fprintf(stderr, "test_engine_checked_open: PASS\n");
    return 0;
}
