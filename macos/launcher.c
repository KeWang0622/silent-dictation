// Silent Dictation.app main executable.
//
// Embeds Python and runs the module silentdict.mac inside this process. Because the app itself runs
// the code, macOS shows "Silent Dictation" in the Dock and in the Camera, Input Monitoring and
// Accessibility permission prompts.
//
// Code, models and the Python environment live in the runtime folder that bin/build-app keeps in sync:
//   ~/Library/Application Support/Silent Dictation/runtime   (override with SILENTDICT_RUNTIME)
// Output goes to ~/Library/Application Support/Silent Dictation/data/app.log (SILENTDICT_DATA overrides).
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void alert(const char *message) {
    char cmd[4096];
    snprintf(cmd, sizeof cmd,
             "/usr/bin/osascript -e 'display alert \"Silent Dictation can't start\" message \"%s\" as critical' "
             ">/dev/null 2>&1",
             message);
    system(cmd);
}

static int mkdirs(const char *path) {
    char tmp[PATH_MAX];
    strlcpy(tmp, path, sizeof tmp);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    return (mkdir(tmp, 0755) != 0 && errno != EEXIST) ? -1 : 0;
}

static int is_file(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static void redirect_output(const char *data_dir) {
    char log_path[PATH_MAX], old_path[PATH_MAX];
    if (mkdirs(data_dir) != 0) return;
    snprintf(log_path, sizeof log_path, "%s/app.log", data_dir);
    struct stat st;
    if (stat(log_path, &st) == 0 && st.st_size > 2 * 1024 * 1024) {  // keep the log small
        snprintf(old_path, sizeof old_path, "%s/app.log.1", data_dir);
        rename(log_path, old_path);
    }
    int fd = open(log_path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) return;
    dup2(fd, STDOUT_FILENO);
    dup2(fd, STDERR_FILENO);
    close(fd);
}

static void check(PyStatus status) {
    if (PyStatus_Exception(status)) {
        fprintf(stderr, "Python failed to start: %s\n", status.err_msg ? status.err_msg : "unknown error");
        alert("Python failed to start. See app.log in ~/Library/Application Support/Silent Dictation/data.");
        Py_ExitStatusException(status);
    }
}

int main(int argc, char **argv) {
    const char *home = getenv("HOME");
    if (!home) return 1;
    char support[PATH_MAX], runtime[PATH_MAX], data[PATH_MAX], python[PATH_MAX], package[PATH_MAX];
    snprintf(support, sizeof support, "%s/Library/Application Support/Silent Dictation", home);
    if (getenv("SILENTDICT_RUNTIME")) strlcpy(runtime, getenv("SILENTDICT_RUNTIME"), sizeof runtime);
    else snprintf(runtime, sizeof runtime, "%s/runtime", support);
    if (getenv("SILENTDICT_DATA")) strlcpy(data, getenv("SILENTDICT_DATA"), sizeof data);
    else snprintf(data, sizeof data, "%s/data", support);
    snprintf(python, sizeof python, "%s/.venv/bin/python", runtime);
    snprintf(package, sizeof package, "%s/silentdict/__init__.py", runtime);

    redirect_output(data);
    if (!is_file(python) || !is_file(package)) {
        fprintf(stderr, "runtime missing: %s\n", runtime);
        alert("Its runtime is missing. In the Silent Dictation source folder run ./setup.sh, then bin/build-app.");
        return 1;
    }

    char exe[PATH_MAX], bundle[PATH_MAX];
    uint32_t size = sizeof exe;
    if (_NSGetExecutablePath(exe, &size) == 0 && realpath(exe, bundle)) {
        for (int i = 0; i < 3; i++) {  // .../Silent Dictation.app/Contents/MacOS/Silent Dictation
            char *slash = strrchr(bundle, '/');
            if (slash) *slash = '\0';
        }
        setenv("SILENTDICT_APP_BUNDLE", bundle, 1);
    }
    setenv("PYTHONPATH", runtime, 1);
    setenv("PYTHONUNBUFFERED", "1", 1);
    setenv("PYTHONDONTWRITEBYTECODE", "1", 0);
    unsetenv("PYTHONHOME");

    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    check(PyConfig_SetBytesString(&config, &config.executable, python));
    check(PyConfig_SetBytesString(&config, &config.program_name, python));
    check(PyConfig_SetBytesString(&config, &config.run_module, "silentdict.mac"));
    config.parse_argv = 0;
    char **args = calloc((size_t)argc + 1, sizeof(char *));
    int n = 0;
    args[n++] = python;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "-psn_", 5) != 0) args[n++] = argv[i];  // old Finder process serial numbers
    }
    check(PyConfig_SetBytesArgv(&config, n, args));
    check(Py_InitializeFromConfig(&config));
    PyConfig_Clear(&config);
    free(args);
    return Py_RunMain();
}
