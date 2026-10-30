// Silent Dictation.app main executable.
//
// A tiny native launcher: macOS attributes Camera / Input Monitoring / Accessibility permissions
// to the app that launched a process, so spawning the Python app from here makes the permission
// prompts and System Settings entries say "Silent Dictation" instead of "python3".
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
static pid_t child = 0;

static void forward(int sig) {
    if (child > 0) kill(child, sig);
}

static void alert(const char *message) {
    char cmd[2048];
    snprintf(cmd, sizeof cmd,
             "/usr/bin/osascript -e 'display alert \"Silent Dictation\" message \"%s\"' >/dev/null 2>&1",
             message);
    system(cmd);
}

static void parent_dir(char *path) {
    char *slash = strrchr(path, '/');
    if (slash) *slash = '\0';
}

int main(void) {
    char exe[PATH_MAX], real[PATH_MAX];
    uint32_t size = sizeof exe;
    if (_NSGetExecutablePath(exe, &size) != 0 || !realpath(exe, real)) return 1;

    char contents[PATH_MAX];  // .../Silent Dictation.app/Contents
    strlcpy(contents, real, sizeof contents);
    parent_dir(contents);  // MacOS
    parent_dir(contents);  // Contents
    char bundle[PATH_MAX];
    strlcpy(bundle, contents, sizeof bundle);
    parent_dir(bundle);

    char root_file[PATH_MAX], root[PATH_MAX] = {0};
    snprintf(root_file, sizeof root_file, "%s/Resources/project_root", contents);
    FILE *f = fopen(root_file, "r");
    if (!f || !fgets(root, sizeof root, f)) {
        alert("Cannot find the project folder. Rebuild the app with bin/build-app.");
        return 1;
    }
    fclose(f);
    root[strcspn(root, "\r\n")] = '\0';

    char python[PATH_MAX], log_dir[PATH_MAX], log_path[PATH_MAX];
    snprintf(python, sizeof python, "%s/.venv/bin/python", root);
    if (access(python, X_OK) != 0) {
        alert("The Python environment is missing. Run ./setup.sh in the project folder, then open the app again.");
        return 1;
    }
    snprintf(log_dir, sizeof log_dir, "%s/data", root);
    mkdir(log_dir, 0755);
    snprintf(log_path, sizeof log_path, "%s/app.log", log_dir);

    setenv("PYTHONPATH", root, 1);
    setenv("PYTHONUNBUFFERED", "1", 1);
    setenv("SILENTDICT_APP_BUNDLE", bundle, 1);

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, 1, log_path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    posix_spawn_file_actions_adddup2(&actions, 1, 2);

    char *args[] = {python, "-m", "silentdict.macapp", NULL};
    signal(SIGTERM, forward);
    signal(SIGINT, forward);
    signal(SIGHUP, forward);
    if (posix_spawn(&child, python, &actions, NULL, args, environ) != 0) {
        alert("Could not start Silent Dictation. See data/app.log in the project folder.");
        return 1;
    }
    int status = 0;
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
