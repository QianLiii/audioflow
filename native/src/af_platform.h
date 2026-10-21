#ifndef AF_PLATFORM_H
#define AF_PLATFORM_H

#ifndef _WIN32
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include <stdint.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <errno.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#endif

#ifdef _MSC_VER
#define AF_THREAD_LOCAL __declspec(thread)
#else
#define AF_THREAD_LOCAL _Thread_local
#endif

#ifdef _WIN32

typedef CRITICAL_SECTION af_mutex;
typedef CONDITION_VARIABLE af_cond;

typedef struct af_thread {
    HANDLE handle;
} af_thread;

typedef void (*af_thread_fn)(void *arg);

static inline int af_mutex_init(af_mutex *m)
{
    InitializeCriticalSection(m);
    return 0;
}

static inline void af_mutex_destroy(af_mutex *m)
{
    DeleteCriticalSection(m);
}

static inline void af_mutex_lock(af_mutex *m)
{
    EnterCriticalSection(m);
}

static inline void af_mutex_unlock(af_mutex *m)
{
    LeaveCriticalSection(m);
}

static inline int af_cond_init(af_cond *c)
{
    InitializeConditionVariable(c);
    return 0;
}

static inline void af_cond_destroy(af_cond *c)
{
    (void)c;
}

static inline void af_cond_signal(af_cond *c)
{
    WakeConditionVariable(c);
}

static inline void af_cond_broadcast(af_cond *c)
{
    WakeAllConditionVariable(c);
}

/* Returns 0 when signalled, -1 on timeout, -2 on error. */
static inline int af_cond_wait_ms(af_cond *c, af_mutex *m, uint32_t ms)
{
    if (SleepConditionVariableCS(c, m, ms))
        return 0;
    return GetLastError() == ERROR_TIMEOUT ? -1 : -2;
}

typedef struct af_thread_start {
    af_thread_fn fn;
    void *arg;
} af_thread_start;

static unsigned __stdcall af_thread_trampoline(void *p)
{
    af_thread_start *s = (af_thread_start *)p;
    af_thread_fn fn = s->fn;
    void *arg = s->arg;
    free(s);
    fn(arg);
    return 0;
}

static inline int af_thread_create(af_thread *t, af_thread_fn fn, void *arg)
{
    af_thread_start *s = (af_thread_start *)malloc(sizeof(*s));
    if (!s)
        return -1;
    s->fn = fn;
    s->arg = arg;
    t->handle = (HANDLE)_beginthreadex(NULL, 0, af_thread_trampoline, s, 0, NULL);
    if (!t->handle) {
        free(s);
        return -1;
    }
    return 0;
}

static inline void af_thread_join(af_thread *t)
{
    if (t->handle) {
        WaitForSingleObject(t->handle, INFINITE);
        CloseHandle(t->handle);
        t->handle = NULL;
    }
}

static inline void af_sleep_ms(uint32_t ms)
{
    Sleep(ms);
}

static inline uint64_t af_now_qpc(void)
{
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    if (frequency.QuadPart <= 0)
        return 0;
    return (uint64_t)((double)counter.QuadPart * 10000000.0 /
                      (double)frequency.QuadPart);
}

#else /* POSIX, for portable tests */

typedef pthread_mutex_t af_mutex;
typedef pthread_cond_t af_cond;

typedef struct af_thread {
    pthread_t handle;
} af_thread;

typedef void (*af_thread_fn)(void *arg);

static inline int af_mutex_init(af_mutex *m)
{
    return pthread_mutex_init(m, NULL);
}

static inline void af_mutex_destroy(af_mutex *m)
{
    pthread_mutex_destroy(m);
}

static inline void af_mutex_lock(af_mutex *m)
{
    pthread_mutex_lock(m);
}

static inline void af_mutex_unlock(af_mutex *m)
{
    pthread_mutex_unlock(m);
}

static inline int af_cond_init(af_cond *c)
{
    return pthread_cond_init(c, NULL);
}

static inline void af_cond_destroy(af_cond *c)
{
    pthread_cond_destroy(c);
}

static inline void af_cond_signal(af_cond *c)
{
    pthread_cond_signal(c);
}

static inline void af_cond_broadcast(af_cond *c)
{
    pthread_cond_broadcast(c);
}

/* Returns 0 when signalled, -1 on timeout, -2 on error. */
static inline int af_cond_wait_ms(af_cond *c, af_mutex *m, uint32_t ms)
{
    struct timespec ts;
    int rc;

    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += (time_t)(ms / 1000u);
    ts.tv_nsec += (long)(ms % 1000u) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1000000000L;
    }

    rc = pthread_cond_timedwait(c, m, &ts);
    if (rc == 0)
        return 0;
    return rc == ETIMEDOUT ? -1 : -2;
}

typedef struct af_thread_start {
    af_thread_fn fn;
    void *arg;
} af_thread_start;

static inline void *af_thread_trampoline(void *p)
{
    af_thread_start *s = (af_thread_start *)p;
    af_thread_fn fn = s->fn;
    void *arg = s->arg;
    free(s);
    fn(arg);
    return NULL;
}

static inline int af_thread_create(af_thread *t, af_thread_fn fn, void *arg)
{
    af_thread_start *s = (af_thread_start *)malloc(sizeof(*s));
    if (!s)
        return -1;
    s->fn = fn;
    s->arg = arg;
    if (pthread_create(&t->handle, NULL, af_thread_trampoline, s) != 0) {
        free(s);
        return -1;
    }
    return 0;
}

static inline void af_thread_join(af_thread *t)
{
    pthread_join(t->handle, NULL);
}

static inline void af_sleep_ms(uint32_t ms)
{
    struct timespec ts;

    ts.tv_sec = (time_t)(ms / 1000u);
    ts.tv_nsec = (long)(ms % 1000u) * 1000000L;
    while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {
    }
}

static inline uint64_t af_now_qpc(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 10000000ull +
           (uint64_t)ts.tv_nsec / 100ull;
}

#endif /* _WIN32 */

#endif /* AF_PLATFORM_H */
