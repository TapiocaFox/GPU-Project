#pragma once

#include <sys/time.h>

/**
 * Provides a convenient way to measure elapsed time
 */
class Timer {
public:
    void   start();
    double getElapsedTimeInMilliSec();

private:
    timeval startTime;
};
