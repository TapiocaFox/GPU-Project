#pragma once

#include <string>
#include <stdio.h>
#include <string.h>
#include <setjmp.h>
#include <iostream>
#include "jpeglib.h"

/**
 * Provides a convenient to read/write JPEG files using libjpeg
 */
struct JpegInfo {
    char *buffer;
    int width;
    int height;
};

class JpegLoader {
public:
    JpegInfo* load(std::string filename);
    bool save(std::string filename, char *imageBuffer, int width, int height, int quality);
};

