#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_IMAGE_HEIGHT 256
#define MAX_IMAGE_WIDTH 256

#define MAX_INPUT_CHANNELS 16

typedef float dataType;

dataType inputImage[MAX_INPUT_CHANNELS * MAX_IMAGE_WIDTH * MAX_IMAGE_WIDTH] = {0};
dataType outputImage[MAX_INPUT_CHANNELS * MAX_IMAGE_WIDTH * MAX_IMAGE_WIDTH] = {0};

void maxPoolingRelu(int inputChannels, int height, int width, int pool_size, int stride) {

    int hout = (height - pool_size) / stride + 1;
    int wout = (width - pool_size) / stride + 1;

    for (int cin = 0; cin < inputChannels; cin++) {
        for (int h = 0; h < hout; h++) {
            for (int w = 0; w < wout; w++) {
                float max_value = -1000000000000.0;
                for (int i = h * stride; i < h * stride + pool_size; i++) {
                    for (int j = w * stride; j < w * stride + pool_size; j++) {
                        if (max_value < inputImage[(cin * height * width) + (i * width) + (j)]) {
                            max_value = inputImage[(cin * height * width) + (i * width) + (j)];
                        }
                    }
                }

                // relu
                outputImage[(cin * hout * wout) + (h * wout) + (w)] = max_value > 0 ? max_value : 0;
            }
        }
    }
}

float readFloatFromMemory(const char *mainMemoryFile, int address) {
    FILE *file = fopen(mainMemoryFile, "r");
    if (!file) {
        perror("Failed to open file");
        return -1;
    }

    // Move to the position where we want to start reading
    fseek(file, address * 3, SEEK_SET); // Each byte is 2 hex digits plus a newline

    unsigned char bytes[4];
    for (int i = 0; i < 4; i++) {
        unsigned int byte;
        fscanf(file, "%02X\n", &byte);
        bytes[i] = (unsigned char)byte;
    }

    fclose(file);

    float *value = (float *)bytes;
    return *value;
}

void readArrayFromMemory(const char *mainMemoryFile, int address, dataType *array, int size) {
    FILE *file = fopen(mainMemoryFile, "r");
    if (!file) {
        perror("Failed to open file");
        return;
    }

    // Move to the position where we want to start reading
    fseek(file, address * 3, SEEK_SET); // Each byte is 2 hex digits plus a newline

    for (int i = 0; i < size; i++) {
        unsigned char bytes[4];
        for (int j = 0; j < 4; j++) {
            unsigned int byte;
            fscanf(file, "%02X\n", &byte);
            bytes[j] = (unsigned char)byte;
        }

        array[i] = *(dataType *)bytes;
    }

    fclose(file);
}

void writeArrayToMemory(const char *mainMemoryFile, int address, dataType *array, int size) {
    FILE *file = fopen(mainMemoryFile, "r+");
    if (!file) {
        perror("Failed to open file");
        return;
    }

    // Move to the position where we want to start writing
    fseek(file, address * 3, SEEK_SET); // Each byte is 2 hex digits plus a newline

    for (int i = 0; i < size; i++) {
        unsigned char *bytes = (unsigned char *)&array[i];
        for (int j = 0; j < 4; j++) { // Assuming dataType is 4 bytes (e.g., float)
            fprintf(file, "%02X\n", bytes[j]);
        }
    }

    fclose(file);
}

int main(int argc, char **argv) {

    // const char *memoryFileName = "memory/memory.txt";

    if (argc < 11) {
        printf("Error: Arguments Should Be \n");
        printf("Error: streamingSetting, baseAddress, inputChannels, hight, weight, outputChannels\n");
        return -1;
    }
    const char *memoryFileName = argv[1];
    const char *streamFileName = argv[2];
    int streamingSetting = atoi(argv[3]);
    int inputAddress = atoi(argv[4]);
    int outputAddress = atoi(argv[5]);
    int inputChannels = atoi(argv[6]);
    int height = atoi(argv[7]);
    int width = atoi(argv[8]);
    int poolSize = atoi(argv[9]);
    int stride = atoi(argv[10]);

    // ASCII value of 0 is 48
    int tileNumber = (int)streamFileName[strlen(streamFileName) - 1] - 48;

    printf("MaxPoolRelu\n");
    printf("memoryFileName: %s\n", memoryFileName);
    printf("tileNumber: %d\n", tileNumber);
    printf("streamFileName: %s\n", streamFileName);
    printf("streamingSetting: %d\n", streamingSetting);
    printf("inputAddress: %d\n", inputAddress);
    printf("outputAddress: %d\n", outputAddress);
    printf("inputChannels: %d\n", inputChannels);
    printf("height: %d\n", height);
    printf("width: %d\n", width);
    printf("poolSize: %d\n", poolSize);
    printf("stride: %d\n", stride);

    assert(inputChannels <= MAX_INPUT_CHANNELS);
    assert(height <= MAX_IMAGE_HEIGHT);
    assert(width <= MAX_IMAGE_WIDTH);

    int hout = (height - poolSize) / stride + 1;
    int wout = (width - poolSize) / stride + 1;

    readArrayFromMemory(memoryFileName, inputAddress, inputImage, inputChannels * height * width);

    maxPoolingRelu(inputChannels, height, width, poolSize, stride);

    writeArrayToMemory(memoryFileName, outputAddress, outputImage, inputChannels * hout * wout);

    return 0;
}
