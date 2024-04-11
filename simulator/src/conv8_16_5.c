#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_IMAGE_HEIGHT 256
#define MAX_IMAGE_WIDTH 256

#define MAX_INPUT_CHANNELS 8
#define MAX_OUTPUT_CHANNELS 16

#define FILTER_SIZE 5
#define PAD 0
#define STRIDE 1

#define MAX_INPUT_SIZE (MAX_INPUT_CHANNELS * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH)
#define MAX_WEIGHT_SIZE (MAX_OUTPUT_CHANNELS * MAX_INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE)

#define KERNELOFFSETADDR (MAX_INPUT_SIZE * 4)
#define BIASOFFSETADDR (KERNELOFFSETADDR + (MAX_WEIGHT_SIZE * 4))
#define OUTPUTOFFSETADDR (BIASOFFSETADDR + (MAX_OUTPUT_CHANNELS * 4))

typedef float dataType;

// dataType inputImage[MAX_INPUT_CHANNELS][MAX_IMAGE_WIDTH][MAX_IMAGE_WIDTH] = {0};
// dataType convWeight[MAX_OUTPUT_CHANNELS][MAX_INPUT_CHANNELS][FILTER_SIZE][FILTER_SIZE] = {0};
// dataType convBias[MAX_OUTPUT_CHANNELS] = {0};
// dataType convOutput[MAX_OUTPUT_CHANNELS][MAX_IMAGE_WIDTH][MAX_IMAGE_WIDTH] = {0};

dataType inputImage[MAX_INPUT_CHANNELS * MAX_IMAGE_WIDTH * MAX_IMAGE_WIDTH] = {0};
dataType convWeight[MAX_OUTPUT_CHANNELS * MAX_INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE] = {0};
dataType convBias[MAX_OUTPUT_CHANNELS] = {0};
dataType convOutput[MAX_OUTPUT_CHANNELS * MAX_IMAGE_WIDTH * MAX_IMAGE_WIDTH] = {0};

void convolution(unsigned int inputChannels, unsigned int height, unsigned int width, unsigned int outputChannels,
                 unsigned int wout, unsigned int hout) {

    assert(inputChannels <= MAX_INPUT_CHANNELS);
    assert(height <= MAX_IMAGE_HEIGHT);
    assert(width <= MAX_IMAGE_WIDTH);
    assert(outputChannels <= MAX_OUTPUT_CHANNELS);

    for (int co = 0; co < outputChannels; co++) {
        for (int h = 0; h < hout; h++) {
            for (int w = 0; w < wout; w++) {

                dataType sum = 0;

                for (int cin = 0; cin < inputChannels; cin++) {
                    for (int i = h, m = 0; i < (h + FILTER_SIZE); i++, m++) {
                        for (int j = w, n = 0; j < (w + FILTER_SIZE); j++, n++) {
                            sum += inputImage[(cin * width * height) + (i * width) + (j)] *
                                   convWeight[(co * inputChannels * FILTER_SIZE * FILTER_SIZE) +
                                              (cin * FILTER_SIZE * FILTER_SIZE) + (m * FILTER_SIZE) + (n)];
                        }
                    }
                }
                convOutput[(co * wout * hout) + (h * wout) + (w)] = sum + convBias[co];
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

    if (argc < 7) {
        printf("Error: Arguments Should Be \n");
        printf("Error: streamingSetting, baseAddress, inputChannels, hight, weight, outputChannels\n");
        return -1;
    }
    const char *memoryFileName = argv[1];
    int streamingSetting = atoi(argv[2]);
    int baseAddress = atoi(argv[3]);
    int inputChannels = atoi(argv[4]);
    int height = atoi(argv[5]);
    int width = atoi(argv[6]);
    int outputChannels = atoi(argv[7]);

    printf("memoryFileName: %s\n", memoryFileName);
    printf("streamingSetting: %d\n", streamingSetting);
    printf("baseAddress: %d\n", baseAddress);
    printf("inputChannels: %d\n", inputChannels);
    printf("height: %d\n", height);
    printf("width: %d\n", width);
    printf("outputChannels: %d\n", outputChannels);

    int wout = (width + 2 * PAD - FILTER_SIZE) / STRIDE + 1;
    int hout = (height + 2 * PAD - FILTER_SIZE) / STRIDE + 1;

    readArrayFromMemory(memoryFileName, baseAddress, inputImage, inputChannels * height * width);

    readArrayFromMemory(memoryFileName, baseAddress + KERNELOFFSETADDR, convWeight,
                        outputChannels * inputChannels * FILTER_SIZE * FILTER_SIZE);

    readArrayFromMemory(memoryFileName, baseAddress + BIASOFFSETADDR, convBias, outputChannels);

    convolution(inputChannels, height, width, outputChannels, hout, wout);

    writeArrayToMemory(memoryFileName, baseAddress + OUTPUTOFFSETADDR, convOutput, outputChannels * hout * wout);

    return 0;
}
