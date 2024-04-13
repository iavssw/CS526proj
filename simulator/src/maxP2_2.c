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

#define POOL_SIZE 2
#define STRIDE 2

#define MAX_HOUT ((MAX_IMAGE_HEIGHT - POOL_SIZE) / STRIDE + 1)
#define MAX_WOUT ((MAX_IMAGE_WIDTH - POOL_SIZE) / STRIDE + 1)

#define MAX_INPUT_SIZE (MAX_INPUT_CHANNELS * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH)

#define OUTPUTOFFSETADDR (MAX_INPUT_SIZE * 4)

typedef float dataType;

dataType inputImage[MAX_INPUT_CHANNELS * MAX_IMAGE_WIDTH * MAX_IMAGE_WIDTH] = {0};
dataType outputImage[MAX_INPUT_CHANNELS * MAX_HOUT * MAX_WOUT] = {0};

void maxPoolingRelu(int inputChannels, int height, int width, int pool_height, int pool_width, int stride) {

    int hout = (height - pool_height) / stride + 1;
    int wout = (width - pool_width) / stride + 1;

    for (int cin = 0; cin < inputChannels; cin++) {
        for (int h = 0; h < hout; h++) {
            for (int w = 0; w < wout; w++) {
                float max_value = -1000000000000.0;
                for (int i = h * 2; i < h * 2 + 2; i++) {
                    for (int j = w * 2; j < w * 2 + 2; j++) {
                        if (max_value < inputImage[(cin * height * width) + (i * width) + (j)]) {
                            max_value = inputImage[(cin * height * width) + (i * width) + (j)];
                        }
                    }
                }

                // relu
                max_value = max_value > 0 ? max_value : 0;

                if (cin == 0)
                    printf("max_value [%d, %d, %d]: %f\n", cin, h, w, max_value);

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

    if (argc < 6) {
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

    printf("memoryFileName: %s\n", memoryFileName);
    printf("streamingSetting: %d\n", streamingSetting);
    printf("baseAddress: %d\n", baseAddress);
    printf("inputChannels: %d\n", inputChannels);
    printf("height: %d\n", height);
    printf("width: %d\n", width);

    int hout = (height - POOL_SIZE) / STRIDE + 1;
    int wout = (width - POOL_SIZE) / STRIDE + 1;

    readArrayFromMemory(memoryFileName, baseAddress, inputImage, inputChannels * height * width);

    maxPoolingRelu(inputChannels, height, width, POOL_SIZE, POOL_SIZE, STRIDE);

    writeArrayToMemory(memoryFileName, baseAddress + OUTPUTOFFSETADDR, outputImage, inputChannels * hout * wout);

    return 0;
}
