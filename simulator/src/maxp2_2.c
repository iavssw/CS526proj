#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef float dataType;

#define MAX_STREAM_SIZE 1 * 1024 * 1024 // 1MB
#define MAX_STREAM_ELEMENTS MAX_STREAM_SIZE / sizeof(dataType)

#define MAX_IMAGE_HEIGHT 256
#define MAX_IMAGE_WIDTH 256

#define MAX_INPUT_CHANNELS 32

dataType inputImage[MAX_INPUT_CHANNELS * MAX_IMAGE_WIDTH * MAX_IMAGE_WIDTH] = {0};
dataType outputImage[MAX_INPUT_CHANNELS * MAX_IMAGE_WIDTH * MAX_IMAGE_WIDTH] = {0};

void maxPooling(int inputChannels, int height, int width, int pool_size, int stride) {

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
                // outputImage[(cin * hout * wout) + (h * wout) + (w)] = max_value > 0 ? max_value : 0;

                outputImage[(cin * hout * wout) + (h * wout) + (w)] = max_value;
            }
        }
    }
}

unsigned int readArrayFromStream(const char *mainMemoryFile, dataType *array) {
    FILE *file = fopen(mainMemoryFile, "r+");
    if (!file) {
        perror("Failed to open file");
        return 0;
    }

    // Move to the position where we want to start reading
    fseek(file, 0, SEEK_SET); // Each byte is 2 hex digits plus a newline

    unsigned int size;
    fscanf(file, "%08X\n", &size);
    // printf("size: %d\n", size);

    if (size > MAX_STREAM_ELEMENTS) {
        printf("Error: Size read from file exceeds the maximum array size.\n");
        fclose(file);
        return 0;
    }

    for (int i = 0; i < size; i++) {
        // srcTile -> for later update
        unsigned int srcTile;
        fscanf(file, "%02X\n", &srcTile);
        // printf("srcTile: %d\n", srcTile);

        // data
        unsigned char bytes[4];
        for (int j = 0; j < 4; j++) {
            unsigned int byte;
            fscanf(file, "%02X\n", &byte);
            bytes[j] = (unsigned char)byte;
        }

        array[i] = *(dataType *)bytes;
    }

    // reset stream size to 0
    unsigned int emptyStream = 0;
    fseek(file, 0, SEEK_SET);
    fprintf(file, "%08X\n", emptyStream);
    fflush(file);

    fclose(file);

    return size;
}

void writeArrayToStream(const char *mainMemoryFile, unsigned char srcTile, dataType *array, unsigned int size) {
    FILE *file = fopen(mainMemoryFile, "r+");
    if (!file) {
        perror("Failed to open file");
        return;
    }

    // Move to start where the lengh metadata is stored
    unsigned int oldStreamSize;
    fseek(file, 0, SEEK_SET); // Each byte is 2 hex digits plus a newline
    fscanf(file, "%08X\n", &oldStreamSize);
    // printf("oldStreamSize: %d\n", oldStreamSize);

    // write the new size
    unsigned int newStreamSize = oldStreamSize + size;

    fseek(file, 0, SEEK_SET);
    fprintf(file, "%08X\n", newStreamSize);

    // Move to the end of the stream 15 = 3 (two hex digits + newline) * 5 (srcTile + 4 data bytes for float)
    fseek(file, oldStreamSize * 15 + 9, SEEK_SET);

    for (int i = 0; i < size; i++) {
        unsigned char *bytes = (unsigned char *)&array[i];
        fprintf(file, "%02X\n", srcTile); // src location encoded into each data packet
        for (int j = 0; j < 4; j++) {     // Assuming dataType is 4 bytes (e.g., float)
            fprintf(file, "%02X\n", bytes[j]);
        }
    }

    fclose(file);
    return;
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

    if (argc < 12) {
        printf("Error: Arguments Should Be \n");
        printf("Error: streamingSetting, baseAddress, inputChannels, hight, weight, outputChannels\n");
        return -1;
    }
    const char *memoryFileName = argv[1];
    const char *streamInput = argv[2];
    const char *streamDest = argv[3];
    int streamingSetting = atoi(argv[4]);
    int inputAddress = atoi(argv[5]);
    int outputAddress = atoi(argv[6]);
    int inputChannels = atoi(argv[7]);
    int height = atoi(argv[8]);
    int width = atoi(argv[9]);
    int poolSize = atoi(argv[10]);
    int stride = atoi(argv[11]);

    // ASCII value of 0 is 48
    int tileNumber = (int)streamInput[strlen(streamInput) - 1] - 48;

    printf("MaxPoolRelu\n");
    printf("memoryFileName: %s\n", memoryFileName);
    printf("tileNumber: %d\n", tileNumber);
    printf("streamInput: %s\n", streamInput);
    printf("streamDest: %s\n", streamDest);
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

    // streamingSetting
    //  00 -> read write data from memory
    //  01 -> read data from memory and write data to stream
    //  10 -> read data from stream and write data to memory
    //  11 -> read write data from stream
    int readStream = streamingSetting / 10;
    int writeStream = streamingSetting % 10;

    if (readStream == 0) { // memory
        readArrayFromMemory(memoryFileName, inputAddress, inputImage, inputChannels * height * width);
    } else { // stream
        // readArrayFromMemory(streamInput, 0, inputImage, inputChannels * height * width);
        readArrayFromStream(streamInput, inputImage);
    }

    maxPooling(inputChannels, height, width, poolSize, stride);

    writeArrayToMemory(memoryFileName, outputAddress, outputImage, inputChannels * hout * wout);

    // output
    if (writeStream == 0) { // memory
        writeArrayToMemory(memoryFileName, outputAddress, outputImage, inputChannels * hout * wout);
    } else { // stream
        // writeArrayToMemory(streamDest, 0, outputImage, inputChannels * hout * wout);
        writeArrayToStream(streamDest, (unsigned char)tileNumber, outputImage, inputChannels * hout * wout);
    }

    return 0;
}
