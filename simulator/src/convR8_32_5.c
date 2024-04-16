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

#define MAX_INPUT_CHANNELS 16
#define MAX_OUTPUT_CHANNELS 128

#define FILTER_SIZE 5
#define PAD 0
#define STRIDE 1

dataType inputImage[MAX_INPUT_CHANNELS * MAX_IMAGE_WIDTH * MAX_IMAGE_WIDTH] = {0};
dataType convWeight[MAX_OUTPUT_CHANNELS * MAX_INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE] = {0};
dataType convBias[MAX_OUTPUT_CHANNELS] = {0};
dataType convOutput[MAX_OUTPUT_CHANNELS * MAX_IMAGE_WIDTH * MAX_IMAGE_WIDTH] = {0};

void convolution(unsigned int inputChannels, unsigned int height, unsigned int width, unsigned int outputChannels,
                 unsigned int wout, unsigned int hout) {

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
                // convOutput[(co * wout * hout) + (h * wout) + (w)] = sum + convBias[co];

                // relu
                dataType sumBias = sum + convBias[co];
                convOutput[(co * wout * hout) + (h * wout) + (w)] = sumBias > 0 ? sumBias : 0;
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

    int extraDest = 0;

    int numberOfMinArguments = 13; // + 1
    if (argc < numberOfMinArguments) {
        printf("Error: Arguments not Correct see Source Code\n");
        return -1;
    } else if (argc > numberOfMinArguments) {
        extraDest = argc - numberOfMinArguments;
        printf("Extra Dest: %d\n", extraDest);
    }
    // plus 1 for mandatory streamDest
    const char *streamDest[extraDest + 1];

    const char *memoryFileName = argv[1];
    const char *streamInput = argv[2];
    // read variable number of streamDest
    for (int i = 0; i <= extraDest; i++) {
        streamDest[i] = argv[3 + i];
    }
    int streamingSetting = atoi(argv[extraDest + 4]);
    int inputAddress = atoi(argv[extraDest + 5]);
    int convWeightAddress = atoi(argv[extraDest + 6]);
    int convBiasAddress = atoi(argv[extraDest + 7]);
    int outputAddress = atoi(argv[extraDest + 8]);
    int inputChannels = atoi(argv[extraDest + 9]);
    int height = atoi(argv[extraDest + 10]);
    int width = atoi(argv[extraDest + 11]);
    int outputChannels = atoi(argv[extraDest + 12]);

    // ASCII value of 0 is 48
    int tileNumber = (int)streamInput[strlen(streamInput) - 1] - 48;

    printf("Conv8_16_5\n");
    printf("memoryFileName: %s\n", memoryFileName);
    printf("tileNumber: %d\n", tileNumber);
    printf("streamInput: %s\n", streamInput);
    // print variable number of streamDest
    for (int i = 0; i <= extraDest; i++) {
        printf("streamDest: %s\n", streamDest[i]);
    }
    printf("streamingSetting: %d\n", streamingSetting);
    printf("inputAddress: %d\n", inputAddress);
    printf("convWeightAddress: %d\n", convWeightAddress);
    printf("convBiasAddress: %d\n", convBiasAddress);
    printf("outputAddress: %d\n", outputAddress);
    printf("inputChannels: %d\n", inputChannels);
    printf("height: %d\n", height);
    printf("width: %d\n", width);
    printf("outputChannels: %d\n", outputChannels);

    assert(inputChannels <= MAX_INPUT_CHANNELS);
    assert(height <= MAX_IMAGE_HEIGHT);
    assert(width <= MAX_IMAGE_WIDTH);
    assert(outputChannels <= MAX_OUTPUT_CHANNELS);

    int wout = (width + 2 * PAD - FILTER_SIZE) / STRIDE + 1;
    int hout = (height + 2 * PAD - FILTER_SIZE) / STRIDE + 1;

    // streamingSetting
    //  00 -> read write data from memory
    //  01 -> read data from memory and write data to stream
    //  10 -> read data from stream and write data to memory
    //  11 -> read write data from stream
    int readStream = streamingSetting / 10;
    int writeStream = streamingSetting % 10;

    // image
    if (readStream == 0) { // memory
        readArrayFromMemory(memoryFileName, inputAddress, inputImage, inputChannels * height * width);
    } else { // stream
        // readArrayFromMemory(streamInput, 0, inputImage, inputChannels * height * width);
        readArrayFromStream(streamInput, inputImage);
    }

    // weight
    readArrayFromMemory(memoryFileName, convWeightAddress, convWeight,
                        outputChannels * inputChannels * FILTER_SIZE * FILTER_SIZE);

    // bias
    readArrayFromMemory(memoryFileName, convBiasAddress, convBias, outputChannels);

    convolution(inputChannels, height, width, outputChannels, hout, wout);

    // output
    if (writeStream == 0) { // memory
        writeArrayToMemory(memoryFileName, outputAddress, convOutput, outputChannels * hout * wout);
    } else { // stream
        // writeArrayToMemory(streamDest, 0, convOutput, outputChannels * hout * wout);
        for (int i = 0; i <= extraDest; i++) {
            writeArrayToStream(streamDest[i], (unsigned char)tileNumber, convOutput, outputChannels * hout * wout);
        }
    }

    return 0;
}
