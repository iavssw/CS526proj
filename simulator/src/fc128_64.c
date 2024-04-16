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

#define MAX_INPUT_CHANNELS 128
#define MAX_OUTPUT_CHANNELS 64

dataType inputImage[MAX_INPUT_CHANNELS] = {0};
dataType fcWeight[MAX_OUTPUT_CHANNELS * MAX_INPUT_CHANNELS] = {0};
dataType fcBias[MAX_OUTPUT_CHANNELS] = {0};
dataType fcOutput[MAX_OUTPUT_CHANNELS] = {0};

void fullyConnected(unsigned int inputChannels, unsigned int outputChannels) {

    for (int co = 0; co < outputChannels; co++) {
        dataType sum = 0;
        for (int cin = 0; cin < inputChannels; cin++) {
            sum += inputImage[cin] * fcWeight[(co * inputChannels) + cin];
        }
        dataType sumBias = sum + fcBias[co];
        fcOutput[co] = sumBias > 0 ? sumBias : 0;
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

    int numberOfMinArguments = 11; // + 1
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
    int fcWeightAddress = atoi(argv[extraDest + 6]);
    int fcBiasAddress = atoi(argv[extraDest + 7]);
    int outputAddress = atoi(argv[extraDest + 8]);
    int inputChannels = atoi(argv[extraDest + 9]);
    int outputChannels = atoi(argv[extraDest + 10]);

    // ASCII value of 0 is 48
    int tileNumber = (int)streamInput[strlen(streamInput) - 1] - 48;

    printf("FullConnected\n");
    printf("memoryFileName: %s\n", memoryFileName);
    printf("tileNumber: %d\n", tileNumber);
    printf("streamInput: %s\n", streamInput);
    // print variable number of streamDest
    for (int i = 0; i <= extraDest; i++) {
        printf("streamDest: %s\n", streamDest[i]);
    }
    printf("streamingSetting: %d\n", streamingSetting);
    printf("inputAddress: %d\n", inputAddress);
    printf("fcWeightAddress: %d\n", fcWeightAddress);
    printf("fcBiasAddress: %d\n", fcBiasAddress);
    printf("outputAddress: %d\n", outputAddress);
    printf("inputChannels: %d\n", inputChannels);
    printf("outputChannels: %d\n", outputChannels);

    assert(inputChannels <= MAX_INPUT_CHANNELS);
    assert(outputChannels <= MAX_OUTPUT_CHANNELS);

    // streamingSetting
    //  00 -> read write data from memory
    //  01 -> read data from memory and write data to stream
    //  10 -> read data from stream and write data to memory
    //  11 -> read write data from stream
    int readStream = streamingSetting / 10;
    int writeStream = streamingSetting % 10;

    // image
    if (readStream == 0) { // memory
        readArrayFromMemory(memoryFileName, inputAddress, inputImage, inputChannels);
    } else { // stream
        // readArrayFromMemory(streamInput, 0, inputImage, inputChannels * height * width);
        readArrayFromStream(streamInput, inputImage);
    }

    // weight
    readArrayFromMemory(memoryFileName, fcWeightAddress, fcWeight, outputChannels * inputChannels);

    // bias
    readArrayFromMemory(memoryFileName, fcBiasAddress, fcBias, outputChannels);

    fullyConnected(inputChannels, outputChannels);

    // output
    if (writeStream == 0) { // memory
        writeArrayToMemory(memoryFileName, outputAddress, fcOutput, outputChannels);
    } else { // stream
        // writeArrayToMemory(streamDest, 0, convOutput, outputChannels * hout * wout);
        for (int i = 0; i <= extraDest; i++) {
            writeArrayToStream(streamDest[i], (unsigned char)tileNumber, fcOutput, outputChannels);
        }
    }

    return 0;
}
