import struct
import subprocess
import numpy as np

from pylib import readbins
from pylib import memManage

MAX_IMAGE_HEIGHT = 256
MAX_IMAGE_WIDTH = 256

MAX_INPUT_CHANNELS = 8
MAX_OUTPUT_CHANNELS = 16

FILTER_SIZE = 5
PAD = 0
STRIDE = 1

# Calculate offsets based on the provided formulas
KERNELOFFSET = MAX_INPUT_CHANNELS * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH * 4
BIASOFFSET = KERNELOFFSET + (MAX_OUTPUT_CHANNELS * MAX_INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE * 4)
OUTPUTOFFSET = BIASOFFSET + (MAX_OUTPUT_CHANNELS * 4)

def convolution(inputImage, convWeight, convBias, inputChannels, height, width, outputChannels, filterSize, stride, pad):
    # Output dimensions, assuming no padding (PAD=0) and stride of 1
    hout = int((height + 2 * PAD - filterSize) / stride) + 1
    wout = int((width + 2 * PAD - filterSize) / stride) + 1

    # Initialize the output image with zeros
    convOutput = np.zeros((outputChannels, hout, wout), dtype=np.float32)

    for co in range(outputChannels):
        for h in range(hout):
            for w in range(wout):
                sum = 0.0
                for cin in range(inputChannels):
                    for i, m in zip(range(h, h + FILTER_SIZE), range(FILTER_SIZE)):
                        for j, n in zip(range(w, w + FILTER_SIZE), range(FILTER_SIZE)):
                            # Ensure indices are within bounds before accessing arrays
                            if i < height and j < width:
                                sum += inputImage[cin, i, j] * convWeight[co, cin, m, n]
                convOutput[co, h, w] = sum + convBias[co]

    return convOutput

def exeCommand(command):
    result = subprocess.run(command, capture_output=True, text=True)
    # Check if the command was executed successfully
    if result.returncode == 0:
        # Command executed successfully, print stdout
        print("Output:")
        print(result.stdout)
    else:
        # There was an error, print stderr
        print("Error:")
        print(result.stderr)

# Example usage
if __name__ == "__main__":
    
    sizeOfmainMemory = 8 * 1024 * 1024 # 8MB KByte
    mainMemoryFile = "memory/memory.txt"

    
    # Initialize a dummy image and kernels
    
    inputChannels = 1
    height = 32
    width = 32
    outputChannels = 6
    filterSize = 5
    stride = 1
    pad = 0
    
    baseAddress = 0
    
    h_out = int((height - filterSize + 2 * pad) / stride) + 1
    w_out = int((width - filterSize + 2 * pad) / stride) + 1
    
    # np.random.seed(42)
    # image = np.random.uniform(low=-10.0, high=10.0, size=(inputChannels, height, width))
    # kernels = np.random.uniform(low=-10.0, high=10.0, size=(outputChannels, inputChannels, filterSize, filterSize))
    # bias = np.random.uniform(low=-10.0, high=10.0, size=outputChannels)
    
    images, (num_images, rows, cols) = readbins.parse_mnist_images("data/images.bin")
    labels = readbins.parse_mnist_labels("data/labels.bin")
    conv1_weights, conv1_bias, conv3_weights, conv3_bias, conv5_weights, conv5_bias, fc6_weights, fc6_bias = readbins.parse_parameters("data/params.bin")
    
    image = readbins.get_image(images, 0)    
    
    #setupMemory
    if True:
    # if False:
        memManage.setupMemory(sizeOfmainMemory, mainMemoryFile)
        # Write the image to memory
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, image, inputChannels * height * width, 0)
        # Write the kernels to memory starting after max image size
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, conv1_weights, outputChannels * inputChannels * filterSize * filterSize, KERNELOFFSET)
        # Write the bias to memory starting after max kernel size
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, conv1_bias, outputChannels, BIASOFFSET)
    
    output_image = convolution(image, conv1_weights, conv1_bias, inputChannels, height, width, outputChannels, filterSize, stride, pad)
    
    exeCommand(["bins/conv8_16_5", "memory/memory.txt", "4", str(baseAddress), str(inputChannels), "32", "32", str(outputChannels)])
       
    cConvolution = memManage.readArrayFromMemory(mainMemoryFile, OUTPUTOFFSET, outputChannels * h_out * w_out)
        
    fdata = output_image.flatten()
    for i in range(len(fdata)):
        if (fdata[i] - cConvolution[i]) > 0.001:
            print("Error")
            print(i, fdata[i], cConvolution[i])   
            exit(1) 
    print("Success")
    

    