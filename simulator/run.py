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
CONV_OUTPUTOFFSET = BIASOFFSET + (MAX_OUTPUT_CHANNELS * 4)

MAXPOOL_OUTPUTOFFSET = CONV_OUTPUTOFFSET + (MAX_OUTPUT_CHANNELS * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH * 4)

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


def max_pooling(inputData, inputChannels, height, width, pool_size=(2,2), stride=2):
    
    pool_height, pool_width = pool_size

    # Calculate the size of the output
    out_height = (height - pool_height) // stride + 1
    out_width = (width - pool_width) // stride + 1

    # Initialize the output with zeros
    output = np.zeros((inputChannels, out_height, out_width))

    # Perform max pooling
    for cin in range(inputChannels):
        for hout in range(out_height):
            for wout in range(out_width):
                h_start = hout * stride
                h_end = h_start + pool_height
                w_start = wout * stride
                w_end = w_start + pool_width
                max_val = np.max(inputData[cin, h_start:h_end, w_start:w_end])
                # relu
                relu_val = max(max_val, 0)
                # commit to output
                output[cin, hout, wout] = relu_val
    return output

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
    
    sizeOfmainMemory = 12 * 1024 * 1024 # 8MB KByte
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
    # if True:
    if False:
        memManage.setupMemory(sizeOfmainMemory, mainMemoryFile)
        # Write the image to memory
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, image, inputChannels * height * width, 0)
        # Write the kernels to memory starting after max image size
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, conv1_weights, outputChannels * inputChannels * filterSize * filterSize, KERNELOFFSET)
        # Write the bias to memory starting after max kernel size
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, conv1_bias, outputChannels, BIASOFFSET)
    
    output_image = convolution(image, conv1_weights, conv1_bias, inputChannels, height, width, outputChannels, filterSize, stride, pad)
    # for i in range(10):
    #     print("output_image[{}]: {}".format(i, output_image[0, 0, i]))
    output_image = max_pooling(output_image, 6, 28, 28)
    
    for i in range(10):
        print("maxRef[{}]: {}".format(i, output_image[0, 3, i]))
        
    exeCommand(["bins/conv8_16_5", "memory/memory.txt", "4", str(baseAddress), str(inputChannels), "32", "32", str(outputChannels)])
    exeCommand(["src/maxP2_2", "memory/memory.txt", "4", str(CONV_OUTPUTOFFSET), str(outputChannels), "28", "28", str(outputChannels)])
       
    cConvolution = memManage.readArrayFromMemory(mainMemoryFile, CONV_OUTPUTOFFSET, outputChannels * h_out * w_out)    
    output = memManage.readArrayFromMemory(mainMemoryFile, MAXPOOL_OUTPUTOFFSET, int(outputChannels * (h_out/2) * (w_out/2)))
        
    # fdata = output_image.flatten()
    # print(len(fdata))
    # for i in range(len(fdata)):
    #     # if (fdata[i] - output[i]) > 0.001:
    #     print("Error")
    #     print(i, fdata[i], output[i])   
    #         # exit(1) 
    # print("Success")
    
    for i in range(10):
        print("maxC[{}]: {}".format(i, output[3 * 14 + i]))
    

# std::cout << "pool2[" << i << "]: " << pool2_output[0][3][i] << std::endl;
# pool2[0]: 0
# pool2[1]: 0
# pool2[2]: 0.00917368
# pool2[3]: 0.348876
# pool2[4]: 0.649001
# pool2[5]: 0.55026
# pool2[6]: 0.721102
# pool2[7]: 0.766258
# pool2[8]: 0.790058
# pool2[9]: 0.761595