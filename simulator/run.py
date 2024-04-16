import struct
import subprocess
import numpy as np

from pylib import readbins
from pylib import memManage

def convolution(inputImage, convWeight, convBias, inputChannels, height, width, outputChannels, filterSize, stride, pad):
    # Output dimensions, assuming no padding (PAD=0) and stride of 1
    hout = int((height + 2 * pad - filterSize) / stride) + 1
    wout = int((width + 2 * pad - filterSize) / stride) + 1

    # Initialize the output image with zeros
    convOutput = np.zeros((outputChannels, hout, wout), dtype=np.float32)

    for co in range(outputChannels):
        for h in range(hout):
            for w in range(wout):
                sum = 0.0
                for cin in range(inputChannels):
                    for i, m in zip(range(h, h + filterSize), range(filterSize)):
                        for j, n in zip(range(w, w + filterSize), range(filterSize)):
                            # Ensure indices are within bounds before accessing arrays
                            if i < height and j < width:
                                sum += inputImage[cin, i, j] * convWeight[co, cin, m, n]
                
                # convOutput[co, h, w] = sum + convBias[co]
                # relu
                convOutput[co, h, w] = max(sum + convBias[co], 0)

    return convOutput


def max_pooling(inputData, inputChannels, height, width, pool_size=2, stride=2):
    
    # Calculate the size of the output
    out_height = (height - pool_size) // stride + 1
    out_width = (width - pool_size) // stride + 1

    # Initialize the output with zeros
    output = np.zeros((inputChannels, out_height, out_width))

    # Perform max pooling
    for cin in range(inputChannels):
        for hout in range(out_height):
            for wout in range(out_width):
                h_start = hout * stride
                h_end = h_start + pool_size
                w_start = wout * stride
                w_end = w_start + pool_size
                max_val = np.max(inputData[cin, h_start:h_end, w_start:w_end])
                # relu
                # relu_val = max(max_val, 0)
                # commit to output
                output[cin, hout, wout] = max_val
    return output

def fullyConnected(input, weights, bias, inputChannels, output_channels):
    
    # Perform the matrix multiplication and add the bias
    rawOutput = np.dot(weights, input.reshape(inputChannels, 1)).reshape(output_channels) + bias
    
    # Apply the ReLU activation function
    output = np.maximum(rawOutput, 0)
    
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

class ConvLayerMemoryManager:
    def __init__(self, base_address, input_channels, height, width, output_channels, filter_size, stride, pad):
        self.base_address = base_address
        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.output_channels = output_channels
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad

        # Calculate output dimensions
        self.h_out = int((self.height - self.filter_size + 2 * self.pad) / self.stride) + 1
        self.w_out = int((self.width - self.filter_size + 2 * self.pad) / self.stride) + 1
        
        # number of pixels
        self.num_input_pixels = self.input_channels * self.height * self.width
        self.num_output_pixels = self.output_channels * self.h_out * self.w_out
        self.num_kernel_elements = self.output_channels * self.input_channels * self.filter_size * self.filter_size
        self.num_bias_elements = self.output_channels
        
        # Calculate memory addresses assuming fp32
        self.addr_image = self.base_address
        self.addr_kernel = self.addr_image + self.num_input_pixels * 4
        self.addr_bias = self.addr_kernel + self.num_kernel_elements * 4
        self.addr_conv_output = self.addr_bias + self.num_bias_elements * 4
        
class MaxPoolReluLayerMemoryManager:
    def __init__(self, base_address, input_channels, height, width, pool_size, stride):
        self.base_address = base_address
        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.pool_size = pool_size
        self.stride = stride
        
        # Calculate output dimensions
        self.h_out = int((self.height - self.pool_size) / self.stride) + 1
        self.w_out = int((self.width - self.pool_size) / self.stride) + 1

        # number of pixels
        self.num_input_pixels = self.input_channels * self.height * self.width
        self.num_output_pixels = self.input_channels * self.h_out * self.w_out
        
        # Calculate memory addresses assuming fp32
        self.addr_input = self.base_address
        self.addr_output = self.addr_input + self.num_input_pixels * 4
        
class fullyConnectedReluLayerMemoryManager:
    def __init__(self, base_address, input_channels, output_channels):
        self.base_address = base_address
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # number of pixels
        self.num_input_pixels = self.input_channels
        self.num_output_pixels = self.output_channels
        self.num_kernel_elements = self.output_channels * self.input_channels
        self.num_bias_elements = self.output_channels
        
        # Calculate memory addresses assuming fp32
        self.addr_input = self.base_address
        self.addr_kernel = self.addr_input + self.num_input_pixels * 4
        self.addr_bias = self.addr_kernel + self.num_kernel_elements * 4
        self.addr_output = self.addr_bias + self.num_bias_elements * 4

# Example usage
if __name__ == "__main__":
    
    sizeOfmainMemory = 4 * 1024 * 1024 # 8MB KByte
    mainMemoryFile = "memory/mainmemory"
    
    baseAddress = 0
    
    conv1 = ConvLayerMemoryManager(base_address = baseAddress, input_channels=1, height=32, width=32, output_channels=6, filter_size=5, stride=1, pad=0)
    maxpr2 = MaxPoolReluLayerMemoryManager(base_address = conv1.addr_conv_output, input_channels=6, height=28, width=28, pool_size=2, stride=2)
    conv3 = ConvLayerMemoryManager(base_address = maxpr2.addr_output, input_channels=6, height=14, width=14, output_channels=16, filter_size=5, stride=1, pad=0)
    maxpr4 = MaxPoolReluLayerMemoryManager(base_address = conv3.addr_conv_output, input_channels=16, height=10, width=10, pool_size=2, stride=2)
    conv5 = ConvLayerMemoryManager(base_address = maxpr4.addr_output, input_channels=16, height=5, width=5, output_channels=120, filter_size=5, stride=1, pad=0)
    fc6 = fullyConnectedReluLayerMemoryManager(base_address = conv5.addr_conv_output, input_channels=120, output_channels=10)
    
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
        memManage.setupMemory(1 * 1024 * 1024, "memory/stream1") #conv1
        memManage.setupMemory(1 * 1024 * 1024, "memory/stream2") #maxpool2
        memManage.setupMemory(1 * 1024 * 1024, "memory/stream3") #conv3
        memManage.setupMemory(1 * 1024 * 1024, "memory/stream4") #maxpool4
        
        memManage.setupMemory(1 * 1024 * 1024, "memory/stream5") #conv5
        memManage.setupMemory(1 * 1024 * 1024, "memory/stream6") #conv5
        memManage.setupMemory(1 * 1024 * 1024, "memory/stream7") #conv5
        memManage.setupMemory(1 * 1024 * 1024, "memory/stream8") #conv5        
        

        # Write the image to memory
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, image, conv1.num_input_pixels, conv1.addr_image)
        
        # Conv1
        # Write the kernels to memory starting after max image size
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, conv1_weights, conv1.num_kernel_elements, conv1.addr_kernel)
        # Write the bias to memory starting after max kernel size
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, conv1_bias, conv1.num_bias_elements, conv1.addr_bias)
        
        # Conv3
        # Write the kernels to memory starting after max image size
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, conv3_weights, conv3.num_kernel_elements, conv3.addr_kernel)
        # Write the bias to memory starting after max kernel size
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, conv3_bias, conv3.num_bias_elements, conv3.addr_bias)
        
        # Conv5
        # Write the kernels to memory starting after max image size
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, conv5_weights, conv5.num_kernel_elements, conv5.addr_kernel)
        # Write the bias to memory starting after max kernel size
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, conv5_bias, conv5.num_bias_elements, conv5.addr_bias)
        
        # FC6
        # Write the kernels to memory starting after max image size
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, fc6_weights, fc6.num_kernel_elements, fc6.addr_kernel)
        # Write the bias to memory starting after max kernel size
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, fc6_bias, fc6.num_bias_elements, fc6.addr_bias)
    
    #sanity check
    # output_image = convolution(image, conv1_weights, conv1_bias, conv1.input_channels, conv1.height, conv1.width, conv1.output_channels, \
    #     conv1.filter_size, conv1.stride, conv1.pad)
    # output_image = max_pooling(output_image, maxpr2.input_channels, maxpr2.width, maxpr2.height, pool_size=maxpr2.pool_size, stride=maxpr2.stride)
    # output_image = convolution(output_image, conv3_weights, conv3_bias, conv3.input_channels, conv3.height, conv3.width, conv3.output_channels, \
    #     conv3.filter_size, conv3.stride, conv3.pad)
    # output_image = max_pooling(output_image, maxpr4.input_channels, maxpr4.width, maxpr4.height, pool_size=maxpr4.pool_size, stride=maxpr4.stride)
    # output_image = convolution(output_image, conv5_weights, conv5_bias, conv5.input_channels, conv5.height, conv5.width, conv5.output_channels, \
    #     conv5.filter_size, conv5.stride, conv5.pad)
    # output_image = fullyConnected(output_image, fc6_weights, fc6_bias, fc6.input_channels, fc6.output_channels)
        
    #conv1
    exeCommand(["src/convR8_32_5", mainMemoryFile, "memory/stream1", "memory/stream2", "01", str(conv1.addr_image), str(conv1.addr_kernel), str(conv1.addr_bias), str(conv1.addr_conv_output), \
        str(conv1.input_channels), str(conv1.height), str(conv1.width), str(conv1.output_channels)])
    #maxpool2
    exeCommand(["src/maxp2_2", mainMemoryFile, "memory/stream2", "memory/stream3", "11", str(maxpr2.addr_input), str(maxpr2.addr_output), \
        str(maxpr2.input_channels), str(maxpr2.width), str(maxpr2.height), str(maxpr2.pool_size), str(maxpr2.stride)])
    #conv3
    exeCommand(["src/convR8_32_5", mainMemoryFile, "memory/stream3", "memory/stream4", "11", str(conv3.addr_image), str(conv3.addr_kernel), str(conv3.addr_bias), str(conv3.addr_conv_output), \
        str(conv3.input_channels), str(conv3.height), str(conv3.width), str(conv3.output_channels)])
    #maxpool4 -> stream to 5 differnt conv5 tiles
    exeCommand(["src/maxp2_2", mainMemoryFile, "memory/stream4", "memory/stream5", "11" \
                , str(maxpr4.addr_input), str(maxpr4.addr_output), str(maxpr4.input_channels), str(maxpr4.width), str(maxpr4.height), str(maxpr4.pool_size), str(maxpr4.stride)])
    
    # conv5 place holder
    exeCommand(["src/convR8_32_5", mainMemoryFile, "memory/stream5", "memory/stream6", "11", str(conv5.addr_image), str(conv5.addr_kernel), str(conv5.addr_bias), str(conv5.addr_conv_output), \
        str(conv5.input_channels), str(conv5.height), str(conv5.width), str(conv5.output_channels)])
    
    # fc6 
    exeCommand(["src/fc128_64", mainMemoryFile, "memory/stream6", "memory/stream6", "10", str(fc6.addr_input), str(fc6.addr_kernel), str(fc6.addr_bias), str(fc6.addr_output), \
        str(fc6.input_channels), str(fc6.output_channels)])
    
    # Read the output from memory
    # cConvolution3 = memManage.readArrayFromMemory(mainMemoryFile, conv3.addr_conv_output, conv3.num_output_pixels)    
    # maxpool4 = memManage.readArrayFromMemory(mainMemoryFile, maxpr4.addr_output, maxpr4.num_output_pixels)
    # cConvolution5 = memManage.readArrayFromMemory(mainMemoryFile, conv5.addr_conv_output, conv5.num_output_pixels)    
    cFC6 = memManage.readArrayFromMemory(mainMemoryFile, fc6.addr_output, fc6.num_output_pixels)    
    
    for i in range(10):
        print("fc6_output[{}]: {}".format(i, cFC6[i]))
        
# Starting inference
# fc6_output[0]: 0.00817657
# fc6_output[1]: 0.00498692
# fc6_output[2]: 0
# fc6_output[3]: 0
# fc6_output[4]: 0
# fc6_output[5]: 0.0189327
# fc6_output[6]: 0.00439781
# fc6_output[7]: 0.980724
# fc6_output[8]: 0.00876922
# fc6_output[9]: 0
# Done with inference

#compare data    
# fdata = output_image.flatten()
# for i in range(len(fdata)):
#     if (fdata[i] - cConvolution5[i]) > 0.001:
#         print("Error")
#         print(i, fdata[i], cConvolution5[i])   
#         exit(1) 
# print("Success")