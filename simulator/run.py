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

# Example usage
if __name__ == "__main__":
    
    sizeOfmainMemory = 4 * 1024 * 1024 # 8MB KByte
    mainMemoryFile = "memory/mainmemory"
    
    baseAddress = 0
    
    conv1 = ConvLayerMemoryManager(base_address = baseAddress, input_channels=1, height=32, width=32, output_channels=6, filter_size=5, stride=1, pad=0)
    maxpr2 = MaxPoolReluLayerMemoryManager(base_address = conv1.addr_conv_output, input_channels=6, height=28, width=28, pool_size=2, stride=2)
    
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
        memManage.setupMemory(1 * 1024 * 1024, "memory/stream1")
        memManage.setupMemory(1 * 1024 * 1024, "memory/stream2")

        # Write the image to memory
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, image, conv1.num_input_pixels, conv1.addr_image)
        # Write the kernels to memory starting after max image size
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, conv1_weights, conv1.num_kernel_elements, conv1.addr_kernel)
        # Write the bias to memory starting after max kernel size
        nextAddress = memManage.writeFloatArrayToMemory(mainMemoryFile, conv1_bias, conv1.num_bias_elements, conv1.addr_bias)
    
    #sanity check
    output_image = convolution(image, conv1_weights, conv1_bias, conv1.input_channels, conv1.height, conv1.width, conv1.output_channels, \
        conv1.filter_size, conv1.stride, conv1.pad)
    output_image = max_pooling(output_image, maxpr2.input_channels, maxpr2.width, maxpr2.height, pool_size=maxpr2.pool_size, stride=maxpr2.stride)
    
    # for i in range(10):
    #     print("maxRef[{}]: {}".format(i, output_image[0, 3, i]))
        
    exeCommand(["src/conv8_16_5", mainMemoryFile, "memory/stream1", "memory/stream2", "01", str(conv1.addr_image), str(conv1.addr_kernel), str(conv1.addr_bias), str(conv1.addr_conv_output), \
        str(conv1.input_channels), str(conv1.height), str(conv1.width), str(conv1.output_channels)])
    exeCommand(["src/maxP2_2", mainMemoryFile, "memory/stream2", "memory/stream2", "10", str(conv1.addr_conv_output), str(maxpr2.addr_output), \
        str(maxpr2.input_channels), str(maxpr2.width), str(maxpr2.height), str(maxpr2.pool_size), str(maxpr2.stride)])
       
    # cConvolution1 = memManage.readArrayFromMemory(mainMemoryFile, CONV_OUTPUTOFFSET, outputChannels * h_out * w_out)    
    maxpool2 = memManage.readArrayFromMemory(mainMemoryFile, maxpr2.addr_output, maxpr2.num_output_pixels)
    
    # memManage.write_float_to_memory("memory/1stream.txt", 0, 42.42)
        
    fdata = output_image.flatten()
    for i in range(len(fdata)):
        if (fdata[i] - maxpool2[i]) > 0.001:
            print("Error")
            print(i, fdata[i], maxpool2[i])   
            exit(1) 
    print("Success")
        

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