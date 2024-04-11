import struct
import subprocess
import numpy as np

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

def setupMemory(memory_size, filename):
    with open(filename, 'w') as mem_file:
        for _ in range(memory_size):
            mem_file.write('00\n')  # Write each byte in hex format, new line as a separator
            
def write_float_to_memory(mainMemoryFile, address, value):
    """Write a float value to a specific starting address in the memory."""
    # Convert the float to 4 bytes
    bytes_value = struct.pack('f', value)
    with open(mainMemoryFile, 'r+') as mem_file:
        lines = mem_file.readlines()
        for i in range(4):  # Assuming a float is 4 bytes
            # Extract each byte from the bytes object and write it to memory
            byte_str = f'{bytes_value[i]:02X}\n'
            lines[address + i] = byte_str
            
        mem_file.seek(0)
        mem_file.writelines(lines)
        
def read_float_from_memory(mainMemoryFile, address):
    """Read a float value from a specific starting address in the memory."""
    bytes_value = bytearray()
    with open(mainMemoryFile, 'r') as mem_file:
        lines = mem_file.readlines()
        for i in range(4):  # Assuming a float is 4 bytes
            byte_str = lines[address + i].strip()
            bytes_value.append(int(byte_str, 16))
    # Convert the 4 bytes back into a float
    return struct.unpack('f', bytes_value)[0]

def readArrayFromMemory(mainMemoryFile, address, size, dtype='f'):
    array = np.zeros(size, dtype=np.float32)  # Assuming dtype='f' corresponds to np.float32
    with open(mainMemoryFile, 'r') as mem_file:
        # Move to the position where we want to start reading
        mem_file.seek(address * 3)  # Each byte is 2 hex digits plus a newline
        
        for i in range(size):
            bytes_value = bytearray()
            for j in range(4):  # Assuming the data type is 4 bytes
                byte_str = mem_file.readline().strip()
                bytes_value.append(int(byte_str, 16))
            
            # Unpack the bytes to the specified dtype and store in the array
            array[i] = struct.unpack(dtype, bytes_value)[0]
    
    return array

def writeFloatArrayToMemory(mainMemoryFile, numpyData, size, address):

    array = numpyData.flatten()
    with open(mainMemoryFile, 'r+') as mem_file:
        lines = mem_file.readlines()
        for i in range(size):
            # floats are 4 bytes            
            bytes_value = struct.pack('f', array[i])
            
            for j in range(4):  # Assuming a float is 4 bytes
                # Extract each byte from the bytes object and write it to memory
                byte_str = f'{bytes_value[j]:02X}\n'
                lines[address + 4 * i + j] = byte_str
            
        mem_file.seek(0)
        mem_file.writelines(lines)
        
    return address + 4 * size

def exeCommand(command):
    result = subprocess.run(command, capture_output=True, text=True)
    # Check if the command was executed successfully
    if result.returncode == 0:
        # Command executed successfully, print stdout
        print("Output:\n", result.stdout)
    else:
        # There was an error, print stderr
        print("Error:\n", result.stderr)

# Example usage
if __name__ == "__main__":
    
    sizeOfmainMemory = 8 * 1024 * 1024 # 8MB KByte
    mainMemoryFile = "memory/memory.txt"

    
    # Initialize a dummy image and kernels
    
    inputChannels = 8
    height = 32
    width = 32
    outputChannels = 16
    filterSize = 5
    stride = 1
    pad = 0
    
    h_out = int((height - filterSize + 2 * pad) / stride) + 1
    w_out = int((width - filterSize + 2 * pad) / stride) + 1
    
    np.random.seed(42)
    image = np.random.uniform(low=-10.0, high=10.0, size=(inputChannels, height, width))
    kernels = np.random.uniform(low=-10.0, high=10.0, size=(outputChannels, inputChannels, filterSize, filterSize))
    bias = np.random.uniform(low=-10.0, high=10.0, size=outputChannels)
    
    #setupMemory
    if True:
    # if False:
        setupMemory(sizeOfmainMemory, mainMemoryFile)
        # Write the image to memory
        nextAddress = writeFloatArrayToMemory(mainMemoryFile, image, inputChannels * height * width, 0)
        # Write the kernels to memory starting after max image size
        nextAddress = writeFloatArrayToMemory(mainMemoryFile, kernels, outputChannels * inputChannels * filterSize * filterSize, KERNELOFFSET)
        # Write the bias to memory starting after max kernel size
        nextAddress = writeFloatArrayToMemory(mainMemoryFile, bias, outputChannels, BIASOFFSET)
    
    output_image = convolution(image, kernels, bias, inputChannels, height, width, outputChannels, filterSize, stride, pad)
    
    exeCommand(["src/conv8_16", "4", "0", "8", "32", "32", "16"])
       
    cConvolution = readArrayFromMemory(mainMemoryFile, OUTPUTOFFSET, outputChannels * h_out * w_out)
        
    fdata = output_image.flatten()
    for i in range(len(fdata)):
        if (fdata[i] - cConvolution[i]) > 0.001:
            print("Error")
            print(i, fdata[i], cConvolution[i])    
    print("Success")
    
    # write_float_to_memory(mainMemoryFile, 4, 1.602)
    # value = read_float_from_memory(mainMemoryFile, nextAddress - 4)
    
    