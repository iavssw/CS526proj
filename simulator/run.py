import struct
import subprocess
import numpy as np

from pylib import readbins
from pylib import memManage

def exeCommand(command):
    # print("Executing command:", ' '.join(command))
    # return
    
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

def parseISA(source_file_path, binDir, dataDir, memoryDir, streamDepth, dataObjects):
    
    programedTiles = [None] * 16
    outdata= {}
    
    # Open the source file to read from
    with open(source_file_path, 'r') as file:
        for line in file:
            # skip empty lines and comments
            if not line.strip() or line.strip().startswith('#'):
                    continue  # Skip the line
            
            data = line.strip().split()            
            if data[0] == "program":
                # print("program", "memory/stream" + data[1], data[2])
                programedTiles[int(data[1])] = data[2]
                memManage.setupMemory(streamDepth, "memory/stream" + data[1]) #conv1
            elif data[0] == "memcpy2device":
                # print("memcpy2device", memoryDir + "mainmemory", data[1], data[2], data[3])
                memManage.writeFloatArrayToMemory(memoryDir + "mainmemory", dataObjects[data[1]], int(data[2]), int(data[3]))
            elif data[0] == "memcpy2host":
                # print("memcpy2host", memoryDir + "mainmemory", data[1], data[2], data[3])
                outdata[data[1]] = memManage.readArrayFromMemory(memoryDir + "mainmemory", int(data[2]), int(data[3]))
            elif data[0] == "convR8_32_5":
                if (not programedTiles[int(data[1])] == data[0]):
                    print("Error: Tile not programmed")
                    return
                # FPGA program
                command = []
                # program name
                command.append(binDir + data[0])
                # main device memory
                command.append(memoryDir + "mainmemory")
                # input stream
                command.append(memoryDir + "stream" + data[1])                
                # output stream
                destTiles = data[2].split("-")
                for tile in destTiles:
                    command.append("memory/stream" + tile)
                
                # stream setting
                if data[3] == "MM":
                    command.append("00")
                elif data[3] == "MS":
                    command.append("01")
                elif data[3] == "SM":
                    command.append("10")
                elif data[3] == "SS":
                    command.append("11")
                else:
                    print("Error: Invalid stream configuration")
                    return
                
                # input start address
                command.append(data[4])
                # conv weights start address
                command.append(data[5])
                # conv bias start address
                command.append(data[6])
                # output start address
                command.append(data[7])
                
                # input channel
                command.append(data[8])
                # input height
                command.append(data[9])
                # input width
                command.append(data[10])
                # output channel
                command.append(data[11])
                    
                exeCommand(command)
            elif data[0] == "maxp2_2":
                if (not programedTiles[int(data[1])] == data[0]):
                    print("Error: Tile not programmed")
                    return
                # FPGA program
                command = []
                # program name
                command.append(binDir + data[0])
                # main device memory
                command.append(memoryDir + "mainmemory")
                # input stream
                command.append(memoryDir + "stream" + data[1])                
                # output stream
                destTiles = data[2].split("-")
                for tile in destTiles:
                    command.append("memory/stream" + tile)
                
                # stream setting
                if data[3] == "MM":
                    command.append("00")
                elif data[3] == "MS":
                    command.append("01")
                elif data[3] == "SM":
                    command.append("10")
                elif data[3] == "SS":
                    command.append("11")
                else:
                    print("Error: Invalid stream configuration")
                    return
                
                # input start address
                command.append(data[4])
                # output start address
                command.append(data[5])
                
                # input channel
                command.append(data[6])
                # input height
                command.append(data[7])
                # input width
                command.append(data[8])
                # poolsize
                command.append(data[9])
                # stride 
                command.append(data[10])
                    
                exeCommand(command)
            elif data[0] == "fc128_64":
                if (not programedTiles[int(data[1])] == data[0]):
                    print("Error: Tile not programmed")
                    return
                # FPGA program
                command = []
                # program name
                command.append(binDir + data[0])
                # main device memory
                command.append(memoryDir + "mainmemory")
                # input stream
                command.append(memoryDir + "stream" + data[1])                
                # output stream
                destTiles = data[2].split("-")
                for tile in destTiles:
                    command.append("memory/stream" + tile)
                
                # stream setting
                if data[3] == "MM":
                    command.append("00")
                elif data[3] == "MS":
                    command.append("01")
                elif data[3] == "SM":
                    command.append("10")
                elif data[3] == "SS":
                    command.append("11")
                else:
                    print("Error: Invalid stream configuration")
                    return
                
                # input start address
                command.append(data[4])
                # conv weights start address
                command.append(data[5])
                # conv bias start address
                command.append(data[6])
                # output start address
                command.append(data[7])
                
                # input channel
                command.append(data[8])
                # output channel
                command.append(data[9])
                    
                exeCommand(command)    
            else:
                print("Error: Invalid Instruction")
                returns        
    # print(programedTiles)
    return outdata

if __name__ == "__main__":
    
    binDir = "src/"
    dataDir = "data/"
    memoryDir = "memory/"
    
    sizeOfmainMemory = 4 * 1024 * 1024 # 8MB
    streamDepth = 1 *1024 * 1024 # 1MB 
    
    #get data for Lenet Test
    images, (num_images, rows, cols) = readbins.parse_mnist_images(dataDir + "images.bin")
    labels = readbins.parse_mnist_labels(dataDir + "labels.bin")
    conv1_weights, conv1_bias, conv3_weights, conv3_bias, conv5_weights, conv5_bias, fc6_weights, fc6_bias = readbins.parse_parameters(dataDir + "params.bin")
    
    dataObjects = {}
    # Parsing image data and storing in dictionary
    dataObjects['images'], (dataObjects['num_images'], dataObjects['rows'], dataObjects['cols']) = readbins.parse_mnist_images(dataDir + "images.bin")
    # Parsing label data and storing in dictionary
    dataObjects['labels'] = readbins.parse_mnist_labels(dataDir + "labels.bin")
    # Parsing network parameters and storing in dictionary
    (dataObjects['conv1_weights'], dataObjects['conv1_bias'], dataObjects['conv3_weights'], dataObjects['conv3_bias'],
     dataObjects['conv5_weights'], dataObjects['conv5_bias'], dataObjects['fc6_weights'], dataObjects['fc6_bias']
    ) = readbins.parse_parameters(dataDir + "params.bin")
    
    dataObjects['image'] = readbins.get_image(dataObjects['images'], 0)
    
    #setup main memory
    memManage.setupMemory(sizeOfmainMemory, memoryDir + "mainmemory")
    
    outdata = parseISA("./lenetFPGA.ISA", binDir, dataDir, memoryDir, streamDepth, dataObjects)
    
    for i in range(10):
        print(outdata['output'][i])

# correct out      
# 0.008176647
# 0.0049868487
# 0.0
# 0.0
# 0.0
# 0.018932745
# 0.0043978393
# 0.9807243
# 0.008769132
# 0.0