import struct
import numpy as np

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

def writeFloatArrayToMemory(mainMemoryFile, numpyData, address, size):
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