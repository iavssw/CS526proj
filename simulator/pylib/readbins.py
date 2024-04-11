import struct
import numpy as np

def parse_mnist_images(filename):
    try:
        with open(filename, "rb") as fil:
            print("Opened mnist images data file")

            # Read and unpack the header
            header = struct.unpack('>IIII', fil.read(16))
            num_images = header[1]
            rows = header[2]
            cols = header[3]

            # Assuming NUM_TESTS is num_images read from the header
            image_data = fil.read(num_images * rows * cols)
            if len(image_data) != num_images * rows * cols:
                print("Can't read images from file")
                return -1
            else:
                print("Read images from file")

            print("Closed images file")
            images = np.frombuffer(image_data, dtype=np.uint8)
            return images, (num_images, rows, cols)

    except IOError as e:
        print(f"ERROR when opening mnist images data file: {e}")
        return -1, None  
        
def parse_mnist_labels(filename):
    try:
        with open(filename, "rb") as fil:
            print("Opened mnist labels data file")

            # Read and unpack the header
            header = struct.unpack('>II', fil.read(8))
            num_labels = header[1]

            labels = fil.read(num_labels)
            if len(labels) != num_labels:
                print("Can't read labels from file")
                return -1
            else:
                print("Read labels from file")

            print("Closed labels file")
            return labels

    except IOError as e:
        print(f"ERROR when opening mnist labels data file: {e}")
        return -1

def parse_parameters(filename):
    try:
        with open(filename, "rb") as fil:
            print("Opened parameter file")

            # Assuming conv1_weights is a 3D array-like structure
            conv1_weights_size = 150  # Total number of floats
            conv1_weights = struct.unpack('f' * conv1_weights_size, fil.read(4 * conv1_weights_size))
            print("Read conv1_weights from file")

            conv1_bias_size = 6
            conv1_bias = struct.unpack('f' * conv1_bias_size, fil.read(4 * conv1_bias_size))
            print("Read conv1_bias from file")

            # Assuming conv3_weights is a 3D array-like structure
            conv3_weights_size = 2400
            conv3_weights = struct.unpack('f' * conv3_weights_size, fil.read(4 * conv3_weights_size))
            print("Read conv3_weights from file")

            conv3_bias_size = 16
            conv3_bias = struct.unpack('f' * conv3_bias_size, fil.read(4 * conv3_bias_size))
            print("Read conv3_bias from file")

            # Assuming conv5_weights is a 3D array-like structure
            conv5_weights_size = 48000
            conv5_weights = struct.unpack('f' * conv5_weights_size, fil.read(4 * conv5_weights_size))
            print("Read conv5_weights from file")

            conv5_bias_size = 120
            conv5_bias = struct.unpack('f' * conv5_bias_size, fil.read(4 * conv5_bias_size))
            print("Read conv5_bias from file")

            # Assuming fc6_weights is a 2D array-like structure
            fc6_weights_size = 1200
            fc6_weights = struct.unpack('f' * fc6_weights_size, fil.read(4 * fc6_weights_size))
            print("Read fc6_weights from file")

            fc6_bias_size = 10
            fc6_bias = struct.unpack('f' * fc6_bias_size, fil.read(4 * fc6_bias_size))
            print("Read fc6_bias from file")

            print("Closed labels file")
            
            conv1_weights = np.array(conv1_weights).reshape((6, 1, 5, 5))
            conv1_bias = np.array(conv1_bias)
            conv3_weights = np.array(conv3_weights).reshape((16, 6, 5, 5))
            conv3_bias = np.array(conv3_bias)
            conv5_weights = np.array(conv5_weights).reshape((120, 16, 5, 5))
            conv5_bias = np.array(conv5_bias)
            fc6_weights = np.array(fc6_weights).reshape((10, 120))
            fc6_bias = np.array(fc6_bias)            
            
            return conv1_weights, conv1_bias, conv3_weights, conv3_bias, conv5_weights, conv5_bias, fc6_weights, fc6_bias

    except IOError as e:
        print(f"ERROR when opening parameter file: {e}")
        return -1
    
def get_image(images, idx):
    # Create an empty 1x32x32 array filled with -1.0 (for padding)
    image = np.full((1, 32, 32), -1.0, dtype=np.float32)
    
    # Extract the idx-th 28x28 image from the flat array
    # and reshape it to 28x28
    original_image = images[idx * 28 * 28 : (idx + 1) * 28 * 28].reshape((28, 28))
    
    # Normalize pixel values to [-1.0, 1.0] and place the image in the center of the 1x32x32 array
    image[0, 2:30, 2:30] = original_image / 255.0 * 2.0 - 1.0
    
    return image

# Example usage
if __name__ == "__main__":
    filename = "path_to_your_mnist_labels_file"
    labels = parse_mnist_labels(filename)
    if labels != -1:
        print("Labels parsed successfully")
    else:
        print("Failed to parse labels")

    filename = "path_to_your_mnist_images_file"
    images, dims = parse_mnist_images(filename)
    if images != -1:
        print("Images parsed successfully")
    else:
        print("Failed to parse images")