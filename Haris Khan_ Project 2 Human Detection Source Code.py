import numpy as np  #For matrix operations
from scipy.misc import imread  #Used to read images
import matplotlib.pyplot as plt  #Used to show images
import math  #Used in normalized edge magnitude and angle calculation
import random


# Input image will be a grayscale image
def makeGrayscale(inputImage):
    return np.round_(np.dot(inputImage[...,:3], [0.299, 0.587, 0.114]))


# input image here will be the image resulting after converting image to grayscale
def xgradient(inputImage):
    outputImage = np.copy(inputImage)
    # Nested for loop to visit each pixel in the image
    for i in range(inputImage.shape[0]-1):
        for j in range(inputImage.shape[1]-1):
            # Check to see if pixel, or pixel's 8-connected neighbors are undefined/None
            if ( ((i-1)<0) or ((j-1)<0) or ((i+1)>inputImage.shape[0]) or ((j+1)>inputImage.shape[1])):
                outputImage[i,j] = 0
            else:
                # Prewitt Operator in x-direction; prewitt operator matrix not made because its elements are equal
                # to 1 or -1. Instead proper coefficients applied to the 3x3 window
                outputImage[i,j] = (inputImage[i-1,j+1] + inputImage[i,j+1] + inputImage[i+1,j+1] \
                                   - inputImage[i-1,j-1] - inputImage[i,j-1] - inputImage[i+1,j-1])/3
    return outputImage #normalized horizontal gradient response


# input image here will be the image resulting after gaussian smoothing
def ygradient(inputImage):
    outputImage = np.copy(inputImage)
    # Nested for loop to visit each pixel in the image
    for i in range(inputImage.shape[0]-1):
        for j in range(inputImage.shape[1]-1):
            # Check to see if pixel, or pixel's 8-connected neighbors are undefined/None
            if ( ((i-1)<0) or ((j-1)<0) or ((i+1)>inputImage.shape[0]) or ((j+1)>inputImage.shape[1])):
                outputImage[i,j] = 0
            else:
                # Prewitt Operator in y-direction; prewitt operator matrix not made because its elements are equal
                # to 1 or -1
                outputImage[i,j] = (inputImage[i-1,j-1] + inputImage[i-1,j] + inputImage[i-1,j+1] \
                                   - inputImage[i+1,j-1] - inputImage[i+1,j] - inputImage[i+1,j+1])/3
    return outputImage #normalized vertical gradient response


# input parameter x is the normalized horizontal gradient response, and y is the normalized vertical gradient response
def EdgeMagnitudesAndGradientAngles(x,y):
    #This function will return 2 arrays: an array with normalized edge magnitude, and an array with gradient angles
    outputImageMagnitude = np.copy(x)
    outputImageAngles = np.copy(x)
    # Nested for loop to visit each pixel in the image
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            # Magnitude and angle are equal to 0 if magnitude in horizontal and vertical gradient is equal to 0
            # Therefore, the case is not considered to eliminate DivideByZero runtime warning 
            if ((x[i,j] != 0) and (y[i,j] != 0)):
                # Calculate normalized edge magnitude
                outputImageMagnitude[i,j] = (math.sqrt(((x[i,j])**2)+((y[i,j])**2)))/math.sqrt(2)
                # Calcualte angle
                outputImageAngles[i,j] = math.degrees(math.atan(y[i,j]/x[i,j]))
    return (outputImageMagnitude, outputImageAngles) #return a tuple containing array with edge magnitude and array with gradient angle


# Creates an HOG descriptor given gradient magnitude and angles of an image
def HoG(magnitudes, angles):
    # Create the final descriptor initially empty
    finalDescriptor = np.array([])
    # The histogram with the center angle of each bin
    centers = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])
    # Block loop to encompass 4 cells
    for blocki in range(0, magnitudes.shape[0] - 8, 8):
        for blockj in range(0, magnitudes.shape[1] - 8, 8):
            # Vector on which normalization will be done
            # Normalized vector to be concatenated to final descriptor
            vector = np.array([])
            # Cell loop. Each cell contains 8x8 cells
            for celli in range(blocki, blocki + 16, 8):
                for cellj in range(blockj, blockj + 16, 8):
                    # Each cell has its own histogram
                    histogram = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    # Iterate through every pixel in the cell
                    for i in range(celli, celli+8):
                        for j in range(cellj, cellj+8):
                            # Skips loop when outside of the image
                            if ((i>=angles.shape[0]) or (j>=angles.shape[1])):
                                continue
                            # Adjusts the gradient angle to be within range of histogram
                            if angles[i,j] < -10:
                                angles[i,j] += 360
                            if angles[i,j] >= 170:
                                angles[i,j] -= 180
                            # Shows the distance from the gradient angle to all bin centers
                            diff = np.abs(angles[i,j] - centers)
                            # Account for edge cases when gradient angle is in bin 1 or 9
                            # In these cases, consider the bins to be circular and assign neighbor accordingly 
                            if angles[i,j] < centers[0]:
                                first_bin_index = 0
                                second_bin_index = centers.size - 1
                            elif angles[i,j] > centers[-1]:
                                first_bin_index = centers.size - 1
                                second_bin_index = 0
                            # All other cases when gradient angle not in bin 1 or 9
                            else:
                                # Locate which bin center has the smallest distance from angle
                                first_bin_index = np.where(diff == np.min(diff))[0][0]
                                # Following to determing neighboring bin
                                temp = centers[[(first_bin_index-1)%centers.size, (first_bin_index+1)%centers.size]]
                                # Angle distance from neighboring bin centers
                                temp2 = np.abs(angles[i,j] - temp)
                                # Which of the neighboring bin centers has smallest distance from angle
                                res = np.where(temp2 == np.min(temp2))[0][0]
                                # Neighbor to left with smaller bin center
                                if res == 0 and first_bin_index != 0:
                                    second_bin_index = first_bin_index-1
                                # Neighbor to right with larger bin center
                                else:
                                    second_bin_index = first_bin_index+1
                            # Calculate distance from gradient angle to center of first bin
                            dist_to_center = abs(angles[i,j] - centers[first_bin_index])
                            # Calculate distance from gradient angle to center of neighbor bin
                            dist_to_neighbor = 20 - dist_to_center
                            # Assign votes to appropriate bins
                            histogram[first_bin_index] += ((abs(20 - dist_to_center)/20) * magnitudes[i,j])
                            histogram[second_bin_index] += ((abs(20 - dist_to_neighbor)/20) * magnitudes[i,j])
                    # Append histogram to vector for normalization
                    vector = np.append(vector, histogram)
            # Normalize Vector
            vector_magnitude = math.sqrt(sum(np.power(vector, 2)))
            np.seterr(divide='ignore', invalid='ignore')
            vector = vector/vector_magnitude
            # Concatenate vector to final descriptor
            finalDescriptor = np.concatenate((finalDescriptor, vector), axis=None)
    return finalDescriptor

# Class for Multi Layer Perceptron
class MLP:
    def __init__(self, input_dimension, hidden_dimension):
            # Variables for feeding forward
            self.input_dimension = input_dimension
            self.hidden_dimension = hidden_dimension
            self.num_of_features = self.input_dimension
            self.num_of_hidden = self.hidden_dimension
            self.weights_ih = np.zeros((self.input_dimension, self.hidden_dimension))
            self.bias_ih = []
            self.hidden_node_value = [0.0] * self.hidden_dimension
            self.weights_ho = []
            self.bias_ho = 1.0
            self.output = 0.0

            #Variables for back propagation
            self.input_layer_grad = [[1.0]*self.hidden_dimension]*self.input_dimension
            self.bias_ih_grad = [1.0]*self.hidden_dimension
            self.hidden_delta = [1.0]*self.hidden_dimension
            self.hidden_layer_grad = [1.0]*self.hidden_dimension 
            self.bias_ho_grad = 1.0

            # Variables for testing 
            self.learn_rate = 0.1
            self.predictions = []
            self.mean_squared_error_old = 10.
            self.mean_squared_error = 1.

    def train(self, HoGs, labels):
        # Assign random number from (0,1) to weights between input and hidden nodes
        for i in range(self.num_of_features):
            for j in range(self.num_of_hidden):
                self.weights_ih[i][j] = float(random.randint(-100, 100) / (100*self.num_of_features))

        # Assign random number from (0,1) to bias when going from input to hidden node
        for i in range(self.num_of_hidden):
            self.bias_ih.append(float(random.randint(0, 100) / (self.num_of_features)))
            
        # Assign random number from (0,1) to weights from hidden to output node
        for i in range(self.num_of_hidden):
            self.weights_ho.append(float(random.randint(-100, 100) / (100*self.num_of_hidden)))

        # Assign random number from (0,1) to bias from hidden to output
        self.bias_ho = float(random.randint(0, 100) / self.num_of_hidden)

        # Training through HoGs until mean squared error does not change very much
        epoch = 0
        while ((abs(self.mean_squared_error - self.mean_squared_error_old))/self.mean_squared_error_old) > 0.001:
            epoch += 1
            self.mean_squared_error_old = self.mean_squared_error
            self.mean_squared_error = 0.0
            # Train each test image in an epoch
            for z in range(len(HoGs)):
                self.feedForward(HoGs[z], labels[z])
                self.back_prop(HoGs[z], labels[z])
            self.mean_squared_error /= 20
            print("Mean Squared Error in epoch " + str(epoch) + " = " + str(self.mean_squared_error))            
        return


    def feedForward(self, HoG, label):
        # Value at node equals summation of each input node multiplied by its weight
        total_sum = [0.0] * self.num_of_hidden
        for j in range(self.num_of_hidden):
            for i in range(self.num_of_features):
                total_sum[j] += (self.weights_ih[i][j] * HoG[i])
            # Add bias
            total_sum[j] += self.bias_ih[j]
            # Use the ReLU activation function at the node
            self.hidden_node_value[j] = max(0, total_sum[0])

        # Value at output node found by the summation of each hidden node multiplied by its weight
        output_sum = 0.0
        for i in range(self.num_of_hidden):
            output_sum += self.hidden_node_value[i] * self.weights_ho[i]
        # Add bias
        output_sum += self.bias_ho
        # Output node value found after applying sigmoid function
        self.output = self.sigmoid_func(output_sum)
        self.mean_squared_error += 0.5 * (label - self.output)**2
        return

    def back_prop(self, HoG, label):
        # Calculate the delta from output node
        delta = self.derivative(self.output) * (label - self.output)
        
        # Set bias gradient from hidden to output equal to delta
        self.bias_ho_grad = delta

        # Set gradient of weights from hidden to output layer
        for i in range(self.num_of_hidden):
            self.hidden_layer_grad[i] = delta * self.hidden_node_value[i]
            
        # Determine the deltas of each hidden node
        for i in range(self.num_of_hidden):
            self.hidden_delta[i] = self.derivative(self.hidden_node_value[i]) * delta * self.weights_ho[i]
            
        # set bias gradient from input to hidden layer
        for i in range(self.num_of_hidden):
            self.bias_ih_grad[i] = self.hidden_delta[i]

        # Set gradient of weights from input to hidden layer
        for i in range(self.num_of_features):
            for j in range(self.num_of_hidden):
                self.input_layer_grad[i][j] = self.hidden_delta[j] * HoG[i]

        #update weights from input to hidden layer
        for i in range(HoG.shape[0]):
            for j in range(self.num_of_hidden):
                d =  self.learn_rate * self.input_layer_grad[i][j]
                self.weights_ih[i][j] += d

        # Update bias from input to hidden
        for i in range(self.num_of_hidden):
            d = self.learn_rate * self.bias_ih_grad[i]
            self.bias_ih[i] += d

        # Update weights from hidden to output
        for i in range(self.num_of_hidden):
            d = self.learn_rate * self.hidden_layer_grad[i]
            self.weights_ho[i] += d

        # Update bias from hidden to output 
        self.bias_ho += self.learn_rate * self.bias_ho_grad
        return

    # Sigmoid function used to determine value in output node
    def sigmoid_func(self, z):
        sigma = 1.0 / (1.0 + math.exp(-z))
        return sigma

    # Sigmoid output function to make predictions of test images
    def sigmoid_output_func(self, z):
        sigma = 1.0 / (1.0 + math.exp(-z))
        if sigma < 0.5:
            sigma = 0.0
        else:
            sigma = 1.0
        return sigma

    # Derivative helper function used in back propagation
    def derivative(self, x):
        return x * (1.0 - x)

    # To predict test images
    #Return list/array of predictions where there is one prediction for each HoG
    def predict(self, test_HoGs):
        # For each HoG in list of test images
        # Similar to code from feed forward
        output_nodes = []
        for k in range(len(test_HoGs)):
            # Value at node equals summation of each input node multiplied by its weight
            total_sum = [0.0] * self.num_of_hidden
            for j in range(self.num_of_hidden):
                for i in range(test_HoGs[k].shape[0]):
                    total_sum[j] += (self.weights_ih[i][j] * test_HoGs[k][i])
                # Add bias
                total_sum[j] += self.bias_ih[j]
                # Use the ReLU activation function at the node
                self.hidden_node_value[j] = max(0, total_sum[j])

            # Value at output node found by the summation of each hidden node multiplied by its weight
            output_sum = 0.0
            for i in range(self.num_of_hidden):
                output_sum += self.hidden_node_value[i] * self.weights_ho[i]
            # Add bias
            output_sum += self.bias_ho
            # Append prediction made from sigmoid output function to final predictions list
            self.predictions.append(self.sigmoid_output_func(output_sum))
            output_nodes.append(self.sigmoid_func(output_sum))
        return (self.predictions, output_nodes)


# Used to prepare images and create HoGs for training and test images
def prepare(pathList):
    res = []
    for path in pathList:
        image = imread(path)
        image = makeGrayscale(image)
        x_gradient = xgradient(image)
        y_gradient = ygradient(image)
        (magnitudes, angles) = EdgeMagnitudesAndGradientAngles(x_gradient, y_gradient)
        #Next 2 lines of code were used to create the HoG Descriptor txt files
        #plt.imshow(magnitudes, cmap='gray')
        #plt.show()
        res.append(HoG(magnitudes, angles))
    return res

# Will be run in main
# Will create a multi layer perceptron with 250, 500, & 1000 hidden nodes
# Each will be its own mlp and provide predictions of its own
# Output Neuron values will be printed after predictions
def HumanDetection():
    TrainingImagesPaths = [r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Positive\crop001030c.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Positive\crop001034b.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Positive\crop001063b.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Positive\crop001070a.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Positive\crop001275b.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Positive\crop001278a.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Positive\crop001500b.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Positive\crop001672b.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Positive\person_and_bike_026a.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Positive\person_and_bike_151a.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Negative\00000003a_cut.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Negative\00000057a_cut.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Negative\00000090a_cut.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Negative\00000091a_cut.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Negative\00000118a_cut.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Negative\01-03e_cut.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Negative\no_person__no_bike_219_cut.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Negative\no_person__no_bike_258_Cut.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Negative\no_person__no_bike_259_cut.bmp",
                           r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Train_Negative\no_person__no_bike_264_cut.bmp"]
    TestImagesPaths = [r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Test_Positive\crop001008b.bmp",
                       r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Test_Positive\crop001028a.bmp",
                       r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Test_Positive\crop001045b.bmp",
                       r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Test_Positive\crop001047b.bmp",
                       r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Test_Positive\crop_000010b.bmp",
                       r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Test_Neg\00000053a_cut.bmp",
                       r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Test_Neg\00000062a_cut.bmp",
                       r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Test_Neg\00000093a_cut.bmp",
                       r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Test_Neg\no_person__no_bike_213_cut.bmp",
                       r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 2\Test_Neg\no_person__no_bike_247_cut.bmp"]
    print("Preparing Training Images...")
    TrainingImagesHoGs = prepare(TrainingImagesPaths)
    correctLabels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print("Preparing Testing Images...")
    TestingImagesHoGs = prepare(TestImagesPaths)

    mlp250 = MLP(TrainingImagesHoGs[0].shape[0], 250)
    print("Training MLP with 250 hidden nodes...")
    mlp250.train(TrainingImagesHoGs, correctLabels)
    print("Predicting Test Images with MLP of 250 hidden nodes...")
    (predictions250, outputs) = mlp250.predict(TestingImagesHoGs)
    print(predictions250)
    print(outputs)

    mlp500 = MLP(TrainingImagesHoGs[0].shape[0], 500)
    print("Training MLP with 500 hidden nodes...")
    mlp500.train(TrainingImagesHoGs, correctLabels)
    print("Predicting Test Images with MLP of 500 hidden nodes...")
    (predictions500, outputs) = mlp500.predict(TestingImagesHoGs)
    print(predictions500)
    print(outputs)

    mlp1000 = MLP(TrainingImagesHoGs[0].shape[0], 1000)
    print("Training MLP with 1000 hidden nodes...")
    mlp1000.train(TrainingImagesHoGs, correctLabels)
    print("Predicting Test Images with MLP of 1000 hidden nodes...")
    (predictions1000, outputs) = mlp1000.predict(TestingImagesHoGs)
    print(predictions1000)
    print(outputs)

if __name__=="__main__":
    HumanDetection()
