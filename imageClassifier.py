# This program was written by Jessica Devlin
# The program was written on 4/13/2020
# And last modified on 4/19/2020
# The purpose of this program is to classify images as one of four categories
# based on a dataset that I have collected of images of chihuahuas, hot dogs, human legs, and blueberry muffins
# The program uses the pyTorch libraries to perform training and evalutation on a custom classifier that is created,
# with the goal of then being able to accurately classify test images provided to it after training

# Import Libraries
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from textwrap import wrap
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import math

# Declare the settings with which to uniformly "normalize" images before training
image_settings = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Import images from training set and images from validation set to their respective dataset objects
training_set = datasets.ImageFolder("data/training", transform = image_settings)
validation_set = datasets.ImageFolder("data/validation", transform = image_settings)

# Read the datasets into dataloaders using torch dataloader library
training_loader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size =32, shuffle=True)

# Create a densenet model from torchvision.models library
# Allow the user to choose the control or variable mode (for experimental purposes)
# train_choice = input("Would you like to run in pretrained mode? (yes or no)\n")
# if train_choice == "no":
#    model = models.densenet161(pretrained=False)
# elif train_choice == "yes":
#    model = models.densenet161(pretrained=True)

model = models.densenet161(pretrained=True)

# Remove classification layer of the imported model
for param in model.parameters():
    param.requires_grad = False

# Create a custom classifier for the densenet model using torch.nn as nn library
classifier_input = model.classifier.in_features

# four categories in this classifier: chihuahuas, hot dogs, blueberry muffins, and legs
categories = 4
# define settings of RELU layers
classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, categories),
                           nn.LogSoftmax(dim=1))

# Set the classifier for the model to be the custom classifier
model.classifier = classifier

# Sets device equal to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move model to the device specified above (preferably GPU)
model.to(device)

# Set the error function using torch.nn as nn library
criterion = nn.NLLLoss()
# Set the optimizer function using Adam algorithm as optimization algorithm
optimizer = optim.Adam(model.classifier.parameters())

# Set number of training and testing rounds to be conducted on classifier
tests = 10
for test in range(tests):
    training_loss = 0
    validation_loss = 0
    accuracy = 0

    # Set the model to "train"
    model.train()
    counter = 0
    for inputs, labels in training_loader:
        # Move to GPU device
        inputs, labels = inputs.to(device), labels.to(device)
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        output = model.forward(inputs)
        # Set loss based on error function
        temp_loss = criterion(output, labels)
        # Backpropogation
        temp_loss.backward()
        # Update optimizer
        optimizer.step()
        # Add loss to overall training loss
        training_loss += temp_loss.item() * inputs.size(0)
        
        # Print training progress
        counter += 1
        print("Training round:", counter, "/", len(training_loader))

    # Set to evaluation mode
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in validation_loader:
            # Move to GPU device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            output = model.forward(inputs)
            # Set loss based on error function
            temp_loss = criterion(output, labels)
            # Add loss overall evaluation loss
            validation_loss += temp_loss.item() * inputs.size(0)
            
            # Get real percentages by reversing the log function of output
            output = torch.exp(output)
            # Get the most likely class of this round
            probability, class_guess = output.topk(1, dim=1)
            # Check how many guesses were correct
            equals = class_guess == labels.view(*class_guess.shape)
            # Calculate the mean accuracy and add it to total accuracy of this round
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Print evaluation progress
            counter += 1
            print("Validation round:", counter, "/", len(validation_loader))

    # Calculate loss of training and validation over all rounds
    training_loss = training_loss/len(training_loader.dataset)
    validation_loss = validation_loss/len(validation_loader.dataset)

    #  Print results
    print('\n')
    print('Accuracy of validation layer: ', accuracy/len(validation_loader))
    # Print loss to the 6th decimal place
    print('Round: {} \nTraining Loss: {:.6f} \nValidation Loss: {:.6f}'.format(test, training_loss, validation_loss))
    print('\n')


def process_image(image_path):
    # Load Image
    img = Image.open(image_path)

    # Get the dimensions of the image
    width, height = img.size
    
    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is 255px
    img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    
    # Get the dimensions of the new image size
    width, height = img.size
    
    # Set the coordinates to do a center crop of 224 x 224
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    
    # Turn image into numpy array
    img = np.array(img)
    
    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))
    
    # Make all values between 0 and 1
    img = img/255
    
    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis,:]
    
    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image


# Predicts a given image by forward propogating it through the specified model  
def predict(image, model):
    # EDIT: image.to(device) was used to fix input and weight being on different devices
    # Forward pass 
    output = model.forward(image.to(device))

    # Reverse the log function of our output
    output = torch.exp(output)
    
    # Return the most likely class of the and the probabilities that it is that class
    probabilities, classes = output.topk(1, dim=1)
    return probabilities.item(), classes.item()


# Displays image in pop-out window during testing
def show_image(image):
    # Convert image to numpy
    image = image.numpy()
    
    # Undo our transformation settings to display image
    image[0] = image[0] * 0.226 + 0.445
    
    # define figure and plot image
    fig = plt.figure(figsize=(25, 4))
    plt.imshow(np.transpose(image[0], (1, 2, 0)))
    # Show image in a pop-out window
    plt.show()


# Convert class index to corresponding name of class for readability
def convert_guess(class_guess):
    class_guess = int(class_guess)
    if class_guess == 0:
        class_name = "chihuahuas"
    elif class_guess == 1:
        class_name = "hot dogs"
    elif class_guess == 2:
        class_name = "legs"
    elif class_guess == 3:
        class_name = "blueberry muffins"
    return class_name

# method to round output
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


# Ensure model is switched to evaluation before prediction
model.eval()

mode = input("Would you like to run....\n 1: interactive mode?\n 2: automatic mode?\n 3: figure mode?\n")
mode = int(mode)

# interactive mode
if mode == 1:
    while True:
        # Prompt user for image path to be tested
        filepath = input("Please enter the file you would like to test (of the form ___.jpg): ")
        # Process Image before testing
        image = process_image("data/" + filepath)
        probability, class_guess = predict(image, model)
        # Pop-out window of image
        show_image(image)
        print("The model is ", probability*100, "% certain that the image is of the class", convert_guess(class_guess)) 

# automatic/batch mode
elif mode == 2:
    out = open("results.txt", "w") 
    folder = input("Please enter the name of the folder you would like to test (of the form ./folderName): ")
    # Use pathlib library to iterate through folder
    filepath = pathlib.Path(folder)
    for item in filepath.iterdir():
        # check that item is not a folder
        if item.is_file():
            # Process Image before testing
            image = process_image(item)
            probability, class_guess = predict(image, model)
            statement = "The model is ", probability*100, "% certain that the image", item, "is of the class", convert_guess(class_guess)
            # Cast statement to string
            statement = str(statement)
            print(statement)
            out.write(statement + '\n')

# figure mode
elif mode == 3:
    print("Model is evaluating test images and will produce a graphed result...")
    filepath = pathlib.Path("./data")

    # define figure
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(10, 10),
                            subplot_kw={'xticks': [], 'yticks': []})
    plt.rcParams["axes.titlesize"] = 8

    results = []
    confidences = []
    evaluated_images = []
    for item in filepath.iterdir():
        if item.is_file():
            image = process_image(item)
            probability, class_guess = predict(image, model)
            results.append(convert_guess(class_guess))
            confidences.append(probability*100)
            evaluated_images.append(image)

    for ax, result, confidence, image in zip(axs.flat, results, confidences, evaluated_images):
        ax.set_title("\n".join(wrap("Prediction: " + str(result) + " (" + str(truncate(confidence, 2)) + "% confidence)", 25)))

        image = image.numpy()
        # Denormalize to plot
        image[0] = image[0] * 0.226 + 0.445
    
        ax.imshow(np.transpose(image[0], (1, 2, 0)))

    fig.subplots_adjust(top=0.8)
    plt.savefig('results.jpg')
    plt.show()
        
