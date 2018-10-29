import torch  # Import Torch Deep Learning Framework
import ImageUtils
from DCGAN import DCGANGenerator, DCGANDiscriminator

print("   Optimal Material Generator using Generative Adversarial Networks   ")
print("                    Developed by ***REMOVED*** (BSc)                    ")
print("In fulfilment of Doctor of Engineering at the University of Nottingham")
print("----------------------------------------------------------------------")
print()

print("Running hardware checks...")

device = torch.device("cpu")
parallelProcessing = False

deviceCount = torch.cuda.device_count()
print("\tGPU devices available: " + str(deviceCount))

if torch.cuda.is_available():
    print("\t\tCUDA processing is available!")

    device = torch.device("cuda")

    if deviceCount > 1:
        print("\t\t\tMulti-GPU Parallel Processing enabled")
        parallelProcessing = True
        # For information on parallel processing with PyTorch:
        # https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
    else:
        print("\t\t\tMulti-GPU Parallel processing is unavailable")

else:
    print("\t\tCUDA processing is unavailable!")
    print("\t\t\tEnabled device 'CPU'")

# Do image loading and preprocesing
directory = "/run/media/***REMOVED***/***REMOVED***/InputData/Brian Atkinson - Mustafa - Asphalt Cores/18-1415/"
myCollection = ImageUtils.ImageCollection()
myCollection.loadImagesFromDirectory(directory)
myCollection.segmentImages()

# Initialise GAN
print("Initialising Generator Adversarial Network")
discriminator = DCGANDiscriminator(parallelProcessing)
generator = DCGANGenerator(parallelProcessing)
