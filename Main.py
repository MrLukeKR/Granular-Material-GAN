import torch

print("   Optimal Material Generator using Generative Adversarial Networks   ")
print("                    Developed by ***REMOVED*** (BSc)                    ")
print("In fulfilment of Doctor of Engineering at the University of Nottingham")
print("----------------------------------------------------------------------")
print()

print("Running hardware checks...")

deviceCount = torch.cuda.device_count()
print("\tGPU devices available: " + str(deviceCount))

if torch.cuda.is_available():
    print("\tCUDA processing is available!")
else:
    print("\tCUDA processing is unavailable!")