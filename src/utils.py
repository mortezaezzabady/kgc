import torch
import os

CUDA = torch.cuda.is_available()

def save_model(model, name, epoch, folder_name):
    print("Saving Model")
    torch.save(model.state_dict(), os.path.join(folder_name, 'trained_' + str(epoch) + '.pth'))
    print("Done saving Model")