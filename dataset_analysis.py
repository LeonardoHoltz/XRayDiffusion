import matplotlib.pyplot as plt
import torch
from datamodules.chest_x_ray_dataset import get_dataloaders_by_label, get_dataloader_from_hf
from tqdm import tqdm

def show_images(images, labels, classes, n_images=5):
    plt.figure(figsize=(15, 5))
    for i in range(n_images):
        ax = plt.subplot(1, n_images, i + 1)
        image = images[i].permute(1, 2, 0)  # Convert from CxHxW to HxWxC
        ax.imshow(image, cmap='gray')
        ax.set_title(classes[labels[i].item()])
        ax.axis('off')
    plt.show()

def show_images_without_resize(classes, n_images):
    dataloaders  = get_dataloaders_by_label("AiresPucrs/chest-xray", 2, batch_size=1)

    i = 0
    plt.figure(figsize=(25, 18))
    for ((images_0, _), (images_1, _)) in zip(dataloaders[0], dataloaders[1]):
        
        # image class 0
        ax = plt.subplot(2, 5, i + 1)
        print(images_0[0].shape)
        image = images_0[0].permute(1, 2, 0)  # Convert from CxHxW to HxWxC
        ax.imshow(image, cmap='gray')
        ax.set_title(classes[0])
        ax.axis('off')
        
        # image class 1
        ax = plt.subplot(2, 5, i + 1 + 5)
        print(images_1[0].shape)
        image = images_1[0].permute(1, 2, 0)  # Convert from CxHxW to HxWxC
        ax.imshow(image, cmap='gray')
        ax.set_title(classes[1])
        ax.axis('off')
        
        i += 1
        
        if i == 5:
            plt.show()
            break

def check_min_image_size():
    dataloaders  = get_dataloader_from_hf("AiresPucrs/chest-xray", batch_size=1)
    min_h = float('inf')
    min_w = float('inf')
    for images, labels in tqdm(dataloaders):
        if images[0].shape[1] < min_h:
            min_h = images[0].shape[1]
        if images[0].shape[2] < min_w:
            min_w = images[0].shape[2]
    print(min_h)
    print(min_w)
    
        
def main():
    
    classes = {0: "Normal", 1: "Pneumonia"}
    check_min_image_size()
    #show_images_without_resize(classes, 10)
    

if __name__ == "__main__":
    main()