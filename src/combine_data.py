import os
import glob
import tqdm 
import multiprocessing

input_path = "/mlcv1/Datasets/NvidiaAICityChallenge/NvidiaAIC2024/FishEye8K"
output_path = "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/resplited_data"

def copy_data(src, dst):
    os.system("cp {} {}".format(src, dst))

def combine_data(input_path, output_path):
    all_images = glob.glob(os.path.join(input_path,"train/images/*")) + glob.glob(os.path.join(input_path,"test/images/*"))
    all_labels = glob.glob(os.path.join(input_path,"train/labels/*")) + glob.glob(os.path.join(input_path,"test/labels/*"))
    all_images.sort()
    all_labels.sort()
    print(len(all_images), len(all_labels))
    #split data into train and val 80-20
    train_images = all_images[:int(len(all_images)*0.8)]
    train_labels = all_labels[:int(len(all_labels)*0.8)]
    val_images = all_images[int(len(all_images)*0.8):]
    val_labels = all_labels[int(len(all_labels)*0.8):]
    print(len(train_images), len(train_labels))
    print(len(val_images), len(val_labels))
    #copy to new folder
    os.makedirs(os.path.join(output_path, "train/images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "train/labels"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "val/images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "val/labels"), exist_ok=True)

    for i in tqdm.tqdm(range(len(train_images))):
        copy_data(train_images[i], os.path.join(output_path, "train/images"))
        copy_data(train_labels[i], os.path.join(output_path, "train/labels"))
    for i in tqdm.tqdm(range(len(val_images))):
        copy_data(val_images[i], os.path.join(output_path, "val/images"))
        copy_data(val_labels[i], os.path.join(output_path, "val/labels"))

    print("Done")

   
combine_data(input_path, output_path)



