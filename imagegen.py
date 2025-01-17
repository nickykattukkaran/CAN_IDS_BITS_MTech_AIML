import pandas as pd
import numpy as np
import os
from PIL import Image
from  Preprocessing import Process, load_data
from DirCreation import directory_creation

# Load the datasets
attack_free_file = r"CSV_Converted/Attack_free_dataset.csv"
dos_attack_file = r"CSV_Converted/DoS_attack_dataset.csv"
fuzzy_attack_file = r"CSV_Converted/Fuzzy_attack_dataset.csv"
impersonation_attack_file = r"CSV_Converted/Impersonation_attack_dataset.csv"

attack_free = load_data(attack_free_file)
dos_attack = load_data(dos_attack_file)
fuzzy_attack = load_data(fuzzy_attack_file)
impersonation_attack = load_data(impersonation_attack_file)


def generate_binary_images(df, output_folder):
    """
    Generates binary images from CAN intrusion detection data.

    Args:
        data: Pandas DataFrame containing preprocessed data with columns:
            ID, RemoteFrame, DLC, Payload, TimeInterval
        output_folder: Path to the output folder for storing images.
    """

    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # Convert all columns to string
    df = df.astype(str)
    df = df.drop(columns=['Index'])
    print(df.dtypes)

    df['combined_output'] = df['ID']+df['RemoteFrame'] + df['DLC']+df['Payload'] + df['TimeInterval']

    df = df.drop(columns=['ID','RemoteFrame','DLC','Payload','TimeInterval'])
    #print(new_df)
    #df.to_csv('output1.csv', index=False) 

    create_binary_images(df, output_folder)


# Define the function to combine consecutive 94 rows and create 94x94 binary images
def create_binary_images(df, output_folder):
    #os.makedirs(output_folder, exist_ok=True)
    num_rows, num_cols = df.shape
    print(num_rows, num_cols)
    num_images = num_rows // 94
    train_img_num = int(num_images*0.8)
    print(f"Total Number of Images in {output_folder}: "+str(num_images)+"   from 0 to "+str(num_images-1))
    print(f"Number of  Training Images in {output_folder}: "+str(train_img_num)+" from 0 to "+str(train_img_num))
    print(f"Number of  Testing Images in {output_folder}: "+str(num_images-train_img_num)+" from "+ str(train_img_num+1)+ "to "+str(num_images-1))
    
    for i in range(num_images):
        # Combine 94 consecutive rows
        combined_rows = df.iloc[i*94:(i+1)*94].values.flatten()
        
        # Reshape to 94x94
        binary_image = combined_rows.reshape(94, 1).astype(str)
        # print(binary_image)
        # Convert the binary strings to a numpy array of 0s and 1s
        binary_image_data = np.array([[int(bit) for bit in row[0]] for row in binary_image])
        # print(binary_image_data)

        # Display binary image data row by row
        # for row in binary_image_data:
        #     print(' '.join(str(cell) for cell in row))

        # print(binary_image_data.shape)

        # Convert the numpy array to an image
        image = Image.fromarray(binary_image_data.astype(np.uint8) * 255)  # 0 = black, 1 = white

        # Save the image or display it
        #image.show()  # To display
        if i<= train_img_num:
            image.save(os.path.join(f'{root_folder}/{output_folder}/train', f"{output_folder}_{i}.jpg"))
        else:
            image.save(os.path.join(f'{root_folder}/{output_folder}/test', f"{output_folder}_{i}.jpg"))

    print(f'The Images of {output_folder} are successfully Stored in {root_folder}/{output_folder}')

# Create the folder structure to store the images
root_folder, subfolders = directory_creation()

df_attack_free = Process(attack_free, "Attack Free")
df_dos_attack = Process(dos_attack, "DoS Attack")
df_fuzzy_attack = Process(fuzzy_attack, "Fuzzy Attack")
df_imper_attack = Process(impersonation_attack, "Impersonation Attack")


generate_binary_images(df_attack_free, subfolders[0])
generate_binary_images(df_dos_attack, subfolders[1])
generate_binary_images(df_fuzzy_attack, subfolders[2])
generate_binary_images(df_imper_attack, subfolders[3])

attack_type = ["Attack_free", "Dos_Attack", "Fuzzy_Attack", "Impersonate_Attack"]
dataframes_list = [df_attack_free, df_dos_attack, df_fuzzy_attack, df_imper_attack]
totalimg = []
train = []
test = []

def train_test_count():
    for i in range(len(attack_type)):
        num_rows, _ = dataframes_list[i].shape
        num_images = num_rows // 94
        train_img = (int(num_images*0.8))
        test_img = num_images - train_img
        totalimg.append(num_images)
        train.append(train_img)
        test.append(test_img)
        print(f"The Total images in {attack_type[i]}: "+str(totalimg[i])+"   from 0 to "+str(totalimg[i]-1))
        print(f"The Train images in {attack_type[i]}: "+str(train[i])+" from 0 to "+str(train[i]))
        print(f"The Test images in {attack_type[i]}: "+str(test[i])+" from "+ str(train[i]+1)+ " to "+str(totalimg[i]-1))
    
        print("*"*50)
    #return attack_type, totalimg, train, test

#Details of generated Total Images, Trainimages and Test images 
train_test_count()