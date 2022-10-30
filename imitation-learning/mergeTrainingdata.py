import os
import pickle
import time
import random
import shutil

input_images = []
input_data = []

combined_data = []
combined_image = []
shuffled_list = []

BATCH_SIZE = 256

def merge(image_list, data_list, output_dir):

    for i in range(len(image_list)):
        dataname = data_list[i]
        imagename = image_list[i]

        print("Processing filename: " + dataname)
        print("Processing imagename: " + imagename)

        individual_data, individual_image = [], []
        # Store in data from each file into a sub list
        with open(dataname, 'rb') as of:
            individual_data = pickle.load(of)
        with open(imagename, 'rb') as f:
            individual_image = pickle.load(f)

        for data in individual_data:
            combined_data.append(data)
        for image in individual_image:
            combined_image.append(image)
    # Now all data has been appended to combined arrays - 1 million data

    # We have a list of sequential numbers from 0 to len - 1 of total data
    for i in range(len(combined_data)):
        shuffled_list.append(i)

    # Shuffled order of the list is here. Use the index to get data and image tuples.
    random.shuffle(shuffled_list)
    print("LENGTH OF COMBINED DATA: " + str(len(combined_data)))
    # Save multiple shuffled data/tuple files of size 256 BS.

    data_seen = 0
    file_counter = 1
    intermediate_data = []
    for shuffled_index in shuffled_list:
        intermediate_data.append((combined_image[shuffled_index][0], 
                                       combined_image[shuffled_index][1],
                                       combined_data[shuffled_index]))
        data_seen += 1

        if data_seen == BATCH_SIZE:
            data_seen = 0
            # Make file and directory names
            file_name = "rgb_sem_data"+str(file_counter)+".pkl"
            file_counter += 1
            output_file = os.path.normpath(os.path.join(output_dir,file_name))

            with open(output_file, "wb") as of:
                for items in intermediate_data:
                    pickle.dump(items, of)
            intermediate_data = []

    # index = 0
    # for i in range(int(len(combined_data)/BATCH_SIZE)):
    #     intermediate_data = []
    #     for index in range(index+BATCH_SIZE):
    #         shuffled_index = shuffled_list[index]
    #         intermediate_data.append((combined_image[shuffled_index][0], 
    #                                   combined_image[shuffled_index][1],
    #                                   combined_data[shuffled_index]))

    #     # Make file and directory names
    #     file_name = "rgb_sem_data"+str(i)+".pkl"
    #     output_file = os.path.normpath(os.path.join(output_dir,file_name))

    #     with open(output_file, "wb") as of:
    #         for items in intermediate_data:
    #             pickle.dump(items, of)
    #     index += BATCH_SIZE

    # Save final concatenated data files 
    # output_data_file = os.path.normpath(os.path.join(os.getcwd(), "1million_data.pkl")) 
    # output_image_file = os.path.normpath(os.path.join(os.getcwd(), "1million_image.pkl"))      
    # with open(output_data_file, "wb") as of:
    #     for items in combined_data:
    #         pickle.dump(items,of)
    # with open(output_image_file, "wb") as f:
    #     for items in combined_image:
    #         pickle.dump(items,f)
    print("Number of training data points: " + str(len(combined_data)))
    #print("Finished with size: " + str(os.path.getsize(output_image_file)))


def main():

    # Create output directory and clean if previously existing.
    output_dir_path = os.path.normpath(os.path.join(os.getcwd(), "pickle_batches"))
    if os.path.exists(output_dir_path):
        shutil.rmtree(output_dir_path) 
    os.makedirs(output_dir_path)

    data_list = []
    image_list = []
    for root, dirs, files in os.walk(os.getcwd(), topdown=True):
        for file_names in files:
            if file_names.endswith('.pkl') and 'training' not in file_names and "million" not in file_names:
                file_path = str(os.path.normpath(os.path.join(root, file_names)))
                #print("FP: " + str(file_path))
                if "data" in file_path.split("/")[-1]:
                    data_list.append(file_path)
                elif "images" in file_path.split("/")[-1]:
                    image_list.append(file_path)
    list.sort(data_list)
    list.sort(image_list)

    for files in data_list:
        print(files)
    for files in image_list:
        print(files)

    merge(image_list, data_list, output_dir_path)
if __name__ == "__main__":
    main()




   