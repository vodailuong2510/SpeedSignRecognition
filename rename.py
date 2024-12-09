import os
import cv2

def read_and_rename_images(image_folder, txt_file):
    i = 180
    for dirpath, dirnames, filenames in os.walk(image_folder):
        for filename in filenames:
            if filename.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(dirpath, filename)

                label = "13"  

                file_extension = os.path.splitext(filename)[1]
                new_filename = f"img_{i}{file_extension}"
                i += 1
                new_image_path = os.path.join(dirpath, new_filename)
                os.rename(image_path, new_image_path)

                # with open(txt_file, 'a') as f:
                #     f.write(f"{label}\n")


image_folder = r'C:\Users\vodai\Downloads\projects\SpeedLimitSignsClassification\data\Speed limit\80'
txt_file = r'C:\Users\vodai\Downloads\projects\SpeedLimitSignsClassification\labels.txt'
read_and_rename_images(image_folder, txt_file)
