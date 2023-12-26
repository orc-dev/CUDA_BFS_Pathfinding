"""
File Name:    randmap.py
Author:       Xin Cai
Email:        xcai72@wisc.edu
Date:         Nov.24 2023

Description:  This prgram generate a random map by using the Perlin noise.
              The generated map can be either .txt or .csv file.

command:      python3 randmap.py 
              
Course:       ME 759
Instructor:   Prof. Dan Negrut 
"""
import noise
import numpy as np
import csv
import os


def generate_map(width, height, scale=50.0, threshold=0.5):
    """
    Use the Perlin noise function to generate map data.
    """
    map_data = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            nx = x / scale
            ny = y / scale
            noise_value = noise.pnoise2(nx, ny, octaves=1)

            if noise_value > threshold:
                map_data[y][x] = 1  # obstacle
            else:
                map_data[y][x] = 0  # open space

    return map_data


def write_map_to_csv(map_data, filename):
    """
    Write map data to a CSV file with column names.
    """
    # Add column names
    column_names = [str(i) for i in range(len(map_data[0]))]

    with open(filename, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        writer.writerows(map(int, row) for row in map_data)

    # Get the size of the file
    file_size = os.path.getsize(filename)
    print(f"Writing to {filename} is done. File size: {file_size} bytes.")


def write_map_to_txt(map_data, filename):
    """
    Write map data to specified file.
    """
    with open(filename, "w") as file:
        for row in map_data:
            line = ''.join(str(int(cell)) for cell in row)
            file.write(line + "\n")


# python3 randmap.py
def main():
    # small testing map
    #map_data = generate_map(width=64, height=64, scale=5.0, threshold=0.2)
    # sparse map
    #map_data = generate_map(width=512, height=512, scale=20.0, threshold=0.2)

    # dense map
    #map_data = generate_map(width=512, height=512, scale=5.0, threshold=0.12)

    txt_file = '../maps/map_64.txt'
    write_map_to_txt(map_data, txt_file)

    csv_file = '../maps/map_64.csv'
    write_map_to_csv(map_data, csv_file)


if __name__ == '__main__':
    main()
