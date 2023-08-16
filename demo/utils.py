import os

import wget


def download_file(url, dest_dir):
    # Extract the file name from the URL
    file_name = os.path.basename(url)
    dest_path = os.path.join(dest_dir, file_name)

    # Check if the file already exists in the destination directory
    if not os.path.exists(dest_path):
        wget.download(url, dest_path)
    else:
        print(f"{file_name} already exists. Skipping download.")


def subset_jsonl(input_file, output_file, num_lines):
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for _ in range(num_lines):
            line = f_in.readline()
            if not line:
                break  # Exit if the file has fewer lines than expected
            f_out.write(line)
