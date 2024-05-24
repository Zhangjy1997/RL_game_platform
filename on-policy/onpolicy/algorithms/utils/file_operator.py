import json
import os
import shutil
import glob
import fnmatch

def check_files_exist(directory, match_str):
    has_str = False

    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, match_str):
            has_str = True

        if has_str:
            break

    return has_str

def copy_files_with_suffix(source_dir, dest_dir, endstr):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for file in os.listdir(source_dir):
        if file.endswith(endstr):
            src_file = os.path.join(source_dir, file)
            dest_file = os.path.join(dest_dir, file)

            shutil.copy(src_file, dest_file)
            print(f"Copied: {src_file} to {dest_file}")

def delete_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"The folder '{folder_path}' has been deleted.")
    else:
        print(f"The folder '{folder_path}' does not exist or is not a directory.")

def delete_files_with_suffix(folder_path, end_str):
    pattern = os.path.join(folder_path, f'*{end_str}')

    files = glob.glob(pattern)

    for file in files:
        os.remove(file)
        print(f'Deleted: {file}')

def create_new_path(path1, path2):
    parts1 = [part for part in path1.split('/') if part]
    parts2 = [part for part in path2.split('/') if part]
    # print(parts1)
    # print(parts2)

    common_folder = None
    for part in parts1:
        if part in parts2[-1]:
            common_folder = part
            break

    if not common_folder:
        raise ValueError("No common folder found.")

    index = parts1.index(common_folder) + 1
    sub_path = '/'.join(parts1[index:])

    new_path = os.path.join(path2, sub_path)

    return new_path

def update_json(file_path, key, value):
    # Check if the file exists
    if os.path.exists(file_path):
        # If the file exists, read and update its content
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except json.JSONDecodeError:
            print(f"Error parsing JSON: {file_path}")
            data = {}
    else:
        # If the file does not exist, create a new dictionary
        data = {}

    # Update or add the key-value pair
    data[key] = value

    # Write the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def rename_file(old_name, new_name):
    """
    Renames a file from old_name to new_name.

    Args:
    old_name (str): The current name of the file.
    new_name (str): The new name for the file.

    Returns:
    bool: True if the file was successfully renamed, False otherwise.
    """
    import os

    try:
        os.rename(old_name, new_name)
        print("rename from {} to {}".format(old_name, new_name))
        return True
    except FileNotFoundError:
        print(f"Error: {old_name} does not exist.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
