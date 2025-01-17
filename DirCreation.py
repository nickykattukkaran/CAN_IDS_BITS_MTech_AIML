import os

def directory_creation():
    # Root folder
    root_folder = "Genimage"

    # Subfolders
    subfolders = ["Attack_free", "Dos_Attack", "Fuzzy_Attack", "Impersonate_Attack"]
    sub_subfolders = ["train", "test"]

    # Create root folder if it doesn't exist
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    # Create subfolders and sub-subfolders
    for subfolder in subfolders:
        for sub_subfolder in sub_subfolders:
            path = os.path.join(root_folder, subfolder, sub_subfolder)
            os.makedirs(path, exist_ok=True)

    print(f"Folder structure created successfully under {root_folder}")

    return root_folder, subfolders
