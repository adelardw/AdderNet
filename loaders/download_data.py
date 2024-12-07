import requests
from tqdm.notebook import tqdm
import tarfile
import zipfile
import os


def download_file(url, file_name):
    """
    Downloads a file from the specified URL
    and saves it locally with the given file name.

    Parameters:
        url (str): The URL of the file to be downloaded.
        file_name (str): The name of the file where
            the downloaded content will be saved.

    Example:
    >>> download_file("https://example.com/file.zip", "file.zip")
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        with open(file_name, "wb") as file, tqdm(
            desc=file_name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    bar.update(len(chunk))
        if downloaded_size == total_size:
            print(f"File '{file_name}' downloaded successfully.")
        else:
            remaining_size = total_size - downloaded_size
            print(f"Download incomplete. Downloaded: {downloaded_size} bytes,"
                  f" Remaining: {remaining_size} bytes.")
    else:
        print("Failed to download file. Check the URL or your connection.")


def extract_archive(file_path, extract_to=None, delete_archive=False):
    """
    Extracts the contents of an archive file to a specified directory.
    Supports `.tar`, `.tar.gz`, `.tgz`, and `.zip` formats.
    If `extract_to` is not provided,the contents are extracted to a directory
    with the same name as the file (without extension).
    Optionally, the archive file can be deleted after extraction.

    Parameters:
        file_path (str): Path to the archive file to extract.
        extract_to (str, optional): Directory for the extracted contents.
            Defaults to a folder with the same name as the archive file.
        delete_archive (bool, optional): If `True`,
            deletes the archive file after extraction.

    Example:
    >>> extract_archive("file.zip",
    extract_to="extracted_files",
    delete_archive=True)
    """
    if extract_to is None:
        extract_to = os.path.splitext(file_path)[0]

    try:
        if file_path.endswith(".tar.gz") or \
                file_path.endswith(".tgz") or \
                file_path.endswith(".tar"):
            with tarfile.open(file_path, "r:*") as archive:
                members = archive.getmembers()
                with tqdm(total=len(members),
                          desc="Extracting",
                          unit="file") as bar:
                    for member in members:
                        archive.extract(member, path=extract_to)
                        bar.update(1)
                print(f"Extracted '{file_path}' to '{extract_to}'")

        elif file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as archive:
                members = archive.namelist()
                with tqdm(total=len(members),
                          desc="Extracting",
                          unit="file") as bar:
                    for member in members:
                        archive.extract(member, path=extract_to)
                        bar.update(1)
                print(f"Extracted '{file_path}' to '{extract_to}'")
        else:
            print(f"Unsupported file format: {file_path}")
            return

        if delete_archive:
            os.remove(file_path)
            print(f"Archive '{file_path}' has been deleted after extraction.")

    except Exception as e:
        print(f"An error occurred while extracting the archive: {e}")
