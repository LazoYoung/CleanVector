import codecs
import os
import subprocess
import sys
import time
import uuid
from enum import Enum
from getpass import getpass

import requests
from PIL import Image
from huggingface_hub import login

token_path = ".token"


class Time(Enum):
    SECOND = 0
    MS = 1
    NS = 2


def get_token(path):
    with open(path, 'r') as file:
        token = file.read().strip()
    return token


def authenticate(token: str = None):
    """
    Login to huggingface account for model access.

    :param token: token Huggingface token
    """
    if not token:
        try:
            file = open(token_path, "r")
            token = file.readline()
        except FileNotFoundError:
            print("Your token is required to access the pretrained models.")
            print("Visit https://huggingface.co/ to create new token.")
            token = getpass(prompt="Your access token: ")
            authenticate(token)
            return

    try:
        login(token)
        try:
            # write token to file
            file = open(token_path, 'w')
            file.write(token)
        except Exception:
            print("Failed to save token to disk.")
    except Exception as e:
        print("Failed to authenticate.")
        raise e


def random_path(ext: str, dir=None, prefix=None):
    unique_id = str(uuid.uuid4())[:8]

    if prefix:
        filename = f"{prefix}_{unique_id}.{ext}"
    else:
        filename = f"{unique_id}.{ext}"

    if dir:
        filename = f"{dir}/{filename}"

    return filename


def read_images(dir_path, max=999) -> list[Image.Image]:
    """Load batch of images from a directory."""
    extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']
    images = []
    count = 0

    for filename in os.listdir(dir_path):
        path = os.path.join(dir_path, filename)
        if os.path.isfile(path) and filename.split(".")[-1].lower() in extensions:
            count += 1
            if count > max:
                break
            try:
                with Image.open(os.path.join(dir_path, filename)) as img:
                    img.load()
                    img = img.convert("RGB")
                    images.append(img)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return images


def measure(func, unit: Time) -> tuple[float, any]:
    """
    Get amount of time elapsed.
    :param func: function to be measured
    :param unit: which time unit to use
    :return: elapsed time and returned-value from func
    """
    start_time = time.perf_counter()
    ret = func()
    end_time = time.perf_counter()

    sec = (end_time - start_time)

    if unit == Time.NS:
        elapsed_time = sec * 1000000
    elif unit == Time.MS:
        elapsed_time = sec * 1000
    elif unit == Time.SECOND:
        elapsed_time = sec
    else:
        elapsed_time = 0

    return elapsed_time, ret


def download_file(url, destination=None):
    """
    Download a file from a given URL.

    Args:
        url (str): The URL of the file to download
        destination (str, optional): The local path to save the file.
                                     If None, uses the filename from the URL.

    Returns:
        str: The local path where the file was saved
    """
    try:
        # Send a GET request to download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # If no destination is provided, extract filename from URL
        if destination is None:
            destination = os.path.basename(url.split('?')[0])

        # Ensure the directory exists
        os.makedirs(os.path.dirname(destination) or '.', exist_ok=True)

        if os.path.isfile(destination):
            return None

        # Write the file
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"File download complete: {destination}")
        return destination

    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)


def detect_file_encoding(file_path):
    """
    Detect the encoding of a file.

    Args:
        file_path (str): Path to the file

    Returns:
        str: Detected encoding
    """
    encodings_to_try = [
        'utf-8', 'latin-1', 'cp1252', 'iso-8859-1',
        'ascii', 'utf-16', 'big5', 'shift_jis'
    ]

    for encoding in encodings_to_try:
        try:
            with codecs.open(file_path, 'r', encoding=encoding) as f:
                f.read()
            return encoding
        except (UnicodeDecodeError, LookupError):
            continue

    return 'utf-8'  # Default fallback


def convert_file_encoding(input_file, target_encoding='utf-8'):
    """
    Convert file to UTF-8 encoding.

    Args:
        input_file (str): Path to the input file
        output_file (str, optional): Path to save the converted file
        target_encoding (str, optional): Target encoding (default UTF-8)

    Returns:
        str: Path to the converted file
    """
    # Detect source encoding
    source_encoding = detect_file_encoding(input_file)
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}{ext}"

    try:
        # Read with source encoding
        with codecs.open(input_file, 'r', encoding=source_encoding) as source_file:
            content = source_file.read()

        os.remove(input_file)

        # Write with target encoding
        with codecs.open(output_file, 'w', encoding=target_encoding) as target_file:
            target_file.write(content)

        print(f"Converted {input_file} from {source_encoding} to {target_encoding}")
        return output_file

    except Exception as e:
        print(f"Error converting file encoding: {e}")
        return input_file


def run_script_with_args(script_path, args=None):
    """
    Run a downloaded script with optional additional arguments.

    Args:
        script_path (str): Path to the script to run
        args (list, optional): Additional command-line arguments
    """
    try:
        # Make the script executable (for Unix-like systems)
        try:
            current_mode = os.stat(script_path).st_mode
            new_mode = current_mode | 0o111
            os.chmod(script_path, new_mode)
        except Exception as e:
            print(f"Failed to make {script_path} executable: {e}")

        # Prepare the command to run
        command = [script_path]
        if args:
            command.extend(args)

        # Run the script
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        # Print output
        print("Script Output:")
        print(result.stdout)

        # Print any error output
        if result.stderr:
            print("Error Output:")
            print(result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
