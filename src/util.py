import time
import uuid
from enum import Enum
from getpass import getpass

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
    except Exception:
        print("Failed to authenticate.")


def random_path(ext: str, dir=None, prefix=None):
    unique_id = str(uuid.uuid4())[:8]

    if prefix:
        filename = f"{prefix}_{unique_id}.{ext}"
    else:
        filename = f"{unique_id}.{ext}"

    if dir:
        filename = f"{dir}/{filename}"

    return filename


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

    match unit:
        case Time.NS:
            elapsed_time = sec * 1000000
        case Time.MS:
            elapsed_time = sec * 1000
        case Time.SECOND:
            elapsed_time = sec
        case _:
            elapsed_time = 0

    return elapsed_time, ret
