import pickle
import os
import re

import requests


def load_inverted_index(path):
    ret_dict = {}
    with open(os.path.join(path, 'inverted_idx') + '.pkl', 'rb') as f:
        while True:
            try:
                 pair = pickle.load(f)
                 ret_dict[pair[0]] = pair[1][0]
            except:
                return ret_dict

def save_list(obj, name, path):
    """
    This function save an object as a pickle.
    :param path:
    :param obj: object to save
    :param name: name of the pickle file.
    :return: -
    """

    with open(os.path.join(path, name) + '.pkl', 'wb') as f:
        to_be_told = f.tell()
        for pair in obj:
            pickle.dump(pair, f, pickle.HIGHEST_PROTOCOL)
        return to_be_told


def load_list(name, path, offset, chunk_length=0):
    """
    This function will load a pickle file
    :param path:
    :param name: name of the pickle file
    :return: loaded pickle file
    """
    ret = []
    with open(os.path.join(path, name) + '.pkl', 'rb') as f:
        f.seek(offset)
        if chunk_length == 0:
            while True:
                try:
                    ret.append(pickle.load(f))
                except:
                    return ret
        for i in range(chunk_length):
            try:
                ret.append(pickle.load(f))
            except:
                return ret, f.tell()
        return ret, f.tell()


def load_dict(name, path):
    """
    This function will load a pickle file
    :param path:
    :param name: name of the pickle file
    :return: loaded pickle file
    """
    ret_dict = {}
    with open(os.path.join(path, name) + '.pkl', 'rb') as f:
        while True:
            try:
                 pair = pickle.load(f)
                 ret_dict[pair[0]] = pair[1]
            except:
                return ret_dict


def save_dict(obj, name, path):
    """
    This function save an object as a pickle.
    :param path:
    :param obj: object to save
    :param name: name of the pickle file.
    :return: -
    """
    with open(os.path.join(path, name) + '.pkl', 'wb') as f:
        to_be_told = f.tell()
        for pair in obj.items():
            pickle.dump(pair, f, pickle.HIGHEST_PROTOCOL)


# THIS PART HAS BEEN TAKEN FROM THE COURSE'S REPOSITORY

__fid_ptrn = re.compile(
    "(?<=/folders/)([\w-]+)|(?<=%2Ffolders%2F)([\w-]+)|(?<=/file/d/)([\w-]+)|(?<=%2Ffile%2Fd%2F)([\w-]+)|(?<=id=)([\w-]+)|(?<=id%3D)([\w-]+)")
__gdrive_url = "https://docs.google.com/uc?export=download"


def download_file_from_google_drive(url, destination):
    m = __fid_ptrn.search(url)
    if m is None:
        raise ValueError(f'Could not identify google drive file id in {url}.')
    file_id = m.group()
    session = requests.Session()

    response = session.get(__gdrive_url, params={'id': file_id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(__gdrive_url, params=params, stream=True)

    _save_response_content(response, destination)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def unzip_file(file_path, target_dir):
    with zipfile.ZipFile(file_path, 'r') as z:
        z.extractall(target_dir)

