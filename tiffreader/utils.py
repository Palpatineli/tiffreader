from typing import List, Optional
from os import path
from glob import iglob
import subprocess as sp
import re
import numpy as np
from libtiff import TIFF

def save_tiff(arr: np.ndarray, file_path: str) -> None:
    temp_img = TIFF.open(file_path, 'w')
    temp_img.write_image(arr)

def binary_search(seq, func, max_size):
    start, end = 0, max_size
    while end - start > 1:
        middle = (end + start) // 2
        if func(seq[middle]) > 0:
            start = middle
        else:
            end = middle
    return end

def tiffinfo(img_path: str, fields: List[str]) -> List[Optional[int]]:
    def _extract(text_str: str, find_str: str) -> Optional[int]:
        find_str_len = len(find_str)
        start_idx = text_str.find(find_str)
        if start_idx == -1:
            return None
        end_idx = text_str.find(' ', start_idx + find_str_len)
        end_idx = min(end_idx, text_str.find('\n', start_idx + find_str_len))
        return int(text_str[start_idx + find_str_len: end_idx])

    if path.isdir(img_path):
        img_path = next(iglob(path.join(img_path, "*.tif")))
    output = sp.check_output(['tiffinfo', '-0', img_path]).decode('utf-8')
    return [_extract(output, field) for field in fields]

def _tsplit(string, *delimiters):
    pattern = '|'.join(map(re.escape, delimiters))
    return re.split(pattern, string)

def _written_frame_count(tif_path: str) -> Optional[int]:
    basename = path.splitext(path.split(tif_path)[-1])[0]
    if "Frames" in basename:
        return int(next(x for x in _tsplit(basename, '-', '_') if "Frames" in x)[0: -6])
    else:
        return None
