from typing import Union, Tuple, List, Callable, Optional
from os import path
import subprocess as sp
import re
import numpy as np
from libtiff import libtiff, TIFF
# noinspection PyUnresolvedReferences
from libtiff.libtiff_ctypes import PLANARCONFIG_CONTIG, COMPRESSION_NONE, PLANARCONFIG_SEPARATE, suppress_warnings

__all__ = ["save_tiff", "tiffinfo", "TiffReader"]

suppress_warnings()


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

_max_size = 65535

def tiffinfo(img_path: str, fields: List[str]) -> List[Optional[int]]:
    def _extract(text_str: str, find_str: str) -> Optional[int]:
        find_str_len = len(find_str)
        start_idx = text_str.find(find_str)
        if start_idx == -1:
            return None
        end_idx = text_str.find(' ', start_idx + find_str_len)
        end_idx = min(end_idx, text_str.find('\n', start_idx + find_str_len))
        return int(text_str[start_idx + find_str_len: end_idx])

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


class TiffReader(object):
    _shape = None
    _length: Optional[int] = None
    _read_func: Optional[Callable] = None
    _template_frame: Optional[np.ndarray] = None

    def __init__(self, tiff_obj: TIFF) -> None:
        self._tiff_ptr: TIFF = tiff_obj

    @classmethod
    def open(cls, filename: Union[str, bytes], mode: str = 'r'):
        return TiffReader(TIFF.open(filename, mode))

    def __iter__(self):
        yield self.read_current()
        while not libtiff.TIFFLastDirectory(self._tiff_ptr):
            libtiff.TIFFReadDirectory(self._tiff_ptr)
            yield self.read_current()
        yield self.read_current()

    def __getitem__(self, item: int) -> np.ndarray:
        libtiff.TIFFSetDirectory(self._tiff_ptr, item)
        return self.read_current()

    def __del__(self):
        self._tiff_ptr.close()

    def seek(self, index: int):
        libtiff.TIFFSetDirectory(self._tiff_ptr, index)

    @property
    def length(self) -> int:
        if self._length is None:
            self._length = binary_search(np.arange(_max_size),
                                         lambda x: libtiff.TIFFSetDirectory(self._tiff_ptr, x), _max_size)
            self.seek(0)
        return self._length

    @property
    def shape(self) -> Tuple[int, int]:
        if self._shape is None:
            self._shape = (self._tiff_ptr.GetField('ImageLength'), self._tiff_ptr.GetField('ImageWidth'))
        # noinspection PyTypeChecker
        return self._shape

    def read_current(self) -> np.ndarray:
        """ Read image from TIFF and return it as an array. """
        if self._read_func is None or self._template_frame is None:
            get_field = self._tiff_ptr.GetField
            samples_pp = get_field('SamplesPerPixel')  # this number includes extra samples
            samples_pp = 1 if samples_pp is None else samples_pp
            bits = get_field('BitsPerSample')
            sample_format = get_field('SampleFormat')
            planar_config = get_field('PlanarConfig')
            if planar_config is None:  # default is contiguous
                planar_config = PLANARCONFIG_CONTIG
            compression = get_field('Compression')
            compression = None if compression == COMPRESSION_NONE else compression
            self._read_func = libtiff.TIFFReadEncodedStrip
            dtype = self._tiff_ptr.get_numpy_type(bits, sample_format)
            if samples_pp == 1:  # only 2 dimensions
                self._template_frame = np.empty(self.shape, dtype)
            elif planar_config == PLANARCONFIG_CONTIG:
                self._template_frame = np.empty((*self.shape, samples_pp), dtype)
            elif planar_config == PLANARCONFIG_SEPARATE:
                self._template_frame = np.empty((samples_pp, *self.shape), dtype)
            else:
                raise IOError("Unexpected PlanarConfig = %d" % planar_config)

        # noinspection PyTypeChecker
        arr = np.empty_like(self._template_frame)
        # actually read strips
        read_func = self._read_func
        pointer = arr.ctypes.data
        size = arr.nbytes
        tiff_ptr = self._tiff_ptr
        for strip in range(libtiff.TIFFNumberOfStrips(tiff_ptr).value):
            elem = read_func(tiff_ptr, strip, pointer, max(size, 0)).value   # type: ignore
            pointer += elem
            size -= elem
        return arr
