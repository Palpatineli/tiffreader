from typing import Union, Tuple, Callable
from pathlib import Path
import numpy as np
from libtiff import libtiff, TIFF
from libtiff.libtiff_ctypes import PLANARCONFIG_CONTIG, PLANARCONFIG_SEPARATE, COMPRESSION_NONE, suppress_warnings
from .utils import binary_search

suppress_warnings()
_max_size = 65535

class TiffFolderReader(object):
    _path: Path
    _name: str
    _channel: int
    _idx: int
    _length: int
    shape: Tuple[int, int]
    length: int
    _read_func: Callable
    _template_frame: np.ndarray

    def __init__(self, folder: Path, file_name: str, channel: int):
        self._path = folder
        self._name = file_name
        self._channel = channel
        self._idx = 0
        tiff_ptr = TIFF.open(self._path.joinpath(self.frame_name(1)), 'r')
        self.shape = (tiff_ptr.GetField("ImageLength"), tiff_ptr.GetField("ImageWidth"))
        self.length = binary_search(np.arange(_max_size), self.exists, _max_size) - 1

    def exists(self, index: int) -> bool:
        return self._path.joinpath(self.frame_name(index)).exists()

    @classmethod
    def open(cls, folder_name: Union[str, bytes, Path]):
        folder = folder_name if isinstance(folder_name, Path) else Path(str(folder_name))
        if not folder.exists():
            raise FileNotFoundError
        if not folder.is_dir():
            raise NotADirectoryError
        file_name = next(iter(folder.glob("*.env"))).stem
        channel = int(next(iter(folder.glob(f"{file_name}_Cycle00001_Ch*_000001.ome.tif"))).stem[27])
        return cls(folder, file_name, channel)

    def frame_name(self, idx: int) -> str:
        return f"{self._name}_Cycle00001_Ch{self._channel}_{idx + 1:06d}.ome.tif"

    def __getitem__(self, idx: int) -> np.ndarray:
        self.seek(idx)
        return self.read_current()

    def __iter__(self):
        while self._idx < self.length:
            yield self.read_current()
            self._idx += 1

    def seek(self, index: int):
        if index < 0 or index >= self.length:
            raise IndexError(f"out of bounds! {index} outside of [1, {self.length}]")
        self._idx = index

    def read_current(self) -> np.ndarray:
        """ Read image from TIFF and return it as an array. """
        file = self._path.joinpath(self.frame_name(self._idx))
        tiff_ptr = TIFF.open(file, 'r')
        if not (hasattr(self, "_read_func") and hasattr(self, "_template_frame")):
            get_field = tiff_ptr.GetField
            samples_pp = get_field('SamplesPerPixel')  # this number includes extra samples
            samples_pp = 1 if samples_pp is None else samples_pp
            bits = get_field('BitsPerSample')
            sample_format = get_field('SampleFormat')
            planar_config = get_field('PlanarConfig')
            if planar_config is None:  # default is contiguous
                planar_config = PLANARCONFIG_CONTIG
            compression = get_field('Compression')
            compression = None if compression == COMPRESSION_NONE else compression
            self._read_func = libtiff.TIFFReadEncodedStrip  # type: ignore
            dtype = tiff_ptr.get_numpy_type(bits, sample_format)
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
        pointer = arr.ctypes.data
        size = arr.nbytes
        for strip in range(libtiff.TIFFNumberOfStrips(tiff_ptr).value):
            elem = self._read_func(tiff_ptr, strip, pointer, max(size, 0)).value   # type: ignore
            pointer += elem
            size -= elem
        return arr
