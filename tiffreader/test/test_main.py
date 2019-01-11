from pkg_resources import Requirement, resource_filename
from os import remove
import numpy as np
from pytest import fixture
from tiffreader import TiffReader, save_tiff

@fixture
def data_path() -> str:
    return resource_filename(Requirement.parse("tiffreader"), "tiffreader/test/data/example.tif")

@fixture
def save_path() -> str:
    return resource_filename(Requirement.parse("tiffreader"), "tiffreader/test/data/temp.tif")

def test_open(data_path: str):
    tif = TiffReader.open(data_path)
    frame = tif[50]
    assert(frame.dtype == np.uint16)
    assert(np.array_equal(frame.shape, [100, 512]))
    assert(tif.length == 100)
    assert(np.array_equal(tif.shape, [100, 512]))

def test_save(save_path: str):
    save_tiff(np.arange(10000).reshape(100, 100), save_path)
    tif = TiffReader.open(save_path)
    assert(tif.length == 1)
    assert(np.array_equal(tif[0], np.arange(10000).reshape(100, 100)))
    remove(save_path)
