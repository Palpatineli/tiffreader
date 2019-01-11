# TiffReader, convenience wrapper for libtiff

## Why?

Compared to PIL and some other tiff related projects, this package is more for scientific imaging.

Features exposed from libtiff:

1. responds to
    1. pixel bit depth
    2. number of channels
    3. compression scheme
2. sequential and random access to frames in multi-frame tiff
3. query length of multi-frame tiff

## Open

```python
from tiffreader import TiffReader
tif = TiffReader.open("file_path.tif")
```

## Random Access

```python
tif.seek(10)
frame = tif.read_current()  # gives a 2D numpy array
```

is equivalent to

```python
frame = tif[10]
```

## Sequential Access

example for an average image of the 10th to 20th frames:

```python
tif.seek(10)
result = np.zeros(tif.shape, dtype=np.uint64)
for frame in zip(tif, range(10)):
    result += frame
result /= 10
```

## Additionally

```python
from tiffreader import save_tiff
array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8)
save_tiff(array, "tif_path.tif")
```

```python
tif.length  # length of multi-frame tiff stack
tif.shape   # shape of one frame
tiffinfo("tif_path.tif", ["width", "height"])  # wraps tiffinfo to query for additional tags
```
