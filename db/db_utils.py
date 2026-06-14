import numpy as np

def array_to_blob(arr: np.ndarray) -> bytes:
    """Converts a numpy array to bytes for SQLite BLOB storage."""
    return arr.tobytes()


# def blob_to_array(blob: bytes, dtype=np.float32, shape=None) -> np.ndarray:
#     """Converts SQLite BLOB bytes back to a numpy array with an optional shape."""
#     arr = np.frombuffer(blob, dtype=dtype)
#     if shape is not None:
#         arr = arr.reshape(shape)
#     return arr