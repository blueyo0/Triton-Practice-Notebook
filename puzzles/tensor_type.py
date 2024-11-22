# Jaxtyping-like typing system for PyTorch
# https://github.com/patrick-kidger/jaxtyping/


from typing import Tuple


class TensorType:
    
    def __init__(self, dtype, *dims):
        self.dtype: str = dtype
        self.dims: Tuple = dims
    
    def __repr__(self):
        return f"{self.dtype}[{', '.join(str(d) for d in self.dims)}]"


class _TypeAnnotation:

    def __init__(self, dtype):
        self.dtype = dtype

    def __getitem__(self, dims):
        if not isinstance(dims, tuple):
            dims = (dims,)
        return TensorType(self.dtype, *dims)


Float32 = _TypeAnnotation("float32")
Int32 = _TypeAnnotation("int32")


if __name__ == "__main__":
    x = Float32[10, 20]
    print(x)
    print(x.dtype)
    print(x.dims)

    y = Int32[10, 20]
    print(y)
    print(y.dtype)
    print(y.dims)

    z = Int32[10]
    print(z)
    print(z.dtype)
    print(z.dims)

