"""
https://github.com/data-apis/array-api/blob/master/spec/API_specification/type_promotion.md
"""

from ._array_module import (arange, add, int8, int16, int32, int64, uint8,
                            uint16, uint32, uint64, float32, float64)


dtypes = {
    'i1': int8,
    'i2': int16,
    'i4': int32,
    'i8': int64,
    'u1': uint8,
    'u2': uint16,
    'u4': uint32,
    'u8': uint64,
    'f4': float32,
    'f8': float64,
}

signed_integer_promotion_table = {
    ('i1', 'i1'): 'i1',
    ('i1', 'i2'): 'i2',
    ('i1', 'i4'): 'i4',
    ('i1', 'i8'): 'i8',
    ('i2', 'i1'): 'i2',
    ('i2', 'i2'): 'i2',
    ('i2', 'i4'): 'i4',
    ('i2', 'i8'): 'i8',
    ('i4', 'i1'): 'i4',
    ('i4', 'i2'): 'i4',
    ('i4', 'i4'): 'i4',
    ('i4', 'i8'): 'i8',
    ('i8', 'i1'): 'i8',
    ('i8', 'i2'): 'i8',
    ('i8', 'i4'): 'i8',
    ('i8', 'i8'): 'i8',
}

unsigned_integer_promotion_table = {
    ('u1', 'u1'): 'u1',
    ('u1', 'u2'): 'u2',
    ('u1', 'u4'): 'u4',
    ('u1', 'u8'): 'u8',
    ('u2', 'u1'): 'u2',
    ('u2', 'u2'): 'u2',
    ('u2', 'u4'): 'u4',
    ('u2', 'u8'): 'u8',
    ('u4', 'u1'): 'u4',
    ('u4', 'u2'): 'u4',
    ('u4', 'u4'): 'u4',
    ('u4', 'u8'): 'u8',
    ('u8', 'u1'): 'u8',
    ('u8', 'u2'): 'u8',
    ('u8', 'u4'): 'u8',
    ('u8', 'u8'): 'u8',
}

mixed_signed_unsigned_promotion_table = {
    ('i1', 'u1'): 'i2',
    ('i1', 'u2'): 'i4',
    ('i1', 'u4'): 'i8',
    ('i2', 'u1'): 'i2',
    ('i2', 'u2'): 'i4',
    ('i2', 'u4'): 'i8',
    ('i4', 'u1'): 'i4',
    ('i4', 'u2'): 'i4',
    ('i4', 'u4'): 'i8',
}

flipped_mixed_signed_unsigned_promotion_table = {(u, i): p for (i, u), p in mixed_signed_unsigned_promotion_table.items()}

float_promotion_table = {
    ('f4', 'f4'): 'f4',
    ('f4', 'f8'): 'f8',
    ('f8', 'f4'): 'f8',
    ('f8', 'f8'): 'f8',
}

promotion_table = {
    **signed_integer_promotion_table,
    **unsigned_integer_promotion_table,
    **mixed_signed_unsigned_promotion_table,
    **flipped_mixed_signed_unsigned_promotion_table,
    **float_promotion_table,
}

def test_add():
    for (type1, type2), res_type in promotion_table.items():
        dtype1 = dtypes[type1]
        dtype2 = dtypes[type2]
        a1 = arange(2, dtype=dtype1)
        a2 = arange(2, dtype=dtype2)
        res = add(a1, a2)
        res_dtype = dtypes[res_type]

        assert res.dtype == res_dtype, f"({dtype1}, {dtype2}) promoted to {res.dtype}, should have promoted to {res_dtype}"

def test_add_0d():
    for (type1, type2), res_type in promotion_table.items():
        dtype1 = dtypes[type1]
        dtype2 = dtypes[type2]
        a1 = arange(0, dtype=dtype1)
        a2 = arange(0, dtype=dtype2)
        res = add(a1, a2)
        res_dtype = dtypes[res_type]

        assert res.dtype == res_dtype, f"({dtype1}, {dtype2}) promoted to {res.dtype}, should have promoted to {res_dtype}"

if __name__ == '__main__':
    for (i, j), p in promotion_table.items():
        print(f"({i}, {j}) -> {p}")
