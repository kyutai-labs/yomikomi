//! Types for elements that can be stored and manipulated using arrays.
use crate::{Error, Result, Storage};

/// The different types of elements allowed in arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    // Unsigned 8 bits integer.
    U8,
    // Signed 8 bits integer.
    I8,
    // Unsigned 32 bits integer.
    U32,
    // Signed 64 bits integer.
    I64,
    // Floating-point using single precision (32 bits).
    F32,
    // Floating-point using double precision (64 bits).
    F64,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DTypeParseError;

impl std::str::FromStr for DType {
    type Err = DTypeParseError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "u8" => Ok(Self::U8),
            "i8" => Ok(Self::I8),
            "u32" => Ok(Self::U32),
            "i64" => Ok(Self::I64),
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            _ => Err(DTypeParseError),
        }
    }
}

impl DType {
    /// String representation for dtypes.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    /// The size used by each element in bytes, i.e. 1 for `U8`, 4 for `F32`.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U32 => 4,
            Self::I64 => 8,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    pub fn is_int(&self) -> bool {
        match self {
            Self::U8 | Self::I8 | Self::U32 | Self::I64 => true,
            Self::F32 | Self::F64 => false,
        }
    }

    pub fn is_float(&self) -> bool {
        match self {
            Self::U8 | Self::I8 | Self::U32 | Self::I64 => false,
            Self::F32 | Self::F64 => true,
        }
    }
}

pub trait WithDType:
    Sized + Copy + std::cmp::PartialOrd + std::fmt::Display + 'static + Send + Sync
{
    const DTYPE: DType;

    fn to_cpu_storage_owned(data: Vec<Self>) -> Storage;

    fn to_cpu_storage(data: &[Self]) -> Storage {
        Self::to_cpu_storage_owned(data.to_vec())
    }

    fn cpu_storage_as_slice(s: &Storage) -> Result<&[Self]>;
    fn cpu_storage_data(s: Storage) -> Result<Vec<Self>>;
}

macro_rules! with_dtype {
    ($ty:ty, $dtype:ident) => {
        impl WithDType for $ty {
            const DTYPE: DType = DType::$dtype;

            fn to_cpu_storage_owned(data: Vec<Self>) -> Storage {
                Storage::$dtype(data)
            }

            fn cpu_storage_data(s: Storage) -> Result<Vec<Self>> {
                match s {
                    Storage::$dtype(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }

            fn cpu_storage_as_slice(s: &Storage) -> Result<&[Self]> {
                match s {
                    Storage::$dtype(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }
        }
    };
}

with_dtype!(u8, U8);
with_dtype!(i8, I8);
with_dtype!(u32, U32);
with_dtype!(i64, I64);
with_dtype!(f32, F32);
with_dtype!(f64, F64);
