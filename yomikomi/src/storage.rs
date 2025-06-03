use crate::{DType, Layout, Result};

fn copy_strided_src_<T: Copy>(src: &[T], dst: &mut [T], dst_offset: usize, src_l: &Layout) {
    match src_l.strided_blocks() {
        crate::StridedBlocks::SingleBlock { start_offset, len } => {
            let to_copy = (dst.len() - dst_offset).min(len);
            dst[dst_offset..dst_offset + to_copy]
                .copy_from_slice(&src[start_offset..start_offset + to_copy])
        }
        crate::StridedBlocks::MultipleBlocks { block_start_index, block_len: 1 } => {
            for (dst_index, src_index) in block_start_index.enumerate() {
                let dst_index = dst_index + dst_offset;
                if dst_index >= dst.len() {
                    break;
                }
                dst[dst_index] = src[src_index]
            }
        }
        crate::StridedBlocks::MultipleBlocks { block_start_index, block_len } => {
            let mut dst_index = dst_offset;
            for src_index in block_start_index {
                let next_dst_index = dst_index + block_len;
                if dst_index >= dst.len() {
                    break;
                }
                let to_copy = usize::min(block_len, dst.len() - dst_index);
                dst[dst_index..dst_index + to_copy]
                    .copy_from_slice(&src[src_index..src_index + to_copy]);
                dst_index = next_dst_index
            }
        }
    }
}

// We use an enum here rather than dynamic dispatch based on a Box dyn as this is a bit easier to
// handle.
pub enum Storage {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U32(Vec<u32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Scalar {
    U8(u8),
    I8(i8),
    U32(u32),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl Storage {
    pub fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::I8(_) => DType::I8,
            Self::U32(_) => DType::U32,
            Self::I64(_) => DType::I64,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    pub(crate) fn copy_strided_src(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        src_l: &Layout,
    ) -> Result<()> {
        match (self, dst) {
            (Self::U8(src), Self::U8(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::U32(src), Self::U32(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::I64(src), Self::I64(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::F32(src), Self::F32(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::F64(src), Self::F64(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (_, dst) => {
                crate::bail!("mismatch in copy_strided {:?} <> {:?}", self.dtype(), dst.dtype())
            }
        }
        Ok(())
    }

    pub fn zeros(elem_count: usize, dtype: DType) -> Self {
        match dtype {
            DType::U8 => Self::U8(vec![0; elem_count]),
            DType::I8 => Self::I8(vec![0; elem_count]),
            DType::U32 => Self::U32(vec![0; elem_count]),
            DType::I64 => Self::I64(vec![0; elem_count]),
            DType::F32 => Self::F32(vec![0.; elem_count]),
            DType::F64 => Self::F64(vec![0.; elem_count]),
        }
    }

    pub fn full(elem_count: usize, s: Scalar) -> Self {
        match s {
            Scalar::U8(v) => Self::U8(vec![v; elem_count]),
            Scalar::I8(v) => Self::I8(vec![v; elem_count]),
            Scalar::U32(v) => Self::U32(vec![v; elem_count]),
            Scalar::I64(v) => Self::I64(vec![v; elem_count]),
            Scalar::F32(v) => Self::F32(vec![v; elem_count]),
            Scalar::F64(v) => Self::F64(vec![v; elem_count]),
        }
    }
}
