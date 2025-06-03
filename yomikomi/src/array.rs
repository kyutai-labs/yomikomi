use crate::{DType, Error, Layout, Result, Scalar, Shape, Storage};
use std::sync::Arc;

#[derive(Clone)]
pub struct Array {
    storage: Arc<Storage>,
    layout: Layout,
}

impl AsRef<Array> for Array {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl Array {
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    pub fn dims(&self) -> &[usize] {
        self.layout.shape().dims()
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    pub fn elem_count(&self) -> usize {
        self.layout.shape().elem_count()
    }

    pub fn stride(&self) -> &[usize] {
        self.layout.stride()
    }

    /// The number of dimensions for this tensor, 0 for a scalar tensor, 1 for a 1D tensor, etc.
    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    // TODO: handle the non-contiguous case.
    pub fn values<T: crate::WithDType>(&self) -> Result<&[T]> {
        T::cpu_storage_as_slice(self.storage.as_ref())
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn get(&self, i: usize) -> Result<Self> {
        let dims = self.dims();
        if dims.is_empty() {
            Ok(self.clone())
        } else {
            self.narrow(0, i, 1)?.reshape(&dims[1..])
        }
    }

    pub fn flatten_all(&self) -> Result<Self> {
        self.reshape(&[self.elem_count()])
    }

    pub fn resize_pad(&self, dim: usize, len: usize, pad_with: Option<f64>) -> Result<Self> {
        let dims = self.dims();
        let v = match dims[dim].cmp(&len) {
            std::cmp::Ordering::Equal => self.clone(),
            std::cmp::Ordering::Greater => {
                let layout = self.layout().narrow(dim, 0, len)?;
                Self { storage: self.storage.clone(), layout }
            }
            std::cmp::Ordering::Less => self.pad(dim, 0, len - dims[dim], pad_with)?,
        };
        Ok(v)
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let dims = self.dims();
        let err = |msg| {
            Err::<(), _>(
                Error::NarrowInvalidArgs { shape: self.shape().clone(), dim, start, len, msg }.bt(),
            )
        };
        if start > dims[dim] {
            err("start > dim_len")?
        }
        if start.saturating_add(len) > dims[dim] {
            err("start + len > dim_len")?
        }
        if start == 0 && dims[dim] == len {
            Ok(self.clone())
        } else {
            let layout = self.layout().narrow(dim, start, len)?;
            Ok(Self { storage: self.storage.clone(), layout })
        }
    }

    /// Returns an iterator over position of the elements in the storage when ranging over the
    /// index tuples in lexicographic order.
    pub fn strided_index(&self) -> crate::StridedIndex {
        self.layout.strided_index()
    }

    /// Similar to `strided_index` but returns the position of the start of each contiguous block
    /// as well as the length of the contiguous blocks. For a contiguous tensor, the index iterator
    /// will only return the start offset and the size would be the number of elements in the
    /// tensor.
    pub fn strided_blocks(&self) -> crate::StridedBlocks {
        self.layout.strided_blocks()
    }

    pub fn zeros<S: Into<Shape>>(s: S, dtype: DType) -> Self {
        let shape = s.into();
        let storage = Storage::zeros(shape.elem_count(), dtype);
        Self { storage: Arc::new(storage), layout: Layout::contiguous(shape) }
    }

    pub fn full<S: Into<Shape>>(s: S, v: Scalar) -> Self {
        let shape = s.into();
        let storage = Storage::full(shape.elem_count(), v);
        Self { storage: Arc::new(storage), layout: Layout::contiguous(shape) }
    }

    pub fn reshape<S: Into<Shape>>(&self, s: S) -> Result<Self> {
        let shape = s.into();
        if shape.elem_count() != self.elem_count() {
            crate::bail!("shape mismatch in reshape {:?} {:?}", self.shape(), shape)
        }
        if self.layout.is_contiguous() {
            let array = Self {
                storage: self.storage.clone(),
                layout: Layout::contiguous_with_offset(shape, self.layout.start_offset()),
            };
            Ok(array)
        } else {
            let mut storage = Storage::zeros(shape.elem_count(), self.dtype());
            self.storage.copy_strided_src(&mut storage, 0, self.layout())?;
            let array = Self { storage: Arc::new(storage), layout: Layout::contiguous(shape) };
            Ok(array)
        }
    }

    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        if dim1 == dim2 {
            return Ok(self.clone());
        }
        let array =
            Self { storage: self.storage.clone(), layout: self.layout.transpose(dim1, dim2)? };
        Ok(array)
    }

    pub fn squeeze(&self, dim: usize) -> Result<Self> {
        let dims = self.dims();
        if dim >= dims.len() {
            crate::bail!("squeeze on inexistent dim {dim}, shape {:?}", self.layout.shape())
        }
        if dims[dim] != 1 {
            crate::bail!("squeeze on dim with a size different from one {dim} {dims:?}")
        }
        let mut dims = dims.to_vec();
        dims.remove(dim);
        self.reshape(dims)
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let mut dims = self.dims().to_vec();
        if dim > dims.len() {
            crate::bail!("unsqueeze on inexistent dim {dim}, shape {:?}", self.layout.shape())
        }
        dims.insert(dim, 1);
        self.reshape(dims)
    }

    fn check_dim(&self, dim: usize, op: &'static str) -> Result<()> {
        if dim >= self.dims().len() {
            Err(Error::DimOutOfRange { shape: self.shape().clone(), dim: dim as i32, op }.bt())?
        } else {
            Ok(())
        }
    }

    pub fn stack<A: AsRef<Self>>(args: &[A], dim: usize) -> Result<Self> {
        if args.is_empty() {
            crate::bail!("stack requires at least one tensor")
        }
        let args = args.iter().map(|t| t.as_ref().unsqueeze(dim)).collect::<Result<Vec<_>>>()?;
        Self::cat(&args, dim)
    }

    pub fn cat<A: AsRef<Self>>(args: &[A], dim: usize) -> Result<Self> {
        if args.is_empty() {
            crate::bail!("cat requires at least one tensor")
        }
        let arg0 = args[0].as_ref();
        if args.len() == 1 {
            return Ok(arg0.clone());
        }
        for arg in args {
            arg.as_ref().check_dim(dim, "cat")?;
        }
        for (arg_idx, arg) in args.iter().enumerate() {
            let arg = arg.as_ref();
            if arg0.rank() != arg.rank() {
                Err(Error::UnexpectedNumberOfDims {
                    expected: arg0.rank(),
                    got: arg.rank(),
                    shape: arg.shape().clone(),
                }
                .bt())?
            }
            for (dim_idx, (v1, v2)) in
                arg0.shape().dims().iter().zip(arg.shape().dims().iter()).enumerate()
            {
                if dim_idx != dim && v1 != v2 {
                    crate::bail!(
                        "shape mismatch in cat on {dim_idx}, [0]: {:?}, [{arg_idx}]: {:?}",
                        arg0.shape(),
                        arg.shape()
                    )
                }
            }
        }
        if dim == 0 {
            Self::cat0(args)
        } else {
            // TODO: Avoid these transpositions and have an implementation that works
            // for dim != 0...
            let args: Vec<Array> =
                args.iter().map(|a| a.as_ref().transpose(0, dim)).collect::<Result<Vec<_>>>()?;
            let cat = Self::cat0(args.as_slice())?;
            cat.transpose(0, dim)
        }
    }

    fn cat0<A: AsRef<Array>>(args: &[A]) -> Result<Self> {
        if args.is_empty() {
            crate::bail!("cat requires at least one tensor")
        }
        let arg0 = args[0].as_ref();
        if args.len() == 1 {
            return Ok(arg0.clone());
        }
        let rank = arg0.rank();
        let dtype = arg0.dtype();
        let first_dims = arg0.shape().dims();
        let mut cat_dims = first_dims.to_vec();
        cat_dims[0] = 0;
        let mut offsets = vec![0usize];
        for (arg_idx, arg) in args.iter().enumerate() {
            let arg = arg.as_ref();
            if arg.dtype() != dtype {
                crate::bail!(
                    "dtype mismatch in cat, [0]: {dtype:?}, [{arg_idx}]: {:?}",
                    arg.dtype()
                )
            }
            if rank != arg.rank() {
                Err(Error::UnexpectedNumberOfDims {
                    expected: rank,
                    got: arg.rank(),
                    shape: arg.shape().clone(),
                }
                .bt())?
            }
            for (dim_idx, (v1, v2)) in
                arg0.shape().dims().iter().zip(arg.shape().dims().iter()).enumerate()
            {
                if dim_idx == 0 {
                    cat_dims[0] += v2;
                }
                if dim_idx != 0 && v1 != v2 {
                    crate::bail!(
                        "shape mismatch in cat on {dim_idx}, [0]: {:?}, [{arg_idx}]: {:?}",
                        arg0.shape(),
                        arg.shape()
                    )
                }
            }
            let next_offset = offsets.last().unwrap() + arg.elem_count();
            offsets.push(next_offset);
        }
        let shape = Shape::from(cat_dims);
        let mut storage = Storage::zeros(shape.elem_count(), dtype);
        for (arg, &offset) in args.iter().zip(offsets.iter()) {
            let arg = arg.as_ref();
            arg.storage().copy_strided_src(&mut storage, offset, arg.layout())?;
        }
        Ok(Self { storage: Arc::new(storage), layout: Layout::contiguous(shape) })
    }

    pub fn pad(
        &self,
        dim: usize,
        left: usize,
        right: usize,
        pad_with: Option<f64>,
    ) -> Result<Self> {
        let pad_with = match self.dtype() {
            DType::U8 => Scalar::U8(pad_with.map_or(0, |v| v as u8)),
            DType::I8 => Scalar::I8(pad_with.map_or(0, |v| v as i8)),
            DType::U32 => Scalar::U32(pad_with.map_or(0, |v| v as u32)),
            DType::I64 => Scalar::I64(pad_with.map_or(0, |v| v as i64)),
            DType::F32 => Scalar::F32(pad_with.map_or(0., |v| v as f32)),
            DType::F64 => Scalar::F64(pad_with.unwrap_or(0.)),
        };
        if left == 0 && right == 0 {
            Ok(self.clone())
        } else if left == 0 {
            let mut dims = self.dims().to_vec();
            dims[dim] = right;
            let right = Self::full(dims.as_slice(), pad_with);
            Self::cat(&[self, &right], dim)
        } else if right == 0 {
            let mut dims = self.dims().to_vec();
            dims[dim] = left;
            let left = Self::full(dims.as_slice(), pad_with);
            Self::cat(&[&left, self], dim)
        } else {
            let mut dims = self.dims().to_vec();
            dims[dim] = left;
            let left = Self::full(dims.as_slice(), pad_with);
            dims[dim] = right;
            let right = Self::full(dims.as_slice(), pad_with);
            Self::cat(&[&left, self, &right], dim)
        }
    }

    pub fn to_vec0<S: crate::WithDType>(&self) -> Result<S> {
        if self.rank() != 0 {
            Err(Error::UnexpectedNumberOfDims {
                expected: 0,
                got: self.rank(),
                shape: self.shape().clone(),
            }
            .bt())?
        }
        let data = S::cpu_storage_as_slice(&self.storage)?;
        Ok::<_, Error>(data[self.layout().start_offset()])
    }

    /// Returns the data contained in a 1D array as a vector of scalar values.
    pub fn to_vec1<S: crate::WithDType>(&self) -> Result<Vec<S>> {
        if self.rank() != 1 {
            Err(Error::UnexpectedNumberOfDims {
                expected: 1,
                got: self.rank(),
                shape: self.shape().clone(),
            }
            .bt())?
        }
        let data = S::cpu_storage_as_slice(&self.storage)?;
        let data = match self.layout.contiguous_offsets() {
            Some((o1, o2)) => data[o1..o2].to_vec(),
            None => self.strided_index().map(|i| data[i]).collect(),
        };
        Ok(data)
    }

    pub fn to_vec2<S: crate::WithDType>(&self) -> Result<Vec<Vec<S>>> {
        let (dim1, dim2) = self.shape().dims2()?;
        let data = S::cpu_storage_as_slice(&self.storage)?;
        let mut rows = vec![];
        match self.layout.contiguous_offsets() {
            Some((o1, o2)) => {
                let data = &data[o1..o2];
                for idx_row in 0..dim1 {
                    rows.push(data[idx_row * dim2..(idx_row + 1) * dim2].to_vec())
                }
            }
            None => {
                let mut src_index = self.strided_index();
                for _idx_row in 0..dim1 {
                    let row = (0..dim2).map(|_| data[src_index.next().unwrap()]).collect();
                    rows.push(row)
                }
                assert!(src_index.next().is_none());
            }
        }
        Ok(rows)
    }

    pub fn to_vec3<S: crate::WithDType>(&self) -> Result<Vec<Vec<Vec<S>>>> {
        let (dim1, dim2, dim3) = self.shape().dims3()?;
        let data = S::cpu_storage_as_slice(&self.storage)?;
        let mut top_rows = vec![];
        match self.layout.contiguous_offsets() {
            Some((o1, o2)) => {
                let data = &data[o1..o2];
                let dim23 = dim2 * dim3;
                for idx1 in 0..dim1 {
                    let data = &data[idx1 * dim23..(idx1 + 1) * dim23];
                    let mut rows = vec![];
                    for idx2 in 0..dim2 {
                        rows.push(data[idx2 * dim3..(idx2 + 1) * dim3].to_vec())
                    }
                    top_rows.push(rows);
                }
            }
            None => {
                let mut src_index = self.strided_index();
                for _idx in 0..dim1 {
                    let mut rows = vec![];
                    for _jdx in 0..dim2 {
                        let row = (0..dim3).map(|_| data[src_index.next().unwrap()]).collect();
                        rows.push(row)
                    }
                    top_rows.push(rows);
                }
                assert!(src_index.next().is_none());
            }
        }
        Ok(top_rows)
    }
}

impl<T: crate::WithDType> From<Vec<T>> for Array {
    fn from(value: Vec<T>) -> Self {
        let layout = Layout::contiguous(vec![value.len()]);
        let storage = Arc::new(T::to_cpu_storage_owned(value));
        Self { storage, layout }
    }
}

impl<T: crate::WithDType> From<T> for Array {
    fn from(value: T) -> Self {
        let layout = Layout::contiguous(vec![]);
        let storage = Arc::new(T::to_cpu_storage_owned(vec![value]));
        Self { storage, layout }
    }
}
