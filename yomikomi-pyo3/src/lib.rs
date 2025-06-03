#![allow(clippy::useless_conversion)]
use numpy::prelude::*;
use pyo3::{prelude::*, BoundObject};
use std::collections::HashMap;
use std::sync::Arc;

use ::yomikomi as yk;
use yk::{bail, Array, DType, Error, Result, Stream, WithDType};

fn w_py(err: PyErr) -> Error {
    Error::msg(err)
}

fn w<E: std::error::Error>(err: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(err.to_string())
}

#[macro_export]
macro_rules! py_bail {
    ($msg:literal $(,)?) => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!($msg)))
    };
    ($err:expr $(,)?) => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!($err)))
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!($fmt, $($arg)*)))
    };
}

trait Iterable {
    fn iter(&self) -> PyResult<StreamIter>;
}

struct Audio {
    file: std::path::PathBuf,
}

impl Iterable for Audio {
    fn iter(&self) -> PyResult<StreamIter> {
        let stream = yk::audio::FileReader::new(self.file.clone()).map_err(w)?;
        Ok(StreamIter { stream: Box::new(stream) })
    }
}

struct JsonL {
    file: std::path::PathBuf,
    offset: u64,
    fields: Option<Vec<String>>,
    filters: Vec<yk::jsonl::Filter>,
    include_if_missing: bool,
}

impl Iterable for JsonL {
    fn iter(&self) -> PyResult<StreamIter> {
        let stream = yk::jsonl::FileReader::new_multi(
            self.file.clone(),
            self.offset,
            self.fields.clone(),
            self.filters.clone(),
            self.include_if_missing,
        )
        .map_err(w)?;
        Ok(StreamIter { stream: Box::new(stream) })
    }
}

struct Warc {
    file: std::path::PathBuf,
}

impl Iterable for Warc {
    fn iter(&self) -> PyResult<StreamIter> {
        let stream = yk::warc::FileReader::new(self.file.clone()).map_err(w)?;
        Ok(StreamIter { stream: Box::new(stream) })
    }
}

struct Filter {
    inner: Arc<dyn Iterable + 'static + Send + Sync>,
    filter_fn: Arc<PyObject>,
    field: Option<String>,
}

impl Iterable for Filter {
    fn iter(&self) -> PyResult<StreamIter> {
        let inner = self.inner.iter()?.stream;
        let field = self.field.clone();
        let filter_fn = self.filter_fn.clone();
        let stream = inner.filter(move |sample| {
            let value = Python::with_gil(|py| {
                let value = match field.as_ref() {
                    None => sample
                        .iter()
                        .map(|(k, v)| {
                            let v = array_to_py(v, py)?;
                            Ok((k.to_string(), v))
                        })
                        .collect::<Result<HashMap<_, _>>>()?
                        .into_pyobject(py)
                        .map_err(w_py)?
                        .into_any(),
                    Some(field) => match sample.get(field.as_str()) {
                        Some(value) => array_to_py(value, py)?,
                        None => bail!("Filter cannot find '{}' in sample", field),
                    },
                };
                let value = filter_fn.call1(py, (value,)).map_err(w_py)?;
                let value = value.is_truthy(py).map_err(w_py)?;
                Ok(value)
            })?;
            Ok(value)
        });
        Ok(StreamIter { stream: Box::new(stream) })
    }
}

struct AndThen {
    inner: Arc<dyn Iterable + 'static + Send + Sync>,
    and_then_fn: Arc<PyObject>,
}

impl Iterable for AndThen {
    fn iter(&self) -> PyResult<StreamIter> {
        let inner = self.inner.iter()?.stream;
        let and_then_fn = self.and_then_fn.clone();
        let stream = inner.and_then(move |sample| {
            let value = Python::with_gil(|py| {
                let value = sample
                    .iter()
                    .map(|(k, v)| {
                        let v = array_to_py(v, py)?;
                        Ok((k.to_string(), v))
                    })
                    .collect::<Result<HashMap<_, _>>>()?
                    .into_pyobject(py)
                    .map_err(w_py)?;
                let value = and_then_fn.call1(py, (value,)).map_err(w_py)?;
                if value.is_none(py) {
                    Ok(None)
                } else {
                    let value = match value.downcast_bound::<pyo3::types::PyDict>(py) {
                        Ok(value) => value,
                        Err(_) => {
                            let value = value.downcast_bound(py).map_err(Error::msg)?;
                            bail!(
                                "map-fn returned an object that is not a dict, {:?}",
                                value.get_type()
                            )
                        }
                    };
                    let value = value
                        .iter()
                        .map(|(key, value)| {
                            let key = match key.downcast::<pyo3::types::PyString>() {
                                Ok(str) => str.to_string_lossy().to_string(),
                                Err(_) => bail!("key is not a string, got {:?}", key.get_type()),
                            };
                            let value = py_to_array(py, value.as_unbound())?;
                            Ok((key, value))
                        })
                        .collect::<Result<HashMap<String, Array>>>()?;
                    Ok(Some(value))
                }
            })?;
            Ok(value)
        });
        Ok(StreamIter { stream: Box::new(stream) })
    }
}

struct KeyTransform {
    inner: Arc<dyn Iterable + 'static + Send + Sync>,
    map_fn: Arc<PyObject>,
    field: String,
}

impl Iterable for KeyTransform {
    fn iter(&self) -> PyResult<StreamIter> {
        let inner = self.inner.iter()?.stream;
        let field = self.field.to_string();
        let map_fn = self.map_fn.clone();
        let stream = inner.and_then(move |mut sample| {
            let value = match sample.remove(field.as_str()) {
                Some(value) => value,
                None => bail!("KeyTransform cannot find '{}' in sample", field),
            };
            let value = Python::with_gil(|py| {
                let value = array_to_py(&value, py)?;
                let value = map_fn.call1(py, (value,)).map_err(w_py)?;
                py_to_array(py, &value)
            })?;
            sample.insert(field.to_string(), value);
            Ok(Some(sample))
        });
        Ok(StreamIter { stream: Box::new(stream) })
    }
}

struct Enumerate {
    inner: Arc<dyn Iterable + 'static + Send + Sync>,
    field: String,
}

impl Iterable for Enumerate {
    fn iter(&self) -> PyResult<StreamIter> {
        let inner = self.inner.iter()?.stream;
        let stream = yk::enumerate::Enumerate::new(inner, self.field.clone());
        Ok(StreamIter { stream: Box::new(stream) })
    }
}

struct Tokenize {
    path: std::path::PathBuf,
    inner: Arc<dyn Iterable + 'static + Send + Sync>,
    in_field: String,
    out_field: String,
    report_bpb: bool,
    include_bos: bool,
    include_eos: bool,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
}

impl Iterable for Tokenize {
    fn iter(&self) -> PyResult<StreamIter> {
        let inner = self.inner.iter()?.stream;
        let stream = yk::tokenize::Tokenize::new(
            self.path.clone(),
            inner,
            self.in_field.clone(),
            self.out_field.clone(),
            self.report_bpb,
            self.include_bos,
            self.include_eos,
            self.bos_id,
            self.eos_id,
        )
        .map_err(w)?;
        Ok(StreamIter { stream: Box::new(stream) })
    }
}

struct SlidingWindow {
    inner: Arc<dyn Iterable + 'static + Send + Sync>,
    field: String,
    window_size: usize,
    stride: usize,
    overlap_over_samples: bool,
}

impl Iterable for SlidingWindow {
    fn iter(&self) -> PyResult<StreamIter> {
        let inner = self.inner.iter()?.stream;
        let stream = yk::sliding_window::SlidingWindow::new(
            inner,
            self.window_size,
            self.stride,
            self.field.clone(),
            self.overlap_over_samples,
        )
        .map_err(w)?;
        Ok(StreamIter { stream: Box::new(stream) })
    }
}

struct FirstSlice {
    inner: Arc<dyn Iterable + 'static + Send + Sync>,
    field: String,
    window_size: usize,
    pad_with: Option<f64>,
}

impl Iterable for FirstSlice {
    fn iter(&self) -> PyResult<StreamIter> {
        let inner = self.inner.iter()?.stream;
        let stream = yk::sliding_window::FirstSlice::new(
            inner,
            self.window_size,
            self.field.clone(),
            self.pad_with,
        )
        .map_err(w)?;
        Ok(StreamIter { stream: Box::new(stream) })
    }
}

struct Batch {
    inner: Arc<dyn Iterable + 'static + Send + Sync>,
    batch_size: usize,
    return_incomplete_last_batch: bool,
}

impl Iterable for Batch {
    fn iter(&self) -> PyResult<StreamIter> {
        let inner = self.inner.iter()?.stream;
        let stream =
            yk::batch::Batch::new(inner, self.batch_size, self.return_incomplete_last_batch)
                .map_err(w)?;
        Ok(StreamIter { stream: Box::new(stream) })
    }
}

struct PreFetch {
    inner: Arc<dyn Iterable + 'static + Send + Sync>,
    num_threads: usize,
    buffer_size: usize,
}

impl Iterable for PreFetch {
    fn iter(&self) -> PyResult<StreamIter> {
        let inner = self.inner.iter()?.stream;
        let stream =
            yk::prefetch::PreFetch::new(inner, self.num_threads, self.buffer_size).map_err(w)?;
        Ok(StreamIter { stream: Box::new(stream) })
    }
}

struct FilterKey {
    inner: Arc<dyn Iterable + 'static + Send + Sync>,
    key: String,
    remove: bool,
}

impl Iterable for FilterKey {
    fn iter(&self) -> PyResult<StreamIter> {
        let inner = self.inner.iter()?.stream;
        let stream = yk::filter_key::FilterKey::new(inner, self.key.clone(), self.remove);
        Ok(StreamIter { stream: Box::new(stream) })
    }
}

#[pyclass]
struct StreamIter {
    stream: Box<dyn Stream + 'static + Send + Sync>,
}

fn array_to_py<'a>(v: &Array, py: Python<'a>) -> Result<Bound<'a, PyAny>> {
    fn to_vec<'a, T: WithDType + numpy::Element + IntoPyObject<'a>>(
        v: &Array,
        py: Python<'a>,
    ) -> Result<Bound<'a, PyAny>> {
        let v = match v.rank() {
            0 => {
                // Return scalar values using the built-in types rather than through a num
                let v = v.to_vec0::<T>()?;
                let obj = v.into_pyobject(py).map_err(|e| Error::wrap(e.into()))?;
                pyo3::BoundObject::into_any(obj).into_bound()
            }
            1 => {
                let v = v.to_vec1::<T>()?;
                numpy::PyArray1::from_vec(py, v).into_any()
            }
            2 => {
                let v = v.to_vec2::<T>()?;
                numpy::PyArray2::from_vec2(py, &v).map_err(Error::wrap)?.into_any()
            }
            3 => {
                let v = v.to_vec3::<T>()?;
                numpy::PyArray3::from_vec3(py, &v).map_err(Error::wrap)?.into_any()
            }
            r => bail!("unsupported rank for numpy conversion {r}"),
        };
        Ok(v)
    }
    match v.dtype() {
        DType::U8 => to_vec::<u8>(v, py),
        DType::I8 => to_vec::<i8>(v, py),
        DType::U32 => to_vec::<u32>(v, py),
        DType::I64 => to_vec::<i64>(v, py),
        DType::F32 => to_vec::<f32>(v, py),
        DType::F64 => to_vec::<f64>(v, py),
    }
}

#[pymethods]
impl StreamIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(
        slf: PyRefMut<'_, Self>,
        py: Python,
    ) -> PyResult<Option<HashMap<String, PyObject>>> {
        let stream = &slf.stream;
        let sample = py.allow_threads(|| stream.next().map_err(w))?;
        match sample {
            None => Ok(None),
            Some(v) => {
                let v = v
                    .into_iter()
                    .map(|(k, v)| {
                        let v = array_to_py(&v, py)?.unbind();
                        Ok((k, v))
                    })
                    .collect::<Result<HashMap<_, _>>>()
                    .map_err(w)?;
                Ok(Some(v))
            }
        }
    }
}

#[pyclass]
struct YkIterable {
    inner: Arc<dyn Iterable + 'static + Send + Sync>,
}

#[pymethods]
impl YkIterable {
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<StreamIter> {
        slf.inner.iter()
    }

    #[pyo3(signature = (f, *, field))]
    fn key_transform(&self, f: PyObject, field: String) -> PyResult<Self> {
        let f = Arc::new(f);
        let inner = KeyTransform { inner: self.inner.clone(), map_fn: f, field };
        Ok(Self { inner: Arc::new(inner) })
    }

    /// Filters a stream, the elements are kept if the provided function `f` returns `True` on
    /// them, otherwise they are discarded. If `field` is specified, the function `f` is only
    /// passed the value associated to this field rather than a whole dictionary.
    #[pyo3(signature = (f, *, field=None))]
    fn filter(&self, f: PyObject, field: Option<String>) -> PyResult<Self> {
        let f = Arc::new(f);
        let inner = Filter { inner: self.inner.clone(), filter_fn: f, field };
        Ok(Self { inner: Arc::new(inner) })
    }

    #[pyo3(signature = (f))]
    fn map(&self, f: PyObject) -> PyResult<Self> {
        let f = Arc::new(f);
        let inner = AndThen { inner: self.inner.clone(), and_then_fn: f };
        Ok(Self { inner: Arc::new(inner) })
    }

    /// Loads a sentencepiece tokenizer, and use it to tokenize the field passed as an argument of
    /// this function.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, *, in_field="text".to_string(), out_field=None, report_bpb=true, include_bos=true, include_eos=false, bos_id=None, eos_id=None))]
    fn tokenize(
        &self,
        path: std::path::PathBuf,
        in_field: String,
        out_field: Option<String>,
        report_bpb: bool,
        include_bos: bool,
        include_eos: bool,
        bos_id: Option<u32>,
        eos_id: Option<u32>,
    ) -> PyResult<Self> {
        let out_field = out_field.unwrap_or_else(|| in_field.clone());
        let inner = Tokenize {
            path,
            inner: self.inner.clone(),
            in_field,
            out_field,
            report_bpb,
            include_bos,
            include_eos,
            bos_id,
            eos_id,
        };
        Ok(Self { inner: Arc::new(inner) })
    }

    #[pyo3(signature = (field))]
    fn enumerate(&self, field: String) -> PyResult<Self> {
        let inner = Enumerate { field, inner: self.inner.clone() };
        Ok(Self { inner: Arc::new(inner) })
    }

    #[pyo3(signature = (window_size, *, stride=None, field="text".to_string(), overlap_over_samples=false))]
    fn sliding_window(
        &self,
        window_size: usize,
        stride: Option<usize>,
        field: String,
        overlap_over_samples: bool,
    ) -> PyResult<Self> {
        let inner = SlidingWindow {
            inner: self.inner.clone(),
            field,
            stride: stride.unwrap_or(window_size),
            window_size,
            overlap_over_samples,
        };
        Ok(Self { inner: Arc::new(inner) })
    }

    #[pyo3(signature = (window_size, *, field="text".to_string(), pad_with=None))]
    fn first_slice(
        &self,
        window_size: usize,
        field: String,
        pad_with: Option<f64>,
    ) -> PyResult<Self> {
        let inner = FirstSlice { inner: self.inner.clone(), field, window_size, pad_with };
        Ok(Self { inner: Arc::new(inner) })
    }

    #[pyo3(signature = (batch_size, *, return_incomplete_last_batch=false))]
    fn batch(&self, batch_size: usize, return_incomplete_last_batch: bool) -> PyResult<Self> {
        let inner = Batch { inner: self.inner.clone(), batch_size, return_incomplete_last_batch };
        Ok(Self { inner: Arc::new(inner) })
    }

    #[pyo3(signature = (*, num_threads, buffer_size=None))]
    fn prefetch(&self, num_threads: usize, buffer_size: Option<usize>) -> PyResult<Self> {
        let buffer_size = buffer_size.unwrap_or(num_threads);
        let inner = PreFetch { inner: self.inner.clone(), num_threads, buffer_size };
        Ok(Self { inner: Arc::new(inner) })
    }

    #[pyo3(signature = (key, *, remove=false))]
    fn filter_key(&self, key: String, remove: bool) -> PyResult<Self> {
        let inner = FilterKey { inner: self.inner.clone(), key, remove };
        Ok(Self { inner: Arc::new(inner) })
    }
}

#[derive(Clone)]
enum JsonFilter_ {
    LowerEq(String, f64, bool),
    Lower(String, f64, bool),
    GreaterEq(String, f64, bool),
    Greater(String, f64, bool),
    Eq(String, f64, bool),
    Neq(String, f64, bool),
}

#[pyclass]
#[derive(Clone)]
struct JsonFilter(JsonFilter_);

#[pymethods]
impl JsonFilter {
    #[pyo3(signature = (field, value, *, include_if_missing=false))]
    #[staticmethod]
    fn lower_eq(field: String, value: f64, include_if_missing: bool) -> Self {
        Self(JsonFilter_::LowerEq(field, value, include_if_missing))
    }

    #[staticmethod]
    #[pyo3(signature = (field, value, *, include_if_missing=false))]
    fn lower(field: String, value: f64, include_if_missing: bool) -> Self {
        Self(JsonFilter_::Lower(field, value, include_if_missing))
    }

    #[pyo3(signature = (field, value, *, include_if_missing=false))]
    #[staticmethod]
    fn greater_eq(field: String, value: f64, include_if_missing: bool) -> Self {
        Self(JsonFilter_::GreaterEq(field, value, include_if_missing))
    }

    #[pyo3(signature = (field, value, *, include_if_missing=false))]
    #[staticmethod]
    fn greater(field: String, value: f64, include_if_missing: bool) -> Self {
        Self(JsonFilter_::Greater(field, value, include_if_missing))
    }

    #[pyo3(signature = (field, value, *, include_if_missing=false))]
    #[staticmethod]
    fn eq(field: String, value: f64, include_if_missing: bool) -> Self {
        Self(JsonFilter_::Eq(field, value, include_if_missing))
    }

    #[pyo3(signature = (field, value, *, include_if_missing=false))]
    #[staticmethod]
    fn neq(field: String, value: f64, include_if_missing: bool) -> Self {
        Self(JsonFilter_::Neq(field, value, include_if_missing))
    }
}

impl From<JsonFilter> for yk::jsonl::Filter {
    fn from(val: JsonFilter) -> Self {
        match val.0 {
            JsonFilter_::LowerEq(field, value, include_if_missing) => yk::jsonl::Filter {
                field,
                cmp: yk::jsonl::FilterCmp::LowerEq,
                value,
                include_if_missing,
            },
            JsonFilter_::GreaterEq(field, value, include_if_missing) => yk::jsonl::Filter {
                field,
                cmp: yk::jsonl::FilterCmp::GreaterEq,
                value,
                include_if_missing,
            },
            JsonFilter_::Lower(field, value, include_if_missing) => yk::jsonl::Filter {
                field,
                cmp: yk::jsonl::FilterCmp::Lower,
                value,
                include_if_missing,
            },
            JsonFilter_::Greater(field, value, include_if_missing) => yk::jsonl::Filter {
                field,
                cmp: yk::jsonl::FilterCmp::Greater,
                value,
                include_if_missing,
            },
            JsonFilter_::Eq(field, value, include_if_missing) => yk::jsonl::Filter {
                field,
                cmp: yk::jsonl::FilterCmp::Eq,
                value,
                include_if_missing,
            },
            JsonFilter_::Neq(field, value, include_if_missing) => yk::jsonl::Filter {
                field,
                cmp: yk::jsonl::FilterCmp::Neq,
                value,
                include_if_missing,
            },
        }
    }
}

/// Returns a stream that iterates over the text contained in a specific field of a jsonl file.
#[pyfunction]
#[pyo3(signature = (file, *, offset=0, field=None, filters=vec![], include_if_missing=false))]
fn jsonl(
    file: std::path::PathBuf,
    offset: u64,
    field: Option<PyObject>,
    filters: Vec<JsonFilter>,
    include_if_missing: bool,
    py: Python,
) -> PyResult<YkIterable> {
    let filters = filters.into_iter().map(|v| v.into()).collect();
    let fields = match field {
        None => Some(vec!["text".to_string()]),
        Some(fields) => {
            if let Ok(fields) = fields.extract::<Vec<String>>(py) {
                if fields.is_empty() {
                    None
                } else {
                    Some(fields)
                }
            } else if let Ok(field) = fields.extract::<String>(py) {
                Some(vec![field])
            } else {
                py_bail!("unexpected type for field {fields:?}")
            }
        }
    };
    let jsonl = JsonL { file, offset, fields, filters, include_if_missing };
    Ok(YkIterable { inner: Arc::new(jsonl) })
}

/// Returns a stream that iterates over the pcm data in an audio file.
#[pyfunction]
#[pyo3(signature = (file))]
fn audio(file: std::path::PathBuf) -> YkIterable {
    let audio = Audio { file };
    YkIterable { inner: Arc::new(audio) }
}

/// Returns a stream that iterates over the documents contained in a warc file.
#[pyfunction]
#[pyo3(signature = (file))]
fn warc(file: std::path::PathBuf) -> YkIterable {
    let warc = Warc { file };
    YkIterable { inner: Arc::new(warc) }
}

struct YkPyIterator {
    iterator: PyObject,
    field: Option<String>,
}

fn py_to_array(py: Python<'_>, value: &PyObject) -> Result<Array> {
    // Be cautious in these conversions. Trying to `downcast_exact` on a numpy array of float32
    // would work fine with a dtype of u8 but hold incorrect results. So instead we first extract
    // the dtype and based on do the appropriate downcasting.
    if let Ok(value) = value.downcast_bound::<numpy::PyUntypedArray>(py) {
        let dtype = value.dtype();
        let shape = value.shape();
        if dtype.is_equiv_to(&numpy::dtype::<u8>(py)) {
            if let Ok(value) = value.downcast_exact::<numpy::PyArrayDyn<u8>>() {
                let value = value.to_vec().map_err(Error::msg)?;
                return Array::from(value).reshape(shape);
            }
        }
        if dtype.is_equiv_to(&numpy::dtype::<i8>(py)) {
            if let Ok(value) = value.downcast_exact::<numpy::PyArrayDyn<i8>>() {
                let value = value.to_vec().map_err(Error::msg)?;
                return Array::from(value).reshape(shape);
            }
        }
        if dtype.is_equiv_to(&numpy::dtype::<u32>(py)) {
            if let Ok(value) = value.downcast_exact::<numpy::PyArrayDyn<u32>>() {
                let value = value.to_vec().map_err(Error::msg)?;
                return Array::from(value).reshape(shape);
            }
        }
        if dtype.is_equiv_to(&numpy::dtype::<i64>(py)) {
            if let Ok(value) = value.downcast_exact::<numpy::PyArrayDyn<i64>>() {
                let value = value.to_vec().map_err(Error::msg)?;
                return Array::from(value).reshape(shape);
            }
        }
        if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
            if let Ok(value) = value.downcast_exact::<numpy::PyArrayDyn<f32>>() {
                let value = value.to_vec().map_err(Error::msg)?;
                return Array::from(value).reshape(shape);
            }
        }
        if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
            if let Ok(value) = value.downcast_exact::<numpy::PyArrayDyn<f64>>() {
                let value = value.to_vec().map_err(Error::msg)?;
                return Array::from(value).reshape(shape);
            }
        }
        bail!("unsupported dtype for np.array {}", dtype)
    }
    if let Ok(value) = value.extract::<String>(py) {
        return Ok(Array::from(value.into_bytes()));
    }
    if let Ok(value) = value.extract::<i64>(py) {
        return Ok(Array::from(value));
    }
    if let Ok(value) = value.extract::<f64>(py) {
        return Ok(Array::from(value));
    }
    if let Ok(value) = value.extract::<Vec<i64>>(py) {
        return Ok(Array::from(value));
    }
    if let Ok(value) = value.extract::<Vec<f64>>(py) {
        return Ok(Array::from(value));
    }
    let value = value.downcast_bound(py).map_err(Error::msg)?;
    bail!("unsupported types for conversion to array {:?}", value.get_type())
}

impl Stream for YkPyIterator {
    fn next(&self) -> Result<Option<yk::Sample>> {
        Python::with_gil(|py| {
            let next = match self.iterator.call_method0(py, "__next__") {
                Ok(next) => next,
                Err(err) if err.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) => {
                    return Ok(None)
                }
                Err(err) => return Err(w_py(err)),
            };
            match &self.field {
                None => {
                    let next = match next.downcast_bound::<pyo3::types::PyDict>(py) {
                        Ok(next) => next,
                        Err(_) => {
                            let ty = next.into_bound(py).get_type();
                            bail!("iterator returned an object that is not a dict, {ty:?}",)
                        }
                    };
                    let next = next
                        .iter()
                        .map(|(key, value)| {
                            let value = value.as_unbound();
                            let key = match key.downcast::<pyo3::types::PyString>() {
                                Ok(str) => str.to_string_lossy().to_string(),
                                Err(_) => bail!("key is not a string, got {:?}", key.get_type()),
                            };
                            let value = py_to_array(py, value)?;
                            Ok((key, value))
                        })
                        .collect::<Result<HashMap<String, Array>>>()?;
                    Ok(Some(next))
                }
                Some(field) => {
                    let mut sample = HashMap::new();
                    let next = py_to_array(py, &next)?;
                    sample.insert(field.clone(), next);
                    Ok(Some(sample))
                }
            }
        })
    }
}

struct YkPyIterable {
    iterable: PyObject,
    field: Option<String>,
}

impl Iterable for YkPyIterable {
    fn iter(&self) -> PyResult<StreamIter> {
        let iterator = Python::with_gil(|py| {
            let iterable = self.iterable.downcast_bound(py)?;
            let v =
                pyo3::types::PyAnyMethods::try_iter(iterable).map(|v| v.into_pyobject(py))??;
            Ok::<_, PyErr>(v.into_any().unbind())
        })?;
        let stream = YkPyIterator { iterator, field: self.field.clone() };
        Ok(StreamIter { stream: Box::new(stream) })
    }
}

/// Returns a stream based on a python iterator. The iterator can either return a whole dictionary
/// or if `field` is specified single values which will be embedded in a dictionary with a single
/// entry named as per the field argument.
#[pyfunction]
#[pyo3(signature = (iterable, *, field=None))]
fn stream(iterable: PyObject, field: Option<String>) -> PyResult<YkIterable> {
    let inner = YkPyIterable { iterable, field };
    Ok(YkIterable { inner: Arc::new(inner) })
}

#[pymodule]
fn yomikomi(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<StreamIter>()?;
    m.add_class::<JsonFilter>()?;
    m.add_class::<YkIterable>()?;
    m.add_function(wrap_pyfunction!(audio, m)?)?;
    m.add_function(wrap_pyfunction!(stream, m)?)?;
    m.add_function(wrap_pyfunction!(jsonl, m)?)?;
    m.add_function(wrap_pyfunction!(warc, m)?)?;
    Ok(())
}
