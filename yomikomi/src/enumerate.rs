use crate::{Array, Result, Stream};
use std::sync::atomic::AtomicI64;

pub struct Enumerate<T> {
    input: T,
    index: AtomicI64,
    field: String,
}

impl<T> Enumerate<T> {
    pub fn new(input: T, field: String) -> Self {
        Self { input, index: AtomicI64::new(0), field }
    }
}

impl<T: Stream> Stream for Enumerate<T> {
    fn next(&self) -> Result<Option<crate::Sample>> {
        let mut sample = match self.input.next()? {
            None => return Ok(None),
            Some(sample) => sample,
        };
        let index = self.index.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let index = Array::from(index);
        sample.insert(self.field.to_string(), index);
        Ok(Some(sample))
    }
}
