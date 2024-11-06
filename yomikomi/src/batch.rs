use crate::{Array, Result, Stream};
use std::collections::HashMap;

pub struct Batch<T> {
    input: T,
    batch_size: usize,
    return_incomplete_last_batch: bool,
}

impl<T> Batch<T> {
    pub fn new(input: T, batch_size: usize, return_incomplete_last_batch: bool) -> Result<Self> {
        if batch_size == 0 {
            crate::bail!("batch_size cannot be 0 in Batch");
        };
        let s = Self { input, batch_size, return_incomplete_last_batch };
        Ok(s)
    }
}

impl<T: Stream> Stream for Batch<T> {
    fn next(&self) -> Result<Option<crate::Sample>> {
        let first_sample = match self.input.next()? {
            None => return Ok(None),
            Some(sample) => sample,
        };
        let mut batch: HashMap<_, _> = first_sample
            .into_iter()
            .map(|(k, v)| {
                let mut vec = Vec::with_capacity(self.batch_size);
                vec.push(v);
                (k, vec)
            })
            .collect();
        let mut cnt = 1;
        for _i in 1..self.batch_size {
            let mut sample = match self.input.next()? {
                None => break,
                Some(sample) => sample,
            };
            batch.iter_mut().for_each(|(k, vec)| {
                if let Some(v) = sample.remove(k.as_str()) {
                    vec.push(v)
                }
            });
            cnt += 1;
        }
        if cnt < self.batch_size && !self.return_incomplete_last_batch {
            Ok(None)
        } else {
            let batch = batch
                .into_iter()
                .map(|(k, v)| {
                    let v = Array::stack(v.as_slice(), 0)?;
                    Ok((k, v))
                })
                .collect::<Result<HashMap<_, _>>>()?;
            Ok(Some(batch))
        }
    }
}
