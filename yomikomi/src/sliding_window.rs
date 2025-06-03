use crate::{Array, Result, Sample, Stream};
use std::collections::VecDeque;
use std::sync::Mutex;

struct Buffers {
    samples: VecDeque<Sample>,
    arrays: Vec<Array>,
    total_len_in_arrays: usize,
}

pub struct SlidingWindow<T> {
    input: T,
    buffers: Mutex<Buffers>,
    key: String,
    window_size: usize,
    stride: usize,
    overlap_over_samples: bool,
}

impl<T> SlidingWindow<T> {
    pub fn new(
        input: T,
        window_size: usize,
        stride: usize,
        key: String,
        overlap_over_samples: bool,
    ) -> Result<Self> {
        if window_size == 0 {
            crate::bail!("window_size cannot be 0 in SlidingWindow");
        };
        if stride == 0 {
            crate::bail!("stride cannot be 0 in SlidingWindow");
        };
        if overlap_over_samples && stride != window_size {
            crate::bail!(
                "stride has to match window_size in SlidingWindow with overlap_over_samples"
            );
        }
        let buffers =
            Buffers { samples: VecDeque::new(), arrays: Vec::new(), total_len_in_arrays: 0 };
        let s = Self {
            input,
            buffers: Mutex::new(buffers),
            key,
            window_size,
            stride,
            overlap_over_samples,
        };
        Ok(s)
    }
}

impl<T: Stream> SlidingWindow<T> {
    fn next_no_overlap(&self) -> Result<Option<crate::Sample>> {
        loop {
            {
                let mut buffers = self.buffers.lock()?;
                if let Some(sample) = buffers.samples.pop_front() {
                    return Ok(Some(sample));
                }
            }
            let sample = match self.input.next()? {
                None => return Ok(None),
                Some(sample) => sample,
            };
            let array = match sample.get(self.key.as_str()) {
                None => crate::bail!("no key {} in SlidingWindow", self.key),
                Some(array) => array,
            };
            let size = array.shape().dims1()?;
            let mut start_index = 0;
            let mut buffers = self.buffers.lock()?;
            while start_index + self.window_size <= size {
                let mut new_sample = sample.clone();
                let sub_array = array.narrow(0, start_index, self.window_size)?;
                new_sample.insert(self.key.to_string(), sub_array);
                buffers.samples.push_back(new_sample);
                start_index += self.stride
            }
        }
    }

    fn next_overlap(&self) -> Result<Option<crate::Sample>> {
        loop {
            {
                let mut buffers = self.buffers.lock()?;
                if let Some(sample) = buffers.samples.pop_front() {
                    return Ok(Some(sample));
                }
            }
            let sample = match self.input.next()? {
                None => return Ok(None),
                Some(sample) => sample,
            };
            let array = match sample.get(self.key.as_str()) {
                None => crate::bail!("no key {} in SlidingWindow", self.key),
                Some(array) => array,
            };
            let size = array.shape().dims1()?;
            let mut start_index = 0;
            let mut buffers = self.buffers.lock()?;
            while start_index < size {
                let to_read =
                    usize::min(size - start_index, self.window_size - buffers.total_len_in_arrays);
                let sub_array = array.narrow(0, start_index, to_read)?;
                buffers.arrays.push(sub_array);
                buffers.total_len_in_arrays += to_read;
                if buffers.total_len_in_arrays == self.window_size {
                    // Flush buffers.arrays.
                    let array = if buffers.arrays.len() == 1 {
                        buffers.arrays.remove(0)
                    } else {
                        Array::cat(buffers.arrays.as_slice(), 0)?
                    };
                    buffers.arrays.clear();
                    buffers.total_len_in_arrays = 0;

                    let mut new_sample = sample.clone();
                    new_sample.insert(self.key.to_string(), array);
                    buffers.samples.push_back(new_sample);
                }

                // This should be changed if we started supporting arbitrary strides in overlapping
                // mode.
                start_index += to_read;
            }
        }
    }
}

impl<T: Stream> Stream for SlidingWindow<T> {
    fn next(&self) -> Result<Option<crate::Sample>> {
        if self.overlap_over_samples {
            self.next_overlap()
        } else {
            self.next_no_overlap()
        }
    }
}

pub struct FirstSlice<T> {
    input: T,
    buffers: Mutex<Buffers>,
    key: String,
    window_size: usize,
    pad_with: Option<f64>,
}

impl<T> FirstSlice<T> {
    pub fn new(input: T, window_size: usize, key: String, pad_with: Option<f64>) -> Result<Self> {
        if window_size == 0 {
            crate::bail!("window_size cannot be 0 in FirstSlice");
        };
        let buffers =
            Buffers { samples: VecDeque::new(), arrays: Vec::new(), total_len_in_arrays: 0 };
        let s = Self { input, buffers: Mutex::new(buffers), key, window_size, pad_with };
        Ok(s)
    }
}

impl<T: Stream> Stream for FirstSlice<T> {
    fn next(&self) -> Result<Option<crate::Sample>> {
        loop {
            {
                let mut buffers = self.buffers.lock()?;
                if let Some(sample) = buffers.samples.pop_front() {
                    return Ok(Some(sample));
                }
            }
            let sample = match self.input.next()? {
                None => return Ok(None),
                Some(sample) => sample,
            };
            let array = match sample.get(self.key.as_str()) {
                None => crate::bail!("no key {} in FirstSlice", self.key),
                Some(array) => array,
            };
            let mut new_sample = sample.clone();
            let sub_array = array.resize_pad(0, self.window_size, self.pad_with)?;
            new_sample.insert(self.key.to_string(), sub_array);
            let mut buffers = self.buffers.lock()?;
            buffers.samples.push_back(new_sample);
        }
    }
}
