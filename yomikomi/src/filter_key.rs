use crate::{Result, Stream};

pub struct FilterKey<T> {
    input: T,
    key: String,
    remove: bool,
}

impl<T> FilterKey<T> {
    pub fn new(input: T, key: String, remove: bool) -> Self {
        Self { input, key, remove }
    }
}

impl<T: Stream> Stream for FilterKey<T> {
    fn next(&self) -> Result<Option<crate::Sample>> {
        let mut sample = match self.input.next()? {
            None => return Ok(None),
            Some(sample) => sample,
        };
        let array = match sample.remove(self.key.as_str()) {
            None => crate::bail!("cannot find '{}' in sample", self.key),
            Some(array) => array,
        };
        if !self.remove {
            sample = std::collections::HashMap::from([(self.key.to_string(), array)])
        }
        Ok(Some(sample))
    }
}
