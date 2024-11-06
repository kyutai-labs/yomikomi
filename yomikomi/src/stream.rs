use crate::{Array, Result};
use std::collections::HashMap;

pub type Sample = HashMap<String, Array>;

pub trait Stream {
    /// Returns the next element from the iterator. This takes as in put a non-mutable reference as
    /// the stream can be queried in a multi-threaded way.
    fn next(&self) -> Result<Option<Sample>>;

    fn filter<P>(self, predicate: P) -> Filter<Self, P>
    where
        Self: Sized,
        P: Fn(&Sample) -> Result<bool>,
    {
        Filter::new(self, predicate)
    }

    fn map<F>(self, f: F) -> Map<Self, F>
    where
        Self: Sized,
        F: Fn(Sample) -> Sample,
    {
        Map::new(self, f)
    }

    fn and_then<F>(self, f: F) -> AndThen<Self, F>
    where
        Self: Sized,
        F: Fn(Sample) -> Result<Option<Sample>>,
    {
        AndThen::new(self, f)
    }

    fn collect(&self) -> Result<Vec<Sample>> {
        let mut samples = vec![];
        while let Some(sample) = self.next()? {
            samples.push(sample)
        }
        Ok(samples)
    }
}

pub struct Filter<S, P>
where
    P: Fn(&Sample) -> Result<bool>,
    S: Stream,
{
    src: S,
    pred: P,
}

impl<S: Stream, P: Fn(&Sample) -> Result<bool>> Filter<S, P> {
    fn new(src: S, pred: P) -> Self {
        Self { src, pred }
    }
}

impl<S: Stream, P: Fn(&Sample) -> Result<bool>> Stream for Filter<S, P> {
    fn next(&self) -> Result<Option<Sample>> {
        while let Some(value) = self.src.next()? {
            if (self.pred)(&value)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }
}

pub struct Map<S, F>
where
    F: Fn(Sample) -> Sample,
    S: Stream,
{
    src: S,
    f: F,
}

impl<S: Stream, F: Fn(Sample) -> Sample> Map<S, F> {
    fn new(src: S, f: F) -> Self {
        Self { src, f }
    }
}

impl<S: Stream, F: Fn(Sample) -> Sample> Stream for Map<S, F> {
    fn next(&self) -> Result<Option<Sample>> {
        match self.src.next()? {
            None => Ok(None),
            Some(v) => Ok(Some((self.f)(v))),
        }
    }
}

pub struct AndThen<S, F>
where
    F: Fn(Sample) -> Result<Option<Sample>>,
    S: Stream,
{
    src: S,
    f: F,
}

impl<S: Stream, F: Fn(Sample) -> Result<Option<Sample>>> AndThen<S, F> {
    fn new(src: S, f: F) -> Self {
        Self { src, f }
    }
}

impl<S: Stream, F: Fn(Sample) -> Result<Option<Sample>>> Stream for AndThen<S, F> {
    fn next(&self) -> Result<Option<Sample>> {
        loop {
            match self.src.next()? {
                None => return Ok(None),
                Some(v) => match (self.f)(v)? {
                    None => (),
                    Some(v) => return Ok(Some(v)),
                },
            }
        }
    }
}

impl Stream for Box<dyn Stream + Send + Sync> {
    fn next(&self) -> Result<Option<Sample>> {
        self.as_ref().next()
    }
}

pub struct StreamI<Item: Into<Sample>> {
    inner: std::sync::Mutex<Box<dyn Iterator<Item = Item>>>,
}

impl<I: Into<Sample>> Stream for StreamI<I> {
    fn next(&self) -> Result<Option<Sample>> {
        let mut inner = self.inner.lock()?;
        Ok(inner.next().map(|i| i.into()))
    }
}

pub fn from_iter<Item: Into<Sample>, I: Iterator<Item = Item> + 'static>(
    inner: I,
) -> StreamI<Item> {
    StreamI { inner: std::sync::Mutex::new(Box::new(inner)) }
}

pub struct StreamA<Item: Into<Array>> {
    inner: std::sync::Mutex<Box<dyn Iterator<Item = Item>>>,
    field: String,
}

impl<I: Into<Array>> Stream for StreamA<I> {
    fn next(&self) -> Result<Option<Sample>> {
        let mut inner = self.inner.lock()?;
        Ok(inner.next().map(|v| {
            let mut hm = HashMap::new();
            hm.insert(self.field.clone(), v.into());
            hm
        }))
    }
}

pub fn from_iter_a<Item: Into<Array>, I: Iterator<Item = Item> + 'static>(
    inner: I,
    field: impl ToString,
) -> StreamA<Item> {
    StreamA { inner: std::sync::Mutex::new(Box::new(inner)), field: field.to_string() }
}
