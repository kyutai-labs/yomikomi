use crate::{Array, Error, Result, Sample};
use std::io::prelude::*;
use std::sync::Mutex;

// Equivalent to std::io::Lines but keeps track of how many bytes have been read.
struct Lines {
    buf_reader: std::io::BufReader<Box<dyn Read + Send + Sync>>,
    bytes_read: usize,
    lines_read: usize,
}

impl Lines {
    fn new(read: impl Read + 'static + Send + Sync) -> Self {
        Self { buf_reader: std::io::BufReader::new(Box::new(read)), bytes_read: 0, lines_read: 0 }
    }

    fn read_line(&mut self) -> std::io::Result<Option<String>> {
        let mut buf = String::new();
        match self.buf_reader.read_line(&mut buf)? {
            0 => Ok(None),
            n => {
                self.bytes_read += n;
                self.lines_read += 1;
                if buf.ends_with('\n') {
                    buf.pop();
                    if buf.ends_with('\r') {
                        buf.pop();
                    }
                }
                Ok(Some(buf))
            }
        }
    }

    fn bytes_read(&self) -> usize {
        self.bytes_read
    }

    fn lines_read(&self) -> usize {
        self.lines_read
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum FilterCmp {
    LowerEq,
    Lower,
    Eq,
    Neq,
    Greater,
    GreaterEq,
}

#[derive(Clone, Debug)]
pub struct Filter {
    pub field: String,
    pub cmp: FilterCmp,
    pub value: f64,
    pub include_if_missing: bool,
}

pub struct FileReader {
    fields: Option<Vec<String>>,
    lines: Mutex<Lines>,
    path: std::path::PathBuf,
    filters: Vec<Filter>,
    initial_offset: u64,
    include_if_missing: bool,
}

fn wrap_err(e: std::io::Error, p: &std::path::Path) -> Error {
    Error::WithPath { inner: Box::new(e.into()), path: p.to_path_buf() }.bt()
}

struct Zstd(std::sync::Mutex<zstd::Decoder<'static, std::io::BufReader<std::fs::File>>>);

impl Read for Zstd {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().read(buf)
    }
}

impl FileReader {
    pub fn new<P: AsRef<std::path::Path>>(
        path: P,
        offset: u64,
        field: String,
        filters: Vec<Filter>,
    ) -> Result<Self> {
        Self::new_multi(path, offset, Some(vec![field]), filters, false)
    }

    pub fn new_multi<P: AsRef<std::path::Path>>(
        path: P,
        offset: u64,
        fields: Option<Vec<String>>,
        filters: Vec<Filter>,
        include_if_missing: bool,
    ) -> Result<Self> {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path).map_err(|e| wrap_err(e, path))?;
        // When the file is compressed, we have to decompress all the stream and skip the first
        // offset bytes which is costly compared to a call to `Seek`. To get around this, we would
        // need some form of block compression and being able to skip whole blocks at once.
        let lines = if path.extension().and_then(|s| s.to_str()) == Some("gz") {
            let mut input = flate2::read::MultiGzDecoder::new(file);
            std::io::copy(
                &mut std::io::Read::by_ref(&mut input).take(offset),
                &mut std::io::sink(),
            )?;
            Lines::new(input)
        } else if path.extension().and_then(|s| s.to_str()) == Some("zstd") {
            let mut input = zstd::Decoder::new(file).map_err(|e| wrap_err(e, path))?;
            std::io::copy(&mut input.by_ref().take(offset), &mut std::io::sink())?;
            Lines::new(Zstd(std::sync::Mutex::new(input)))
        } else {
            file.seek(std::io::SeekFrom::Start(offset)).map_err(|e| wrap_err(e, path))?;
            Lines::new(file)
        };
        Ok(Self {
            fields,
            lines: Mutex::new(lines),
            path: path.to_path_buf(),
            filters,
            initial_offset: offset,
            include_if_missing,
        })
    }
}

impl crate::Stream for FileReader {
    fn next(&self) -> Result<Option<Sample>> {
        loop {
            // Note that line_index is only counting starting from the start offset if any.
            let (bytes_read, line_index, line) = {
                let mut lines = self.lines.lock()?;
                let line = lines.read_line().map_err(|e| wrap_err(e, &self.path))?;
                (lines.bytes_read(), lines.lines_read().saturating_sub(1), line)
            };
            let line = match line {
                None => return Ok(None),
                Some(line) => line,
            };
            let line: serde_json::Value = match serde_json::from_str(&line) {
                Err(err) => {
                    eprintln!("{:?}:{line_index} invalid json {err}", self.path);
                    continue;
                }
                Ok(text) => text,
            };
            let mut all_ok = true;
            for filter in self.filters.iter() {
                let line_value = match line.get(filter.field.as_str()) {
                    Some(v) => v,
                    None => {
                        if filter.include_if_missing {
                            continue;
                        }
                        eprintln!("{:?}:{line_index} missing '{}' field", self.path, filter.field);
                        all_ok = false;
                        break;
                    }
                };
                let line_value = match line_value {
                    serde_json::Value::Number(v) => v.as_f64(),
                    serde_json::Value::String(v) => v.parse::<f64>().ok(),
                    _ => None,
                };
                let line_value = match line_value {
                    Some(v) => v,
                    None => {
                        eprintln!(
                            "{:?}:{line_index} wrong type for '{}' field",
                            self.path, filter.field
                        );
                        all_ok = false;
                        break;
                    }
                };
                let ok = match &filter.cmp {
                    FilterCmp::Eq => line_value == filter.value,
                    FilterCmp::Neq => line_value != filter.value,
                    FilterCmp::LowerEq => line_value <= filter.value,
                    FilterCmp::Lower => line_value < filter.value,
                    FilterCmp::GreaterEq => line_value >= filter.value,
                    FilterCmp::Greater => line_value > filter.value,
                };
                if !ok {
                    all_ok = false;
                    break;
                }
            }
            if !all_ok {
                continue;
            }
            let mut sample = std::collections::HashMap::from([
                ("bytes_read".to_string(), Array::from(bytes_read as i64)),
                ("line_index".to_string(), Array::from(line_index as i64)),
                ("offset".to_string(), Array::from(self.initial_offset as i64 + bytes_read as i64)),
            ]);
            let skip = 'skip: {
                match &self.fields {
                    None => {
                        let line = match line.as_object() {
                            None => {
                                eprintln!("{:?}:{line_index} not a json object", self.path);
                                break 'skip true;
                            }
                            Some(line) => line,
                        };
                        for (field, value) in line.iter() {
                            let value = match value {
                                serde_json::value::Value::String(text) => {
                                    Array::from(text.as_bytes().to_vec())
                                }
                                serde_json::value::Value::Number(n) => {
                                    match (n.as_i64(), n.as_u64(), n.as_f64()) {
                                        (Some(v), _, _) => Array::from(v),
                                        (_, Some(v), _) => Array::from(v as i64),
                                        (_, _, Some(v)) => Array::from(v),
                                        (None, None, None) => {
                                            eprintln!(
                                                "{:?}:{line_index} weird '{field}'",
                                                self.path
                                            );
                                            break 'skip true;
                                        }
                                    }
                                }
                                _ => continue,
                            };
                            sample.insert(field.to_string(), value);
                        }
                    }
                    Some(fields) => {
                        for field in fields.iter() {
                            let data = match line.get(field) {
                                None => {
                                    if self.include_if_missing {
                                        continue;
                                    }
                                    eprintln!("{:?}:{line_index} missing '{field}'", self.path);
                                    break 'skip true;
                                }
                                Some(serde_json::value::Value::String(text)) => {
                                    Array::from(text.as_bytes().to_vec())
                                }
                                Some(serde_json::value::Value::Number(n)) => {
                                    match (n.as_i64(), n.as_u64(), n.as_f64()) {
                                        (Some(v), _, _) => Array::from(v),
                                        (_, Some(v), _) => Array::from(v as i64),
                                        (_, _, Some(v)) => Array::from(v),
                                        (None, None, None) => {
                                            eprintln!(
                                                "{:?}:{line_index} weird '{field}'",
                                                self.path
                                            );
                                            break 'skip true;
                                        }
                                    }
                                }
                                Some(_) => {
                                    eprintln!(
                                        "{:?}:{line_index} invalid type for '{field}'",
                                        self.path
                                    );
                                    break 'skip true;
                                }
                            };
                            sample.insert(field.to_string(), data);
                        }
                    }
                }
                false
            };
            if skip {
                continue;
            }
            return Ok(Some(sample));
        }
    }
}
