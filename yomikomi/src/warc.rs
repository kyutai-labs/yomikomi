use crate::{Array, Error, Result, Stream};
use std::io::{BufRead, Read};
use std::str::FromStr;
use std::sync::Mutex;

#[derive(Debug)]
pub struct Document {
    content_length: usize,
    url: String,
    date: String,
    hash: String,
    type_: String,
    content: String,
}

struct LineReader<T: Read> {
    line: Vec<u8>,
    reader: std::io::BufReader<T>,
}

#[repr(usize)]
enum Header {
    ContentLength = 0,
    Url = 1,
    Date = 2,
    Hash = 3,
    Type = 4,
}

fn wrap_err(e: std::io::Error, p: &std::path::Path) -> Error {
    Error::WithPath { inner: Box::new(e.into()), path: p.to_path_buf() }.bt()
}

impl<T: Read> LineReader<T> {
    fn new(r: T) -> Self {
        Self { line: Vec::with_capacity(1024), reader: std::io::BufReader::new(r) }
    }

    fn next_line(&mut self) -> std::io::Result<Option<&[u8]>> {
        self.line.clear();
        if self.reader.read_until(b'\n', &mut self.line)? == 0 {
            return Ok(None);
        }
        if self.line.last() == Some(&b'\n') {
            self.line.pop();
        }
        if self.line.last() == Some(&b'\r') {
            self.line.pop();
        }
        Ok(Some(&self.line))
    }

    fn read_exact(&mut self, len: usize) -> std::io::Result<Vec<u8>> {
        let mut content = vec![0u8; len];
        self.reader.read_exact(&mut content)?;
        Ok(content)
    }

    fn read_document(&mut self) -> Result<Option<Document>> {
        loop {
            match self.next_line()? {
                None => return Ok(None),
                Some(b"WARC/1.0") => break,
                Some(line) => {
                    eprintln!("SKIPPING <{}>", String::from_utf8_lossy(line));
                    continue;
                }
            }
        }
        let mut headers = vec![None; 5];
        loop {
            let line = match self.next_line()? {
                None => return Ok(None),
                Some(b"") => break,
                Some(line) => line,
            };
            let mut tokens = line.split(|&v| v == b' ');
            let key = match tokens.next() {
                Some(b"Content-Length:") => Header::ContentLength,
                Some(b"WARC-Target-URI:") => Header::Url,
                Some(b"WARC-Date:") => Header::Date,
                Some(b"WARC-Block-Digest:") => Header::Hash,
                Some(b"WARC-Type:") => Header::Type,
                _ => continue,
            };
            if let Some(value) = tokens.next() {
                headers[key as usize] = Some(String::from_utf8_lossy(value).into_owned())
            }
        }
        let content_length = match &headers[Header::ContentLength as usize] {
            None => crate::bail!("no content_length"),
            Some(len) => usize::from_str(len)?,
        };
        let url = headers[Header::Url as usize].take().unwrap_or_default();
        let date = headers[Header::Date as usize].take().unwrap_or_default();
        let hash = headers[Header::Hash as usize].take().unwrap_or_default();
        let type_ = headers[Header::Type as usize].take().unwrap_or_default();
        let content = self.read_exact(content_length)?;
        let mut crlf2 = [0u8; 4];
        self.reader.read_exact(&mut crlf2)?;
        if &crlf2 != b"\r\n\r\n" {
            crate::bail!("unexpected end of block")
        }
        // TODO: Maybe try to find a way to do the following in place, or instead just
        // remove all \r without conditioning on the \n?
        let content = String::from_utf8(content).map_err(Error::wrap)?.replace("\r\n", "\n");
        Ok(Some(Document { content_length, url, date, hash, type_, content }))
    }
}

enum FileOrCompressed {
    File(std::fs::File),
    Gz(flate2::read::MultiGzDecoder<std::fs::File>),
    Zstd(zstd::Decoder<'static, std::io::BufReader<std::fs::File>>),
}

impl Read for FileOrCompressed {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::File(file) => file.read(buf),
            Self::Gz(gz) => gz.read(buf),
            Self::Zstd(zstd) => zstd.read(buf),
        }
    }
}

pub struct FileReader {
    reader: Mutex<LineReader<FileOrCompressed>>,
}

impl FileReader {
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file_or_compressed = if path.extension().and_then(|s| s.to_str()) == Some("gz") {
            let input = std::fs::File::open(path).map_err(|e| wrap_err(e, path))?;
            let input = flate2::read::MultiGzDecoder::new(input);
            FileOrCompressed::Gz(input)
        } else if path.extension().and_then(|s| s.to_str()) == Some("zstd") {
            let input = std::fs::File::open(path).map_err(|e| wrap_err(e, path))?;
            let input = zstd::Decoder::new(input).map_err(|e| wrap_err(e, path))?;
            FileOrCompressed::Zstd(input)
        } else {
            let file = std::fs::File::open(path).map_err(|e| wrap_err(e, path))?;
            FileOrCompressed::File(file)
        };
        let reader = LineReader::new(file_or_compressed);
        Ok(Self { reader: Mutex::new(reader) })
    }
}

impl Stream for FileReader {
    fn next(&self) -> Result<Option<crate::Sample>> {
        let d = {
            let mut reader = self.reader.lock()?;
            loop {
                let document = match reader.read_document()? {
                    None => return Ok(None),
                    Some(document) => document,
                };
                if document.type_ != "warcinfo" {
                    break document;
                }
            }
        };
        let sample = std::collections::HashMap::from([
            ("content_length".to_string(), Array::from(d.content_length as i64)),
            ("url".to_string(), Array::from(d.url.into_bytes())),
            ("date".to_string(), Array::from(d.date.into_bytes())),
            ("hash".to_string(), Array::from(d.hash.into_bytes())),
            ("type_".to_string(), Array::from(d.type_.into_bytes())),
            ("content".to_string(), Array::from(d.content.into_bytes())),
        ]);
        Ok(Some(sample))
    }
}
