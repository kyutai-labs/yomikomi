use crate::{Array, Error, Result, Stream};
use sentencepiece::SentencePieceProcessor;
use std::sync::{Arc, Mutex};

pub struct Tokenize<T> {
    spp: Arc<SentencePieceProcessor>,
    input: T,
    in_key: String,
    out_key: String,
    nl_id: u32,
    tokens_and_chars: Option<Mutex<(usize, usize)>>,
    include_bos: bool,
    include_eos: bool,
}

impl<T> Tokenize<T> {
    pub fn new<P: AsRef<std::path::Path>>(
        path: P,
        input: T,
        in_key: String,
        out_key: String,
        report_bpb: bool,
        include_bos: bool,
        include_eos: bool,
    ) -> Result<Self> {
        let spp = SentencePieceProcessor::open(path).map_err(Error::wrap)?;
        let nl_id = match spp.encode("\n").map_err(Error::wrap)?.last() {
            None => crate::bail!("no specific token id for newline"),
            Some(p) => p.id,
        };
        let tokens_and_chars = if report_bpb { Some(Mutex::new((0, 0))) } else { None };
        Ok(Self {
            spp: Arc::new(spp),
            input,
            in_key,
            out_key,
            nl_id,
            tokens_and_chars,
            include_bos,
            include_eos,
        })
    }
}

impl<T: Stream> Stream for Tokenize<T> {
    fn next(&self) -> Result<Option<crate::Sample>> {
        let sample = self.input.next()?;
        let mut sample = match sample {
            None => return Ok(None),
            Some(sample) => sample,
        };
        let values = match sample.get(self.in_key.as_str()) {
            Some(values) => values,
            None => {
                crate::bail!("missing key {}", self.in_key)
            }
        };
        let values = match values.values::<u8>() {
            Err(_) => crate::bail!("tokenizer error, expected u8, got {:?}", values.dtype()),
            Ok(values) => values,
        };
        let text = String::from_utf8_lossy(values);
        let mut all_tokens = Vec::new();
        if self.include_bos {
            if let Some(bos_id) = self.spp.bos_id() {
                all_tokens.push(bos_id)
            }
        }
        let mut bpb = None;
        for (idx, text) in text.split('\n').enumerate() {
            #[allow(clippy::collapsible_if)]
            if idx > 0 {
                all_tokens.push(self.nl_id)
            }
            let tokens = match self.spp.encode(text) {
                Ok(tokens) => tokens,
                Err(err) => {
                    eprintln!("tokenizer encode error {err:?}");
                    continue;
                }
            };
            if let Some(tokens_and_chars) = &self.tokens_and_chars {
                let mut tokens_and_chars = tokens_and_chars.lock()?;
                tokens_and_chars.0 += tokens.len();
                tokens_and_chars.1 += text.len();
                bpb = Some(tokens_and_chars.0 as f64 / tokens_and_chars.1 as f64 / f64::ln(2.))
            };
            for token in tokens {
                all_tokens.push(token.id)
            }
        }
        if self.include_eos {
            if let Some(eos_id) = self.spp.eos_id() {
                all_tokens.push(eos_id)
            }
        }
        let all_tokens = Array::from(all_tokens);
        sample.insert(self.out_key.to_string(), all_tokens);
        if let Some(bpb) = bpb {
            sample.insert("c_bpb".to_string(), Array::from(bpb));
        }
        Ok(Some(sample))
    }
}
