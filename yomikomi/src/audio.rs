use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::{Result, Sample};
use std::sync::{Arc, Mutex};

struct State {
    decoder: Box<dyn symphonia::core::codecs::Decoder>,
    format: Box<dyn symphonia::core::formats::FormatReader>,
}

pub struct FileReader {
    path: std::path::PathBuf,
    track_id: u32,
    state: Arc<Mutex<State>>,
}

impl FileReader {
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let src = std::fs::File::open(path)?;
        let mss = MediaSourceStream::new(Box::new(src), Default::default());
        let mut hint = Hint::new();
        if let Some(extension) = path.extension().and_then(|v| v.to_str()) {
            hint.with_extension(extension);
        }

        // Use the default options for metadata and format readers.
        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();

        // Probe the media source.
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .map_err(|e| crate::Error::wrap(e).with_path(path))?;

        // Get the instantiated format reader.
        let format = probed.format;

        // Find the first audio track with a known (decodeable) codec.
        let track =
            format.tracks().iter().find(|t| t.codec_params.codec != CODEC_TYPE_NULL).ok_or_else(
                || crate::Error::Msg("no supported tracks".to_string()).with_path(path),
            )?;

        // Use the default options for the decoder.
        let dec_opts: DecoderOptions = Default::default();

        // Create a decoder for the track.
        let decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &dec_opts)
            .map_err(|e| crate::Error::wrap(e).with_path(path))?;

        // Store the track identifier, it will be used to filter packets.
        let track_id = track.id;
        let state = State { decoder, format };
        Ok(Self { path: path.to_path_buf(), track_id, state: Arc::new(Mutex::new(state)) })
    }
}

fn conv<T>(data: std::borrow::Cow<symphonia::core::audio::AudioBuffer<T>>) -> Vec<f32>
where
    T: symphonia::core::sample::Sample,
    f32: symphonia::core::conv::FromSample<T>,
{
    use symphonia::core::conv::FromSample;
    data.planes().planes()[0].iter().map(|v| f32::from_sample(*v)).collect::<Vec<_>>()
}

impl crate::Stream for FileReader {
    fn next(&self) -> Result<Option<Sample>> {
        let mut s = self.state.lock()?;
        // Get the next packet from the media format.
        let packet = match s.format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(ioerr)) if ioerr.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None)
            }
            Err(err) => {
                crate::bail!("{:?} next_packet: {err}", self.path)
            }
        };

        // Consume any new metadata that has been read since the last packet.
        while !s.format.metadata().is_latest() {
            // Pop the old head of the metadata queue.
            s.format.metadata().pop();

            // Consume the new metadata at the head of the metadata queue.
        }

        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id() != self.track_id {
            crate::bail!("incorrect track-id");
        }

        // Decode the packet into audio samples.
        match s.decoder.decode(&packet) {
            Ok(decoded) => {
                let pcm_data = match decoded {
                    symphonia::core::audio::AudioBufferRef::F32(data) => {
                        data.planes().planes()[0].to_vec()
                    }
                    symphonia::core::audio::AudioBufferRef::U8(data) => conv(data),
                    symphonia::core::audio::AudioBufferRef::U16(data) => conv(data),
                    symphonia::core::audio::AudioBufferRef::U24(data) => conv(data),
                    symphonia::core::audio::AudioBufferRef::U32(data) => conv(data),
                    symphonia::core::audio::AudioBufferRef::S8(data) => conv(data),
                    symphonia::core::audio::AudioBufferRef::S16(data) => conv(data),
                    symphonia::core::audio::AudioBufferRef::S24(data) => conv(data),
                    symphonia::core::audio::AudioBufferRef::S32(data) => conv(data),
                    symphonia::core::audio::AudioBufferRef::F64(data) => conv(data),
                };
                let pcm_data = crate::Array::from(pcm_data);
                let mut sample = Sample::new();
                sample.insert("pcm".to_string(), pcm_data);
                Ok(Some(sample))
            }
            Err(err) => {
                // An unrecoverable error occurred, halt decoding.
                crate::bail!("{:?} decode: {}", self.path, err);
            }
        }
    }
}
