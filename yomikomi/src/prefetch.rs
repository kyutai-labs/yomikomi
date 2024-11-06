use crate::{Result, Sample, Stream};
use std::sync::{Arc, Mutex};

struct WorkerShared<T> {
    input: T,
    sender: std::sync::mpsc::SyncSender<Result<Option<Sample>>>,
}

struct Worker<T> {
    shared: Arc<WorkerShared<T>>,
}

impl<T: Stream> Worker<T> {
    fn run(&self) {
        loop {
            let next = self.shared.input.next();
            let is_done = next.as_ref().map_or(true, |v| v.is_none());
            match self.shared.sender.send(next) {
                Err(_) => break,
                Ok(next) => next,
            };
            if is_done {
                break;
            }
        }
    }
}

struct PreFetchInner {
    receiver: std::sync::mpsc::Receiver<Result<Option<Sample>>>,
    remaining_threads: usize,
}

pub struct PreFetch {
    inner: Mutex<PreFetchInner>,
}

impl PreFetch {
    pub fn new<T: Stream + Send + Sync + 'static>(
        input: T,
        num_threads: usize,
        buffer_size: usize,
    ) -> Result<Self> {
        if num_threads == 0 {
            crate::bail!("num_threads cannot be 0 in PreFetch");
        };
        if buffer_size < num_threads {
            crate::bail!("buffer-size {buffer_size} cannot be smaller than num_threads {num_threads} in PreFetch");
        };
        let (sender, receiver) = std::sync::mpsc::sync_channel(buffer_size);
        let shared = Arc::new(WorkerShared { sender, input });
        for _thread_id in 0..num_threads {
            let worker = Worker { shared: shared.clone() };
            std::thread::spawn(move || worker.run());
        }
        let inner = PreFetchInner { receiver, remaining_threads: num_threads };
        Ok(Self { inner: Mutex::new(inner) })
    }
}

impl Stream for PreFetch {
    fn next(&self) -> Result<Option<Sample>> {
        let mut inner = self.inner.lock()?;
        loop {
            let msg = match inner.receiver.recv() {
                Ok(msg) => msg,
                Err(_) => return Ok(None),
            };
            match msg {
                Err(err) => {
                    inner.remaining_threads = inner.remaining_threads.saturating_sub(1);
                    return Err(err);
                }
                Ok(None) => {
                    inner.remaining_threads = inner.remaining_threads.saturating_sub(1);
                    if inner.remaining_threads == 0 {
                        return Ok(None);
                    }
                }
                Ok(Some(v)) => return Ok(Some(v)),
            }
        }
    }
}
