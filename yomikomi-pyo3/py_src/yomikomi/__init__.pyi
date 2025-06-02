# Generated content DO NOT EDIT
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from os import PathLike

@staticmethod
def audio(file):
    """
    Returns a stream that iterates over the pcm data in an audio file.
    """
    pass

@staticmethod
def jsonl(file, *, offset=0, field=None, filters=..., include_if_missing=False):
    """
    Returns a stream that iterates over the text contained in a specific field of a jsonl file.
    """
    pass

@staticmethod
def stream(iterable, *, field=None):
    """
    Returns a stream based on a python iterator. The iterator can either return a whole dictionary
    or if `field` is specified single values which will be embedded in a dictionary with a single
    entry named as per the field argument.
    """
    pass

@staticmethod
def warc(file):
    """
    Returns a stream that iterates over the documents contained in a warc file.
    """
    pass

class JsonFilter:
    @staticmethod
    def eq(field, value, *, include_if_missing=False):
        """ """
        pass

    @staticmethod
    def greater(field, value, *, include_if_missing=False):
        """ """
        pass

    @staticmethod
    def greater_eq(field, value, *, include_if_missing=False):
        """ """
        pass

    @staticmethod
    def lower(field, value, *, include_if_missing=False):
        """ """
        pass

    @staticmethod
    def lower_eq(field, value, *, include_if_missing=False):
        """ """
        pass

    @staticmethod
    def neq(field, value, *, include_if_missing=False):
        """ """
        pass

class StreamIter:
    pass

class YkIterable:
    def batch(self, batch_size, *, return_incomplete_last_batch=False):
        """ """
        pass

    def enumerate(self, field):
        """ """
        pass

    def filter(self, f, *, field=None):
        """
        Filters a stream, the elements are kept if the provided function `f` returns `True` on
        them, otherwise they are discarded. If `field` is specified, the function `f` is only
        passed the value associated to this field rather than a whole dictionary.
        """
        pass

    def filter_key(self, key, *, remove=False):
        """ """
        pass

    def first_slice(self, window_size, *, field=...):
        """ """
        pass

    def key_transform(self, f, *, field):
        """ """
        pass

    def map(self, f):
        """ """
        pass

    def prefetch(self, *, num_threads, buffer_size=None):
        """ """
        pass

    def sliding_window(self, window_size, *, stride=None, field=..., overlap_over_samples=False):
        """ """
        pass

    def tokenize(
        self,
        path,
        *,
        in_field=...,
        out_field=None,
        report_bpb=True,
        include_bos=True,
        include_eos=False,
        bos_id=None,
        eos_id=None
    ):
        """
        Loads a sentencepiece tokenizer, and use it to tokenize the field passed as an argument of
        this function.
        """
        pass
