import argparse
import json
import numpy as np
import yomikomi as yk
import os
from functools import reduce

print(os.getpid())

# Reference python implementation that we compare yk too.
def tokenize(text, tokenizer, bos=True, eos=True):
    nl_piece = tokenizer.encode("\n")[-1]
    tokens = tokenizer.encode(text.split("\n"))
    tokens = reduce(lambda a, b: a + [nl_piece] + b, tokens)
    return tokens


def detokenize(tokens, tokenizer):
    text = tokenizer.decode(tokens)
    return text.replace("\n ", "\n")


def jsonl_iterator(filename, remove_example=lambda x: False):
    while True:
        fin = open(filename)
        for line in fin:
            data = json.loads(line)
            if not remove_example(data):
                yield data
        break


def token_iterator(iterator, tokenizer, field="text"):
    for data in iterator:
        tokens = tokenize(data[field], tokenizer)
        yield {"tokens": tokens}


def batch_iterator(iterator, bsz, csz):
    tokens = []
    n = bsz * (csz + 1)
    for data in iterator:
        tokens += data["tokens"]
        if len(tokens) > n:
            a = len(tokens) // n
            x = np.asarray(tokens[: a * n]).reshape(a, bsz, -1)
            for i in range(a):
                yield x[i, :, :]
            tokens = tokens[a * n :]


def load_data(filename, tokenizer, bsz, csz, fname="quality", threshold=0.2):
    tokens = []
    n = bsz * (csz + 1)
    while True:
        fin = open(filename)
        for line in fin:
            data = json.loads(line)
            if fname in data and data[fname] < threshold:
                continue
            t = tokenize(data["text"], tokenizer)
            tokens += t
            while len(tokens) >= n:
                yield np.asarray(tokens[:n]).reshape(bsz, csz + 1)
                tokens = tokens[n:]


parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--datafile", type=str)
parser.add_argument("--fname", type=str, default="quality")
parser.add_argument("--threshold", type=float, default=0.2)
parser.add_argument("--csz", type=int, default=512)
parser.add_argument("--bsz", type=int, default=8)
parser.add_argument("--ntest", type=int, default=123)

args = parser.parse_args()

print(f"pid: {os.getpid()}")


def comparison_test(args):
    if args.ntest == 0:
        return
    # Python dataset loader
    import sentencepiece

    tokenizer = sentencepiece.SentencePieceProcessor(model_file=args.tokenizer)
    dataset = jsonl_iterator(
        args.datafile, lambda x: args.fname in x and x[args.fname] < args.threshold
    )
    dataset = token_iterator(dataset, tokenizer)
    dataset = batch_iterator(dataset, args.bsz, args.csz)

    # dataset = load_data(args.datafile, tokenizer, args.bsz, args.csz, fname=args.fname, threshold=args.threshold)

    filters = [yk.JsonFilter.greater(args.fname, 20, include_if_missing=True)]
    # ****CAUTION****
    # It is important for the `prefetch` not to be after a section that is mutex protected when using 2 threads or more,
    # i.e. having a `prefetch` just after `jsonl` would likely result in no speedup.
    yk_ds = (
        yk.jsonl(args.datafile, field=None, filters=filters)
        .tokenize(args.tokenizer, in_field="text", out_field="text", include_bos=False)
        .prefetch(num_threads=1, buffer_size=4)
        .sliding_window(args.csz + 1, field="text", overlap_over_samples=True)
        .batch(args.bsz)
    )

    nbatches = 0
    nrows = 0
    dataset = dataset.__iter__()
    yk_ds = yk_ds.__iter__()
    for i, (batch_py, batch_yk) in enumerate(zip(dataset, yk_ds, strict=True)):
        if i == args.ntest:
            break
        batch_yk = batch_yk["text"]
        if batch_py.shape != batch_yk.shape:
            raise ValueError(f"shape mismatch {i}: {batch_py.shape} {batch_yk.shape}")
        diff = np.abs(batch_py - batch_yk).sum()
        if diff != 0:
            print("batch python\n", batch_py)
            print("batch yk\n", batch_yk)
            raise ValueError(f"value mismatch {i}: {batch_py.shape} {batch_yk.shape}")
        nbatches += 1
        nrows += batch_py.shape[0]
    print(f"test success, batches: {nbatches}, rows: {nrows}")


comparison_test(args)


for v in yk.stream([1, 2, np.array([1, 2, 3, 4, 5])], field="foo").key_transform(
    lambda x: x + 1, field="foo"
):
    print(v)
for v in yk.stream([1, 2, np.array([1, 2, 3, 4, 5])], field="foo"):
    print(v)
for v in (
    yk.stream([[i] for i in range(42)], field="foo")
    .sliding_window(7, field="foo", overlap_over_samples=True)
    .batch(2)
    .enumerate("idx")
):
    print(v)
