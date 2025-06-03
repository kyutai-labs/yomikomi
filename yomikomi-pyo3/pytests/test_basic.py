import os
import numpy as np
import yomikomi as yk


def test_jsonl():
    pwd = os.path.realpath(__file__)
    current_dir = os.path.dirname(pwd)
    ds = yk.jsonl(os.path.join(current_dir, "sample.jsonl"), field="text")
    ds = list(ds)
    assert "".join([chr(c) for c in ds[0]["text"]]) == "this is a text"
    assert "".join([chr(c) for c in ds[2]["text"]]) == "this is a text3"
    assert ds[0].get("quality") is None

    ds = yk.jsonl(os.path.join(current_dir, "sample.jsonl"), field=[])
    ds = list(ds)
    assert "".join([chr(c) for c in ds[0]["text"]]) == "this is a text"
    assert "".join([chr(c) for c in ds[2]["text"]]) == "this is a text3"
    assert ds[0]["quality"] == 3.14
    assert ds[2]["quality"] == 299792458

def test_range():
    ds = yk.stream(range(10), field="foo")
    values = np.array([d["foo"] for d in ds]).flatten()
    np.testing.assert_equal(values, range(10))


def test_scalar():
    ds = yk.stream(range(10), field="bar")
    ds = list(ds)
    assert isinstance(ds[0], dict)
    assert isinstance(ds[0]["bar"], int)
    ds = yk.stream([float(r) for r in range(10)], field="bar")
    ds = list(ds)
    assert isinstance(ds[0], dict)
    assert isinstance(ds[0]["bar"], float)


def test_filter():
    ds = yk.stream(range(10), field="foo").filter(lambda x: x["foo"] % 2 == 0)
    values = np.array([d["foo"] for d in ds]).flatten()
    np.testing.assert_equal(values, range(0, 10, 2))
    ds = ds.key_transform(lambda x: x // 2, field="foo")
    values = np.array([d["foo"] for d in ds]).flatten()
    np.testing.assert_equal(values, range(5))
    values = np.array([d["foo"] for d in ds]).flatten()
    np.testing.assert_equal(values, range(5))


def test_map():
    ds = yk.stream(range(10), field="foo")
    ds = ds.map(
        lambda x: {"bar": np.array([x["foo"]] * 2)} if x["foo"] % 3 == 0 else None
    )
    values = list(ds)
    np.testing.assert_equal(
        values,
        [
            {"bar": np.array([0, 0])},
            {"bar": np.array([3, 3])},
            {"bar": np.array([6, 6])},
            {"bar": np.array([9, 9])},
        ],
    )

def test_prefetch():
    ds = yk.stream(range(4), field="foo").prefetch(num_threads=1)
    values = list(ds)
    np.testing.assert_equal(
        values,
        [
            {"foo": 0},
            {"foo": 1},
            {"foo": 2},
            {"foo": 3},
        ],
    )
    ds = yk.stream(range(4), field="foo").filter(lambda x: x % 2 == 1, field="foo").prefetch(num_threads=1)
    values = list(ds)
    np.testing.assert_equal(
        values,
        [
            {"foo": 1},
            {"foo": 3},
        ],
    )


def test_first():
    ds = yk.stream([[c] for c in range(3)], field="foo").first_slice(
        2, field="foo", pad_with=42
    )
    values = [d["foo"] for d in ds]
    np.testing.assert_equal(values, [[0, 42], [1, 42], [2, 42]])

def test_window():
    ds = yk.stream([[c] for c in range(7)], field="foo").sliding_window(
        2, field="foo", overlap_over_samples=True
    )
    values = [d["foo"] for d in ds]
    np.testing.assert_equal(values, [[0, 1], [2, 3], [4, 5]])

    ds = (
        yk.stream([[c] for c in range(50)], field="foo")
        .sliding_window(20, field="foo", overlap_over_samples=True)
        .sliding_window(7, field="foo")
    )
    values = [d["foo"] for d in ds]
    np.testing.assert_equal(
        values,
        [
            [0, 1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12, 13],
            [20, 21, 22, 23, 24, 25, 26],
            [27, 28, 29, 30, 31, 32, 33],
        ],
    )
