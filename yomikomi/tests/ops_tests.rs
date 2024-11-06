use yk::{Result, Stream};
use yomikomi as yk;

#[test]
fn iter() -> Result<()> {
    let it = [vec![42u8, 13u8, 37u8], vec![3u8, 1, 4, 1, 5, 9, 2]];
    let stream = yk::stream::from_iter_a(it.into_iter(), "foo");
    let values = stream.collect()?;
    assert_eq!(values.len(), 2);
    let v0 = values[0].get("foo").unwrap().to_vec1::<u8>()?;
    assert_eq!(v0, [42u8, 13, 37]);
    Ok(())
}

#[test]
fn jsonl() -> Result<()> {
    let jsonl = yk::jsonl::FileReader::new("tests/samples.jsonl", 0, "text".to_string(), vec![])?;
    let values = jsonl.collect()?;
    assert_eq!(values.len(), 2);
    let v0 = values[0].get("text").unwrap().to_vec1::<u8>()?;
    let v0 = String::from_utf8(v0).unwrap();
    assert_eq!(v0, "this is some sample");
    assert_eq!(values[0].get("bytes_read").unwrap().to_vec0::<i64>()?, 45);
    assert_eq!(values[0].get("line_index").unwrap().to_vec0::<i64>()?, 0);
    let v1 = values[1].get("text").unwrap().to_vec1::<u8>()?;
    let v1 = String::from_utf8(v1).unwrap();
    assert_eq!(v1, "line1\nline2");
    assert_eq!(values[1].get("bytes_read").unwrap().to_vec0::<i64>()?, 85);
    assert_eq!(values[1].get("line_index").unwrap().to_vec0::<i64>()?, 1);

    let jsonl = yk::jsonl::FileReader::new("tests/samples.jsonl", 45, "text".to_string(), vec![])?;
    let values = jsonl.collect()?;
    assert_eq!(values.len(), 1);
    let v0 = values[0].get("text").unwrap().to_vec1::<u8>()?;
    let v0 = String::from_utf8(v0).unwrap();
    assert_eq!(v0, "line1\nline2");
    assert_eq!(values[0].get("bytes_read").unwrap().to_vec0::<i64>()?, 40);
    assert_eq!(values[0].get("line_index").unwrap().to_vec0::<i64>()?, 0);
    Ok(())
}

#[test]
fn sliding_window() -> Result<()> {
    // Without overlap
    let jsonl = yk::jsonl::FileReader::new("tests/samples.jsonl", 0, "text".to_string(), vec![])?;
    let sw = yk::sliding_window::SlidingWindow::new(jsonl, 4, 4, "text".to_string(), false)?;
    let values = sw.collect()?;
    assert_eq!(values.len(), 6);
    let values = values
        .iter()
        .map(|v| {
            let v = v.get("text").unwrap().to_vec1::<u8>()?;
            let v = String::from_utf8(v).unwrap();
            Ok(v)
        })
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(values, ["this", " is ", "some", " sam", "line", "1\nli"]);

    // With overlap.
    let jsonl = yk::jsonl::FileReader::new("tests/samples.jsonl", 0, "text".to_string(), vec![])?;
    let sw = yk::sliding_window::SlidingWindow::new(jsonl, 4, 4, "text".to_string(), true)?;
    let values = sw.collect()?;
    let values = values
        .iter()
        .map(|v| {
            let v = v.get("text").unwrap().to_vec1::<u8>()?;
            let v = String::from_utf8(v).unwrap();
            Ok(v)
        })
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(values, ["this", " is ", "some", " sam", "plel", "ine1", "\nlin"]);
    Ok(())
}

#[test]
fn sliding_window2() -> Result<()> {
    let it = [
        vec![42u8, 13u8, 37u8],
        vec![3u8, 1, 4, 1, 5, 9, 2],
        vec![0u8],
        vec![42u8],
        vec![2u8, 7, 1, 8, 2, 8, 1, 8, 2, 8],
        vec![0u8],
        vec![42u8, 42u8],
        vec![2u8, 7, 1, 8],
        vec![2u8, 8, 1, 8, 2, 8],
    ];
    let stream = yk::stream::from_iter_a(it.clone().into_iter(), "foo");
    let stream = yk::sliding_window::SlidingWindow::new(stream, 3, 3, "foo".to_string(), false)?;
    let values = stream
        .collect()?
        .iter()
        .map(|v| v.get("foo").unwrap().to_vec1::<u8>())
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(
        values,
        [
            [42, 13, 37],
            [3, 1, 4],
            [1, 5, 9],
            [2, 7, 1],
            [8, 2, 8],
            [1, 8, 2],
            [2, 7, 1],
            [2, 8, 1],
            [8, 2, 8]
        ]
    );
    let stream = yk::stream::from_iter_a(it.into_iter(), "foo");
    let stream = yk::sliding_window::SlidingWindow::new(stream, 3, 3, "foo".to_string(), true)?;
    let values = stream
        .collect()?
        .iter()
        .map(|v| v.get("foo").unwrap().to_vec1::<u8>())
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(
        values,
        [
            [42, 13, 37],
            [3, 1, 4],
            [1, 5, 9],
            [2, 0, 42],
            [2, 7, 1],
            [8, 2, 8],
            [1, 8, 2],
            [8, 0, 42],
            [42, 2, 7],
            [1, 8, 2],
            [8, 1, 8]
        ]
    );
    Ok(())
}
