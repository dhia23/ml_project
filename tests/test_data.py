from data.load_data import load_data
def test_load_data():
    df = load_data()
    assert not df.empty, "DataFrame is empty"
    assert 'Species' in df.columns, "'Species' column is missing in the DataFrame"
    assert len(df.columns) == 5