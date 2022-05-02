from forest_model.train import parse_args


def test_parser():
    parser = parse_args(["--random-state", "42"])
    assert 42 in parser
    assert "1" in parser
    assert "log" in parser
