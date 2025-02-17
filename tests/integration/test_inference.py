import pytest
from oocr.cli.inference import extract_text_ckpt, clean_whitespaces

def test_clean_whitespaces():
    text = "  hello   world  "
    assert clean_whitespaces(text) == "hello world"

def test_extract_text_invalid_mode():
    with pytest.raises(ValueError, match="Invalid mode"):
        extract_text_ckpt(None, None, None, False, mode="invalid")