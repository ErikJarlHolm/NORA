"""Tests for NORA file reader module."""

import io
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from nora.file_reader import FileContent, _read_text, _read_csv


def test_file_content_summary():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    fc = FileContent(filename="test.xlsx", text="hello world", tables=[df])
    summary = fc.summary()
    assert "test.xlsx" in summary
    assert "Tabell 1" in summary
    assert "2 rader" in summary


def test_read_text_utf8(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_text("Dette er en test med tall: 42 og 3.14", encoding="utf-8")
    fc = _read_text(f)
    assert fc.filename == "test.txt"
    assert "42" in fc.text
    assert "3.14" in fc.text


def test_read_csv_basic(tmp_path: Path):
    f = tmp_path / "data.csv"
    f.write_text("navn,beløp\nAlpha,1000\nBeta,2500\n", encoding="utf-8")
    fc = _read_csv(f)
    assert fc.filename == "data.csv"
    assert len(fc.tables) == 1
    assert fc.tables[0].shape == (2, 2)
    assert "1000" in fc.text or 1000 in fc.tables[0]["beløp"].values


def test_read_file_unsupported_falls_back_to_text(tmp_path: Path):
    from nora.file_reader import read_file
    f = tmp_path / "notes.log"
    f.write_text("2024-01-01 revenue: 99999", encoding="utf-8")
    fc = read_file(f)
    assert "99999" in fc.text


def test_read_folder_empty(tmp_path: Path):
    from nora.file_reader import read_folder
    result = read_folder(tmp_path)
    assert result == []
