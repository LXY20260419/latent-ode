#!/usr/bin/env python3
"""Pure-stdlib tests for the frozen data-contract checker."""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_experiment as runner  # noqa: E402


def write_npy_header(
    path: Path,
    *,
    shape=(38439, 2991),
    descr="<f4",
    fortran_order=False,
) -> None:
    header = repr(
        {
            "descr": descr,
            "fortran_order": fortran_order,
            "shape": shape,
        }
    )
    prefix_length = 6 + 2 + 2
    padding = (
        16 - ((prefix_length + len(header) + 1) % 16)
    ) % 16
    encoded = (header + (" " * padding) + "\n").encode("latin1")
    path.write_bytes(
        b"\x93NUMPY"
        + bytes([1, 0])
        + struct.pack("<H", len(encoded))
        + encoded
    )


class NpyHeaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(
            prefix="npy_header_"
        )
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "matrix.npy"

    def test_reads_v1_header_without_numpy(self) -> None:
        write_npy_header(self.path)
        header = runner.read_npy_header(self.path)
        self.assertEqual(header["shape"], (38439, 2991))
        self.assertEqual(header["descr"], "<f4")
        self.assertFalse(header["fortran_order"])

    def test_rejects_invalid_magic(self) -> None:
        self.path.write_bytes(b"not-a-npy")
        with self.assertRaisesRegex(ValueError, "invalid NumPy magic"):
            runner.read_npy_header(self.path)


class DataDirectoryTest(unittest.TestCase):
    def test_default_data_directory_is_preserved(self) -> None:
        self.assertEqual(runner.resolve_data_dir({}), runner.DEFAULT_DATA_DIR)

    def test_relative_override_is_resolved_from_repository_root(self) -> None:
        self.assertEqual(
            runner.resolve_data_dir(
                {"LEWORLD_DATA_DIR": "data/integrated_processed/leworld_input"}
            ),
            runner.ROOT / "data/integrated_processed/leworld_input",
        )


if __name__ == "__main__":
    unittest.main()
