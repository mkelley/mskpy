# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Support for the Planetary Spectrum Generator
============================================
"""

import os
import re
from collections import UserDict, defaultdict
import requests
import numpy as np
from astropy.table import Table

__all__ = ["PSGConfig", "PSGModel"]


class PSGConfig(UserDict):
    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as config:
            contents = config.read(-1)

        config = cls()

        pattern = r"^<([^<>]+)>([^<>]*)\n"
        for match in re.finditer(pattern, contents, re.MULTILINE):
            config[match[1]] = cls._parse_value(match[2])

        return config

    @classmethod
    def _parse_value(cls, value):
        patterns = [
            ("none", "^$"),
            ("integer", r"[+-]?\d+$"),
            ("float", r"[+-]?\d+(\.\d+)?(([eE][+-]?\d+){0,1})?$"),
            ("list", r"([^,]+,)+[^,]+$"),
            ("string", r".+$"),
        ]
        pattern = "|".join(["(?P<%s>%s)" % pair for pair in patterns])
        match = re.match(pattern, value.rstrip())
        kind = match.lastgroup
        v = match.group()
        if kind == "none":
            return None
        elif kind == "integer":
            return int(v)
        elif kind == "float":
            return float(v)
        elif kind == "list":
            return cls._parse_list(v)
        elif kind == "string":
            return v
        else:
            raise ValueError("Unparsable value: {}".format(v))

    @classmethod
    def _parse_list(cls, values):
        value_list = []
        for value in values.split(","):
            value_list.append(cls._parse_value(value))
        return value_list

    def __str__(self):
        s = ""
        for key, value in self.items():
            s += f"<{key}>{self._format_value(value)}\n"
        return s

    def write(self, filename, overwrite=False):
        """Write to file.


        Parameters
        ----------
        filename : string
            The name of the file.

        overwrite : bool, optional
            If the file exists, overwrite it.

        """

        if os.path.exists(filename) and not overwrite:
            raise RuntimeError(
                "File exists, overwrite it with overwrite=True.")

        with open(filename, "w") as outf:
            for key, value in self.items():
                outf.write(f"<{key}>{self._format_value(value)}\n")

    @classmethod
    def _format_value(cls, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            formatted_value = ",".join([cls._format_value(v) for v in value])
        elif value is None:
            formatted_value = ""
        else:
            formatted_value = str(value)
        return formatted_value

    def run(self, fn):
        """Run the PSG with this configuration.


        Parameters
        ----------

        fn : string
            Save results to this file name.  An existing file will be
            overwritten.

        """

        data = {"file": str(self)}
        response = requests.post(
            "https://psg.gsfc.nasa.gov/api.php", data=data)
        response.raise_for_status()
        with open(fn, "w") as outf:
            outf.write(response.text)


class PSGModel:
    """PSG model spectra."""

    def __init__(self, fn):
        with open(fn, "r") as model:
            reading_data = False
            rows = []
            for line in model:
                line = line.strip()
                if len(line) == 0:
                    continue
                elif line.startswith("# Molecules considered:"):
                    molecules = [x.strip()
                                 for x in line.split(":")[1].split(",")]
                    continue
                elif "Molecular sources" in line:
                    sources = [x.strip()
                               for x in line.split(":")[1].split(",")]
                    if len(set(sources)) != len(sources):
                        raise ValueError(
                            "Molecular source names are not unique:" + ",".join(sources))

                    self.molecules = defaultdict(list)
                    for m, source in zip(molecules, sources):
                        self.molecules[m].append(source)

                    continue
                elif line.startswith("# Wave/freq"):
                    reading_data = True
                    self.source_molecules = defaultdict(list)
                    columns = line[2:].split()
                    n = len(sources)
                    columns = ["wave"] + columns[1:-n] + sources
                    continue

                if reading_data:
                    data = [float(x) for x in line.split()]
                    row = {k: v for k, v in zip(columns, data)}
                    rows.append(row)

        self.data = Table(rows)

    def __getitem__(self, k):
        if k in self.molecules:
            spec = 0
            i = 0
            for source in self.molecules[k]:
                spec = spec + self.data[source].data

            return spec
        else:
            return self.data[k]
