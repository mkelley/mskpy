# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Support for the Planetary Spectrum Generator
============================================
"""

import os
import re
from collections import UserDict
import numpy as np

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
            raise RuntimeError("File exists, overwrite it with overwrite=True.")
        
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