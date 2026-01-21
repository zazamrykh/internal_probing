# Deprecated

from pathlib import Path
import configparser

class Config:
    def __init__(self, path: str):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(self._path)

        parser = configparser.ConfigParser()
        parser.read(self._path)

        self._parser = parser

    @property
    def sections(self):
        return self._parser.sections()

    def __getattr__(self, section: str):
        if section in self._parser:
            return self._parser[section]
        raise AttributeError(section)

    def as_dict(self):
        return {
            section: dict(self._parser[section])
            for section in self._parser.sections()
        }
