""" SAT dataset"""

import os
from collections.abc import MutableMapping
from typing import Generator

import pandas as pd

from datatypes import SATSample
#from  torch.utils.data import Dataset


#class SATDataset(Dataset):
class SATDataset:
    """Dataset of SAT Problems"""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i)-> SATSample:
        """get ith sample with self[i]"""
        return SATSample.from_fields(**self.df.iloc[i].to_dict())

    @classmethod
    def load(cls, load_from: str):
        """Load the dataset from a csv file"""
        return cls(
            pd.read_csv(
                load_from, converters={"formula": lambda x: str(x).replace("\\n", "\n")}
            )
        )

    def sample(self, seed: int | None = None, **kwargs) -> None:
        """Return a random sample of items from an axis of object.
        Wrapper for pandas.DataFrame.sample
        """
        self.df = self.df.sample(random_state=seed, **kwargs)

    def shuffle(self, seed: int | None = None, **kwargs) -> None:
        """Shuffle the dataset"""
        self.df = self.df.sample(frac=1, random_state=seed, **kwargs).reset_index(
            drop=True
        )

    def generator(self) -> Generator[SATSample, None, None]:
        """Create a generator for all samples in the dataset."""
        for _, row in self.df.iterrows():
            row = row.dropna()
            yield SATSample.from_fields(**row.to_dict())


class SplitDataset(MutableMapping):
    """Combines multiple splits"""

    def __init__(self, *args, **kwargs) -> None:
        self.store = {}
        self.update(dict(*args, **kwargs))

    @property
    def splits(self) -> dict[str, SATDataset]:
        """Splits like train, val, test"""
        return self.store

    @property
    def split_names(self) -> list[str]:
        """Thename of the splits"""
        return list(self)

    def __getitem__(self, key: str) -> SATDataset:
        """get split with self[split_name]"""
        return self.store[key]

    def __setitem__(self, key: str, value: SATDataset) -> None:
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __len__(self) -> int:
        return len(self.store)

    def __iter__(self):
        return iter(self.store)

    def generator(
        self, splits: list[str] | None = None, **kwargs
    ) -> Generator[SATSample, None, None]:
        """Yields dataset samples from all splits"""
        split_names = splits if splits else self.split_names
        for name in split_names:
            split = self[name]
            for sample in split.generator(**kwargs):
                yield sample

    @classmethod
    def load(cls, path: str):
        """Loads the dataset from a path"""
        path = os.path.expanduser(path)
        return cls(
            **{
                p.name: SATDataset.load(os.path.join(p.path, p.name + ".csv"))
                for p in os.scandir(path)
                if p.is_dir()
            },
            **{
                p.name[:-4]: SATDataset.load(p.path)
                for p in os.scandir(path)
                if p.name.endswith(".csv")
            },
        )
