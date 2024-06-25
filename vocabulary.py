import json

from dataset import SplitDataset

class Vocabulary:
    """A class for creating a vocabulary"""

    padding = 0
    """padding will always be the first entry"""

    def __init__(self, dataset_location= None, vocab_location= None, split= "train", sampling=0.1) -> None:
        """init with a dataset or from vocab file"""
        assert dataset_location is not None or vocab_location is not None

        if vocab_location is not None:
            with open(vocab_location, "r") as f:
                self.mapping = json.load(f)
        else:
            self.mapping = self.collect_vocab(split, dataset_location, sampling)

    def __len__(self) -> int:
        """size of the vocabulary"""
        return len(self.mapping) + 1

    def collect_vocab(self, split, dataset_location, sampling):
        """collect vocabulary from dataset"""
        data: SplitDataset = SplitDataset.load(dataset_location)
        tokens=set()

        data[split].sample(frac=sampling)
        for sample in data[split]:
            tokens.update(sample.input.to_tokens())
            tokens.update(sample.target.to_tokens())

        sorted_tokens  = sorted((t for t in tokens if not t.isdigit() ), reverse=True) + [str(t) for t in sorted(int(t) for t in tokens if t.isdigit())]
        return {item:idx + 1 for idx,item in enumerate(sorted_tokens)}


    def to_vocab(self, tokens: list[str]) -> list[int]:
        """convert a list of tokens to a list of ints"""
        def convert(token: str) -> int:
            if token in self.mapping:
                return self.mapping[token]
            else:
                raise ValueError("Unable to read " + token)

        return [convert(token) for token in tokens]

    def from_vocab(self, vocab: list[int]) -> list[str]:
        """convert a list of ints to a list of tokens"""
        rev = {v:k for k,v in self.mapping.items()}

        def convert(voc: int) -> str:
            if voc in rev:
                return rev[voc]
            else:
                raise ValueError("Unable to read " + str(voc))

        return [convert(voc) for voc in vocab]

    def save(self, path: str) -> None:
        """save vocab in file for reproducibility"""
        with open(path, 'w') as f:
            json.dump(self.mapping, f)
