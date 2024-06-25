""" SAT datatypes"""

from abc import ABCMeta, abstractmethod
from typing import Iterator


class ParsingException(Exception):
    pass


class Tokenizable(metaclass=ABCMeta):
    """Abstract Class for all DTypes that can be tokenized"""

    @abstractmethod
    def to_tokens(self) -> list[str]:
        pass

    @classmethod
    @abstractmethod
    def from_tokens(cls, tokens: list[str]) -> 'Tokenizable':
        pass


class CSVDType(metaclass=ABCMeta):
    """Abstract Class for all DTypes that can be created from a CSV row"""

    @classmethod
    @abstractmethod
    def from_fields(cls, **kwargs) -> 'CSVDType':
        """Create this DType from a dict based on the CSV file"""

    @abstractmethod
    def to_fields(self) -> dict[str, str]:
        """Create a dict representing this DType for csv writing"""


class Literals(Tokenizable, metaclass=ABCMeta):
    """Abstract class that summarizes a collection of literals."""

    def __init__(self, lits: list[int]) -> None:
        if len(set(abs(lit) for lit in lits)) != len(lits):
            raise ParsingException("Literals not distinct or a tautology")

        if 0 in lits:
            raise ParsingException("0 cannot be a literal")

        self.lits: list[int] = lits

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.lits)})"

    @classmethod
    def from_strs(cls, lit_strs: list[str]) -> 'Literals':
        """create from a list of string literals"""
        try:
            lits = [int(lit_str) for lit_str in lit_strs]
        except ValueError as exc:
            raise ParsingException("Non-integer in literal list") from exc
        return cls(lits)

    @property
    def max_var(self) -> int:
        """The largest literal in this set"""
        return abs(max(self.lits, key=abs))

    def __len__(self) -> int:
        """the number of literals"""
        return len(self.lits)

    def __iter__(self) -> Iterator[int]:
        return iter(self.lits)

    def __eq__(self, value: object) -> bool:
        """two Literals classes are equal if they contain the same literals"""
        if isinstance(value, self.__class__):
            return set(self.lits) == set(value.lits)
        return False

    def __hash__(self) -> int:
        return hash(frozenset(self))

    @property
    def atoms(self) -> set[int]:
        """All atoms/variables"""
        return {abs(lit) for lit in self.lits}

    def sort(self) -> None:
        """sort the literals according the abstract value"""
        self.lits.sort(key=abs)

    def polarity(self, atom: int) -> bool:
        """Get the polarity of an atom

        Raises:
            ValueError: if the atom does not exist in the set
        """
        if atom in [abs(lit) for lit in self.lits if lit < 0]:
            return False
        if atom in [abs(lit) for lit in self.lits if lit > 0]:
            return True
        raise ValueError("Atom not defined")

    def to_tokens(self) -> list[str]:
        raise NotImplementedError

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> 'Literals':
        raise NotImplementedError

class Clause(Literals):
    """A clause (set of literals)"""

    def to_str(self) -> str:
        """return the clause in the DIMACS CNF Format"""
        return f'{" ".join([str(lit) for lit in self.lits])}{" " if self.lits else ""}0'

    def __str__(self) -> str:
        return self.to_str()

    @classmethod
    def from_str(cls, from_str: str) -> 'Literals':
        """create a clause from a clause in the DIMACS CNF Format"""
        return cls.from_strs(from_str.rstrip("0").rstrip(" ").split(" "))


class SATAssignment(Literals, CSVDType):

    @classmethod
    def from_fields(cls, assignment: str, **_) -> 'SATAssignment':
        """load from a csv row"""
        return cls.from_str(from_str=assignment)

    def to_fields(self) -> dict[str, str]:
        """write a csv row"""
        return {"assignment": self.to_str()}

    def to_str(self) -> str:
        return "v " + " ".join([str(v) for v in self.lits]) + " 0"

    def __str__(self) -> str:
        return self.to_str()

    @classmethod
    def from_str(cls, from_str: str) -> 'SATAssignment':
        """create an assignment from a DIMACS Assignment str"""
        assignment = []
        read_zero = False
        for l in from_str.split("\n"):
            if not l.startswith("v"):
                raise ParsingException("New line does not start with v")
            for v in l.split(" ")[1:]:
                if v == "0":
                    read_zero = True
                    break
                assignment.append(v)
            if read_zero:
                break
        if read_zero:
            return cls.from_strs(assignment)
        raise ParsingException("Assignment string does not end with zero")

    def to_tokens(self) -> list[str]:
        raise NotImplementedError

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> 'SATAssignment':
        raise NotImplementedError


class CNFFormula(Tokenizable, CSVDType):
    def __init__(
        self, clauses: list[Clause], comments: list[str] | None = None
    ) -> None:
        self.clauses: list[Clause] = clauses
        self._num_vars = None
        self.comments = comments if comments is not None else []

    @classmethod
    def from_fields(cls, formula: str, **_) -> 'CNFFormula':
        """create from csv row"""
        return cls.from_str(formula)

    def to_fields(self) -> dict[str, str]:
        """write a csv row"""
        return {"formula": self.to_str()}

    def __iter__(self) -> Iterator[Clause]:
        return iter(self.clauses)

    @property
    def nbclauses(self) -> int:
        """the number of clauses"""
        return len(self.clauses)

    @property
    def nbvars(self) -> int:
        "the number of literals"
        return (
            self._num_vars if self._num_vars else max(c.max_var for c in self.clauses)
        )

    @classmethod
    def from_str(cls, from_str: str) -> 'CNFFormula':
        """Create CNFFormula from a string in DIMACS CNF format"""
        nbvars = -1
        nbclauses = -1
        clauses: list[Clause] = []
        comments: list[str] = []
        from_str = from_str.rstrip("\n ")
        for clause_str in from_str.split("\n"):
            if clause_str.startswith("c"):
                comments.append(clause_str[2:])
            elif clause_str.startswith("p"):
                header = clause_str.split(" ")
                if nbvars != -1 or nbclauses != -1 or header[1] != "cnf":
                    raise ParsingException("could not read header")
                nbvars = int(header[2])
                nbclauses = int(header[3])
            else:
                clauses.append(Clause.from_str(clause_str))

        formula = cls(clauses)

        if nbclauses == -1 or nbvars == -1:
            raise ParsingException("No header found")

        if nbclauses != formula.nbclauses:
            raise ParsingException(
                "Specified number of clauses does not match parsed number of clauses"
            )

        if nbvars != formula.nbvars:
            raise ParsingException(
                "Specified number of variables does not match parsed number of variables"
            )

        return formula

    def to_str(self) -> str:
        """Create the DIMACS CNF format representation of this CNF"""
        comment_str = "".join([f"c {co}\n" for co in self.comments])
        header_str = f"p cnf {self.nbvars} {self.nbclauses}\n"
        clause_str = "".join([cl.to_str() + "\n" for cl in self.clauses])
        return comment_str + header_str + clause_str

    def __str__(self) -> str:
        return self.to_str()

    def __repr__(self) -> str:
        clauses = f'[{", ".join([c.__repr__() for c in self.clauses])}]'
        return f'{self.__class__.__name__}({clauses},{str(self.comments)})'

    def to_tokens(self) -> list[str]:
        raise NotImplementedError

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> 'CNFFormula':
        raise NotImplementedError

    def __eq__(self, value: object) -> bool:
        if isinstance(value, self.__class__):
            return set(self.clauses) == set(value.clauses)
        return False

    @property
    def atoms(self) -> set[int]:
        """get all atoms of this formula"""
        return {a for clause in self.clauses for a in clause.atoms}


class SATSample(CSVDType):
    """combination of a SAT Problem (i.e. CNF formula) and SAT solution (i.e. assignment, prediction)"""

    def __init__(self, formula: CNFFormula, target_assignment: None | SATAssignment = None, prediction_assignment: None | SATAssignment= None) -> None:
        """initialize with formula (mandatory) and target (for supervised samples) and prediction (if exists)"""
        self.formula = formula
        self.target_assignment = target_assignment
        self.prediction_assignment = prediction_assignment

    @property
    def input(self) -> CNFFormula:
        return self.formula

    @property
    def target(self) -> SATAssignment:
        assert self.target_assignment is not None
        return self.target_assignment

    @property
    def prediction(self) -> SATAssignment:
        assert self.prediction_assignment is not None
        return self.prediction_assignment

    @classmethod
    def from_fields(cls, **kwargs) -> 'SATSample':
        """create from a csv row"""
        return cls(
            formula=CNFFormula.from_fields(**kwargs),
            target_assignment=SATAssignment.from_fields(**kwargs)
        )

    def to_fields(self) -> dict[str, str]:
        """convert to a csv row"""
        return {**self.formula.to_fields(), **(self.target_assignment.to_fields() if self.target_assignment is not None else {})}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.formula.__repr__(), self.target_assignment.__repr__()})"


    def equal(self) -> bool:
        """true if the prediction and the target are the same assignment"""
        raise NotImplementedError

    def equal_tk(self) -> bool:
        """true if the prediction and target have the same tokenization (i.e. same order of literals)"""
        raise NotImplementedError

    def correct(self) -> bool:
        """true if the assignment satisfies the formula"""
        raise NotImplementedError
