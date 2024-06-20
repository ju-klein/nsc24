""" SAT datatypes"""

from abc import ABCMeta, abstractmethod
from typing import Iterator, Self


class ParsingException(Exception):
    pass


class CSVDType(metaclass=ABCMeta):
    """Abstract Class for all DTypes that can be created from a CSV row"""

    @classmethod
    @abstractmethod
    def from_fields(cls, **kwargs) -> Self:
        """Create this DType from a dict based on the CSV file"""

    @abstractmethod
    def to_fields(self) -> dict[str, str]:
        """Create a dict representing this DType for csv writing"""


class Literals(metaclass=ABCMeta):
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
    def from_strs(cls, lit_strs: list[str]) -> Self:
        """create from a list of string literals

        Args:
            lit_strs (list[str]): the list of string literals

        Raises:
            ParsingException: if not all literals are ints
        """
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
        if isinstance(value, self.__class__):
            return set(self.lits) == set(value.lits)
        return False

    def __hash__(self) -> int:
        return hash(frozenset(self))

    def sort(self) -> None:
        """sort the literals according the abstract value"""
        self.lits.sort(key=abs)

    def polarity(self, literal: int) -> bool:
        """Get the polarity of a literal

        Raises:
            ValueError: if the literal does not exist in the set
        """
        if literal in [abs(lit) for lit in self.lits if lit < 0]:
            return False
        if literal in [abs(lit) for lit in self.lits if lit > 0]:
            return True
        raise ValueError("Literal not defined")


class Clause(Literals):
    """A clause (set of literals)"""

    def to_str(self) -> str:
        """return the clause in the DIMACS CNF Format"""
        return f"{" ".join([str(lit) for lit in self.lits])}{' ' if self.lits else ''}0"

    def __str__(self) -> str:
        return self.to_str()

    @classmethod
    def from_str(cls, from_str: str) -> Self:
        """create a clause from a clause in the DIMACS CNF Format"""
        return cls.from_strs(from_str.rstrip("0").rstrip(" ").split(" "))


class SATAssignment(Literals, CSVDType):

    @classmethod
    def from_fields(cls, assignment: str, **_) -> Self:
        return cls.from_str(from_str=assignment)

    def to_fields(self) -> dict[str, str]:
        return {"assignment": self.to_str()}

    def to_str(self) -> str:
        return "v " + " ".join([str(v) for v in self.lits]) + " 0"

    def __str__(self) -> str:
        return self.to_str()

    @classmethod
    def from_str(cls, from_str: str) -> Self:
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


class CNFFormula(CSVDType):
    def __init__(
        self, clauses: list[Clause], comments: list[str] | None = None
    ) -> None:
        self.clauses: list[Clause] = clauses
        self._num_vars = None
        self.comments = comments if comments is not None else []

    @classmethod
    def from_fields(cls, formula: str, **_) -> Self:
        return cls.from_str(formula)

    def to_fields(self) -> dict[str, str]:
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
    def from_str(cls, from_str: str) -> Self:
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
        clauses = f"[{", ".join([c.__repr__() for c in self.clauses])}]"
        return f"{self.__class__.__name__}({clauses},{str(self.comments)})"

    def __eq__(self, value: object) -> bool:
        if isinstance(value, self.__class__):
            return set(self.clauses) == set(value.clauses)
        return False


class SATSample(CSVDType):
    """combination of a SAT Problem (i.e. CNF formula) and SAT solution (i.e. assignment)"""

    def __init__(self, formula: CNFFormula, assignment: SATAssignment) -> None:
        self.formula = formula
        self.assignment = assignment

    @property
    def input(self) -> CNFFormula:
        return self.formula

    @property
    def target(self) -> SATAssignment:
        return self.assignment

    @classmethod
    def from_fields(cls, **kwargs) -> Self:
        return cls(
            formula=CNFFormula.from_fields(**kwargs),
            assignment=SATAssignment.from_fields(**kwargs)
        )

    def to_fields(self) -> dict[str, str]:
        return {**self.formula.to_fields(), **self.assignment.to_fields()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.formula.__repr__(), self.assignment.__repr__()})"
