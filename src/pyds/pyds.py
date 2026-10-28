"""
A framework for performing computations in the Dempster-Shafer theory.
"""

import sys
from collections.abc import Callable, Iterable, Mapping
from functools import reduce
from itertools import chain, combinations
from math import fsum, log, log2, sqrt
from operator import mul
from random import random, shuffle, uniform
from typing import Literal, TypeVar, overload

try:
    import numpy

    try:
        from scipy.optimize import linprog
        from scipy.stats import chi2
    except ModuleNotFoundError:
        print("SciPy not found: some features will not work.", file=sys.stderr)
except ModuleNotFoundError:
    print("NumPy not found: some features will not work.", file=sys.stderr)

LP_ZERO_TOLERANCE = 1e-12


class NotProbabilisticError(Exception):
    """Indicates that a mass function is missspecified."""


K = TypeVar("K")
"""Keys of single hypotheses."""


class MassFunction(dict[frozenset[K], float]):
    """
    A Dempster-Shafer mass function (basic probability assignment) based on a dictionary.

    Both normalized and unnormalized mass functions are supported.
    The underlying frame of discernment is assumed to be discrete.

    Hypotheses and their associated mass values can be added/changed/removed using the standard dictionary methods.
    Each hypothesis can be an arbitrary sequence which is automatically converted to a 'frozenset', meaning its elements must be hashable.
    """

    def __init__(
        self,
        source: Iterable[tuple[Iterable[K], float]]
        | Mapping[Iterable[K], float]
        | None = None,
    ):
        """
        Creates a new mass function.

        If 'source' is not None, it is used to initialize the mass function.
        It can either be a dictionary mapping hypotheses to non-negative mass values
        or an iterable containing tuples consisting of a hypothesis and a corresponding mass value.
        """
        if isinstance(source, Mapping):
            for h, v in source.items():
                self[frozenset(h)] += v
        elif source is not None:
            for h, v in source:
                self[frozenset(h)] += v

    @staticmethod
    def _convert(hypothesis: Iterable[K]) -> frozenset[K]:
        """Convert hypothesis to a 'frozenset' in order to make it hashable."""
        return frozenset(hypothesis)

    @staticmethod
    def _gbt_likelihoods(
        likelihoods: Mapping[K, float] | Iterable[tuple[K, float]],
    ) -> list[tuple[K, float]]:
        """Normalize input into a list of `(hypothesis, likelihood)` pairs."""
        if isinstance(likelihoods, Mapping):
            return list(likelihoods.items())
        return list(likelihoods)

    @staticmethod
    def gbt(
        likelihoods: Mapping[K, float] | Iterable[tuple[K, float]],
        normalization: bool = True,
        sample_count: int | None = None,
    ) -> "MassFunction[K]":
        """
        Constructs a mass function using the generalized Bayesian theorem.
        For more information, see Smets. 1993. Belief functions:
        The disjunctive rule of combination and the generalized Bayesian theorem. International Journal of Approximate Reasoning.

        'likelihoods' specifies the conditional plausibilities for a set of singleton hypotheses.
        It can either be a dictionary mapping singleton hypotheses to plausibilities or an iterable
        containing tuples consisting of a singleton hypothesis and a corresponding plausibility value.

        'normalization' determines whether the resulting mass function is normalized, i.e., whether m({}) == 0.

        If 'sample_count' is not None, the true mass function is approximated using the specified number of samples.
        """
        m = MassFunction[K]()
        likelihoods = MassFunction._gbt_likelihoods(likelihoods)

        # filter trivial likelihoods 0 and 1
        ones: list[K] = [h for (h, l) in likelihoods if l >= 1.0]
        likelihoods = [(h, l) for (h, l) in likelihoods if 0.0 < l < 1.0]
        if sample_count is None:  # deterministic

            def traverse(
                m: "MassFunction[K]",
                likelihoods: list[tuple[K, float]],
                ones: list[K],
                index: int,
                hyp: list[K],
                mass: float,
            ) -> None:
                if index == len(likelihoods):
                    m[hyp + ones] = mass
                else:
                    traverse(
                        m,
                        likelihoods,
                        ones,
                        index + 1,
                        hyp + [likelihoods[index][0]],
                        mass * likelihoods[index][1],
                    )
                    traverse(
                        m,
                        likelihoods,
                        ones,
                        index + 1,
                        hyp,
                        mass * (1.0 - likelihoods[index][1]),
                    )

            traverse(m, likelihoods, ones, 0, [], 1.0)
            if normalization:
                m.normalize()
        else:  # Monte-Carlo
            if normalization:
                empty_mass = reduce(mul, [1.0 - l[1] for l in likelihoods], 1.0)
            for _ in range(sample_count):
                rv = [random() for _ in range(len(likelihoods))]
                subtree_mass = 1.0
                hyp: set[K] = set(ones)
                for k in range(len(likelihoods)):
                    l = likelihoods[k][1]
                    p_t = l * subtree_mass
                    p_f = (1.0 - l) * subtree_mass
                    if (
                        normalization and not hyp
                    ):  # avoid empty hypotheses in the normalized case
                        p_f -= empty_mass
                    if p_t > rv[k] * (p_t + p_f):
                        hyp.add(likelihoods[k][0])
                    else:
                        subtree_mass *= (
                            1 - l
                        )  # only relevant for the normalized empty case
                m[hyp] += 1.0 / sample_count
        return m

    @staticmethod
    def from_bel(bel: Mapping[frozenset[K], float]) -> "MassFunction[K]":
        """
        Creates a mass function from a corresponding belief function.

        'bel' is a dictionary mapping hypotheses to belief values (like the dictionary returned by 'bel(None)').
        """
        m = MassFunction[K]()
        for h1 in bel:
            v = fsum([bel[h2] * (-1) ** (len(h1 - h2)) for h2 in powerset(h1)])
            if v > 0:
                m[h1] = v
        mass_sum = fsum(m.values())
        if mass_sum < 1.0:
            m[frozenset()] = 1.0 - mass_sum
        return m

    @staticmethod
    def from_pl(pl: Mapping[frozenset[K], float]) -> "MassFunction[K]":
        """
        Creates a mass function from a corresponding plausibility function.

        'pl' is a dictionary mapping hypotheses to plausibility values (like the dictionary returned by 'pl(None)').
        """
        frame = max(pl.keys(), key=len)
        bel_theta = pl[frame]
        bel = {
            frozenset(frame - h): bel_theta - v for (h, v) in pl.items()
        }  # follows from bel(-A) = bel(frame) - pl(A)
        return MassFunction.from_bel(bel)

    @staticmethod
    def from_q(q: Mapping[frozenset[K], float]) -> "MassFunction[K]":
        """
        Creates a mass function from a corresponding commonality function.

        'q' is a dictionary mapping hypotheses to commonality values (like the dictionary returned by 'q(None)').
        """
        m = MassFunction[K]()
        frame = max(q.keys(), key=len)
        for h1 in q:
            v = fsum(
                [q[h1 | h2] * (-1) ** (len(h2 - h1)) for h2 in powerset(frame - h1)]
            )
            if v > 0:
                m[h1] = v
        mass_sum = fsum(m.values())
        if mass_sum < 1.0:
            m[frozenset()] = 1.0 - mass_sum
        return m

    def __missing__(self, key):
        """Return 0 mass for hypotheses that are not contained."""
        return 0.0

    def __copy__(self):
        c = MassFunction()
        for k, v in self.items():
            c[k] = v
        return c

    def copy(self):
        """Creates a shallow copy of the mass function."""
        return self.__copy__()

    def __contains__(self, hypothesis):
        return dict.__contains__(self, MassFunction._convert(hypothesis))

    def __getitem__(self, hypothesis):
        return dict.__getitem__(self, MassFunction._convert(hypothesis))

    def __setitem__(self, hypothesis: Iterable[K], value: float):
        """
        Adds or updates the mass value of a hypothesis.

        'hypothesis' is automatically converted to a 'frozenset' meaning its elements must be hashable.
        In case of a negative mass value, a ValueError is raised.
        """
        if value < 0.0:
            raise ValueError(f"mass value is negative: {value:f}")
        dict.__setitem__(self, MassFunction._convert(hypothesis), value)

    def __delitem__(self, hypothesis):
        return dict.__delitem__(self, MassFunction._convert(hypothesis))

    def frame(self) -> frozenset[K]:
        """
        Returns the frame of discernment of the mass function as a 'frozenset'.

        The frame of discernment is the union of all contained hypotheses.
        In case the mass function does not contain any hypotheses, an empty set is returned.
        """
        if not self:
            return frozenset()
        else:
            return frozenset.union(*self.keys())

    def singletons(self) -> set[frozenset[K]]:
        """
        Returns the set of all singleton hypotheses.

        Like 'frame()', except that each singleton is wrapped in a frozenset
        and can thus be directly passed to methods like 'bel()'.
        """
        return {frozenset((s,)) for s in self.frame()}

    def focal(self) -> set[frozenset[K]]:
        """
        Returns the set of all focal hypotheses.

        A focal hypothesis has a mass value greater than 0.
        """
        return {h for (h, v) in self.items() if v > 0}

    def core(self, *mass_functions: "MassFunction[K]") -> frozenset[K]:
        """
        Returns the core of one or more mass functions as a 'frozenset'.

        The core of a single mass function is the union of all its focal hypotheses.
        In case a mass function does not contain any focal hypotheses, its core is an empty set.
        If multiple mass functions are given, their combined core (intersection of all single cores) is returned.
        """
        if mass_functions:
            return frozenset.intersection(
                self.core(), *[m.core() for m in mass_functions]
            )
        else:
            focal = self.focal()
            if not focal:
                return frozenset()
            else:
                return frozenset.union(*focal)

    def all(self):
        """Returns an iterator over all subsets of the frame of discernment, including the empty set."""
        return powerset(self.frame())

    @overload
    def bel(self, hypothesis: None = None) -> Mapping[frozenset[K], float]: ...

    @overload
    def bel(self, hypothesis: Iterable[K]) -> float: ...

    def bel(
        self, hypothesis: Iterable[K] | None = None
    ) -> Mapping[frozenset[K], float] | float:
        """
        Computes either the belief of 'hypothesis' or the entire belief function (hypothesis=None).

        If 'hypothesis' is None (default), a dictionary mapping hypotheses to their respective belief values is returned.
        Otherwise, the belief of 'hypothesis' is returned.
        In this case, 'hypothesis' is automatically converted to a 'frozenset' meaning its elements must be hashable.
        """
        if hypothesis is None:
            return {h: self.bel(h) for h in powerset(self.core())}
        else:
            hypothesis = MassFunction._convert(hypothesis)
            if not hypothesis:
                return 0.0
            else:
                return fsum(
                    [v for (h, v) in self.items() if h and hypothesis.issuperset(h)]
                )

    @overload
    def pl(self, hypothesis: None = None) -> Mapping[frozenset[K], float]: ...

    @overload
    def pl(self, hypothesis: Iterable[K]) -> float: ...

    def pl(
        self, hypothesis: Iterable[K] | None = None
    ) -> Mapping[frozenset[K], float] | float:
        """
        Computes either the plausibility of 'hypothesis' or the entire plausibility function (hypothesis=None).

        If 'hypothesis' is None (default), a dictionary mapping hypotheses to their respective plausibility values is returned.
        Otherwise, the plausibility of 'hypothesis' is returned.
        In this case, 'hypothesis' is automatically converted to a 'frozenset' meaning its elements must be hashable.
        """
        if hypothesis is None:
            return {h: self.pl(h) for h in powerset(self.core())}
        else:
            hypothesis = MassFunction._convert(hypothesis)
            if not hypothesis:
                return 0.0
            else:
                return fsum([v for (h, v) in self.items() if hypothesis & h])

    @overload
    def q(self, hypothesis: None) -> dict[frozenset[K], float]: ...

    @overload
    def q(self, hypothesis: Iterable[K]) -> float: ...

    def q(
        self, hypothesis: Iterable[K] | None = None
    ) -> float | dict[frozenset[K], float]:
        """
        Computes either the commonality of 'hypothesis' or the entire commonality function (hypothesis=None).

        If 'hypothesis' is None (default), a dictionary mapping hypotheses to their respective commonality values is returned.
        Otherwise, the commonality of 'hypothesis' is returned.
        In this case, 'hypothesis' is automatically converted to a 'frozenset' meaning its elements must be hashable.
        """
        if hypothesis is None:
            return {h: self.q(h) for h in powerset(self.core())}
        else:
            if not hypothesis:
                return 1.0
            else:
                return fsum([v for (h, v) in self.items() if h.issuperset(hypothesis)])

    def __and__(self, mass_function):
        """Shorthand for 'combine_conjunctive(mass_function)'."""
        return self.combine_conjunctive(mass_function)

    def __or__(self, mass_function):
        """Shorthand for 'combine_disjunctive(mass_function)'."""
        return self.combine_disjunctive(mass_function)

    def __str__(self):
        hyp = sorted([(v, h) for (h, v) in self.items()], reverse=True)
        return "{" + "; ".join([str(set(h)) + ":" + str(v) for (v, h) in hyp]) + "}"

    def __mul__(self, scalar):
        if not isinstance(scalar, float):
            raise TypeError("Can only multiply by a float value.")
        m = MassFunction()
        for h, v in self.items():
            m[h] = v * scalar
        return m

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __add__(self, m):
        if not isinstance(m, MassFunction):
            raise TypeError("Can only add two mass functions.")
        result = self.copy()
        for h, v in m.items():
            result[h] += v
        return result

    def weight_function(self):
        """
        Computes the weight function corresponding to this mass function.
        """
        weights = {}
        theta = self.frame()
        for h in powerset(theta):
            if len(h) < len(theta):  # weight is undefined for theta
                sets = [h | c for c in powerset(theta - h)]
                # Use self.q(h2) directly to avoid KeyError when core doesn't span full frame
                q_even = reduce(
                    mul, [self.q(h2) for h2 in sets if len(h2) % 2 == 0], 1.0
                )
                q_odd = reduce(
                    mul, [self.q(h2) for h2 in sets if len(h2) % 2 == 1], 1.0
                )
                # Use inf when denominator is zero (fully conflicting evidence)
                if len(h) % 2 == 0:
                    weights[h] = q_odd / q_even if q_even != 0.0 else float("inf")
                else:
                    weights[h] = q_even / q_odd if q_odd != 0.0 else float("inf")
        return weights

    def combine_conjunctive(
        self,
        mass_function,
        normalization=True,
        sample_count: int | None = None,
        importance_sampling=False,
    ):
        """
        Conjunctively combines the mass function with another mass function and returns the combination as a new mass function.

        The other mass function is assumed to be defined over the same frame of discernment.
        If 'mass_function' is not of type MassFunction, it is assumed to be an iterable containing multiple mass functions that are iteratively combined.

        If the mass functions are flatly contracting or if one of the mass functions is empty, an empty mass function is returned.

        'normalization' determines whether the resulting mass function is normalized (default is True).

        If 'sample_count' is not None, the true combination is approximated using the specified number of samples.
        In this case, 'importance_sampling' determines the method of approximation (only if normalization=True, otherwise 'importance_sampling' is ignored).
        The default method (importance_sampling=False) independently generates samples from both mass functions and computes their intersections.
        If importance_sampling=True, importance sampling is used to avoid empty intersections, which leads to a lower approximation error but is also slower.
        This method should be used if there is significant evidential conflict between the mass functions.
        """
        return self._combine(
            mass_function,
            rule=lambda s1, s2: s1 & s2,
            normalization=normalization,
            sample_count=sample_count,
            importance_sampling=importance_sampling,
        )

    def combine_disjunctive(self, mass_function, sample_count: int | None = None):
        """
        Disjunctively combines the mass function with another mass function and returns the combination as a new mass function.

        The other mass function is assumed to be defined over the same frame of discernment.
        If 'mass_function' is not of type MassFunction, it is assumed to be an iterable containing multiple mass functions that are iteratively combined.

        If 'sample_count' is not None, the true combination is approximated using the specified number of samples.
        """
        return self._combine(
            mass_function,
            rule=lambda s1, s2: s1 | s2,
            normalization=False,
            sample_count=sample_count,
            importance_sampling=False,
        )

    def combine_cautious(self, mass_function):
        """
        Combines the mass function with another mass function using the cautious rule and returns the combination as a new mass function.

        For more details, see:
        T. Denoeux (2008), "Conjunctive and disjunctive combination of belief functions induced by nondistinct bodies of evidence",
        Artificial Intelligence 172, 234-264.
        """
        w1 = self.weight_function()
        w2 = mass_function.weight_function()
        w_min = {h: min(w1[h], w2[h]) for h in w1}
        theta = self.frame()
        m = MassFunction({theta: 1.0})
        for h, w in w_min.items():
            # Clamp to 0.0 to handle floating point imprecision where w is marginally > 1.0
            m_simple = MassFunction({theta: w, h: max(0.0, 1.0 - w)})
            m = m.combine_conjunctive(m_simple, normalization=False)
        return m

    def _combine(
        self, mass_function, rule, normalization, sample_count, importance_sampling
    ):
        """Helper method for combining two or more mass functions."""
        combined = self
        if isinstance(mass_function, MassFunction):
            mass_function = [mass_function]  # wrap single mass function
        for m in mass_function:
            if not isinstance(m, MassFunction):
                raise TypeError(
                    f"expected type MassFunction but got {type(m)}; make sure to use keyword arguments for anything other than mass functions"
                )
            if sample_count is None:
                combined = combined._combine_deterministic(m, rule)
            else:
                if importance_sampling and normalization:
                    combined = combined._combine_importance_sampling(m, sample_count)
                else:
                    combined = combined._combine_direct_sampling(m, rule, sample_count)
        if normalization:
            return combined.normalize()
        else:
            return combined

    def _combine_deterministic(self, mass_function, rule):
        """Helper method for deterministically combining two mass functions."""
        combined = MassFunction()
        for h1, v1 in self.items():
            for h2, v2 in mass_function.items():
                combined[rule(h1, h2)] += v1 * v2
        return combined

    def _combine_direct_sampling(self, mass_function, rule, sample_count):
        """Helper method for approximatively combining two mass functions using direct sampling."""
        combined = MassFunction()
        samples1 = self.sample(sample_count)
        samples2 = mass_function.sample(sample_count)
        for i in range(sample_count):
            combined[rule(samples1[i], samples2[i])] += 1.0 / sample_count
        return combined

    def _combine_importance_sampling(self, mass_function, sample_count):
        """Helper method for approximatively combining two mass functions using importance sampling."""
        combined = MassFunction()
        for s1, n in self.sample(sample_count, as_dict=True).items():
            weight = mass_function.pl(s1)
            for s2 in mass_function.condition(s1).sample(n):
                combined[s2] += weight
        return combined

    def combine_gbt(
        self,
        likelihoods: Mapping[K, float] | Iterable[tuple[K, float]],
        normalization: bool = True,
        sample_count: int | None = None,
        importance_sampling=True,
    ) -> "MassFunction[K]":
        """
        Conjunctively combines the mass function with a mass function obtained from a sequence of
        likelihoods via the generalized Bayesian theorem and returns the combination as a new mass function.

        Equivalent to 'combine_conjunctive(MassFunction.gbt(likelihoods))'.
        By ignoring incompatible likelihoods, it is generally faster than the former
        method and yields a better Monte-Carlo approximation in case of normalization.

        'likelihoods' specifies the conditional plausibilities for a set of singleton hypotheses.
        It can either be a dictionary mapping singleton hypotheses to plausibilities or an iterable
        containing tuples consisting of a singleton hypothesis and a corresponding plausibility value.

        All arguments except for 'likelihoods' must be specified as keyword arguments.
        'normalization' determines whether the resulting mass function is normalized, i.e., whether m({}) == 0.
        If 'sample_count' is not None, the true mass function is approximated using the specified number of samples.
        See 'combine_conjunctive' for details on the effect of setting 'importance_sampling'.
        """
        core = self.core()  # restrict to generally compatible likelihoods
        likelihoods = MassFunction._gbt_likelihoods(likelihoods)
        likelihoods = [l for l in likelihoods if l[1] > 0 and l[0] in core]
        if sample_count is None:  # deterministic
            return self.combine_conjunctive(
                MassFunction.gbt(likelihoods), normalization=normalization
            )
        else:  # Monte-Carlo
            if (
                not normalization
            ):  # only use importance sampling in case of normalization
                importance_sampling = False
            combined = MassFunction[K]()
            for s, n in self.sample(sample_count, as_dict=True).items():
                if importance_sampling:
                    compatible_likelihoods = [l for l in likelihoods if l[0] in s]
                    weight = 1.0 - reduce(
                        mul, [1.0 - l[1] for l in compatible_likelihoods], 1.0
                    )
                else:
                    compatible_likelihoods = likelihoods
                if not compatible_likelihoods:
                    continue
                if normalization:
                    empty_mass = reduce(
                        mul, [1.0 - l[1] for l in compatible_likelihoods], 1.0
                    )
                for _ in range(n):
                    rv = [random() for _ in range(len(compatible_likelihoods))]
                    subtree_mass = 1.0
                    hyp: set[K] = set()
                    for k in range(len(compatible_likelihoods)):
                        h, l = compatible_likelihoods[k]
                        norm = (
                            1.0
                            if hyp or not normalization
                            else 1.0 - empty_mass / subtree_mass
                        )
                        if l / norm > rv[k]:
                            hyp.add(h)
                        else:
                            subtree_mass *= 1.0 - l  # only relevant for negative case
                    if importance_sampling:
                        combined[hyp] += weight
                    else:
                        combined[hyp & s] += 1.0
            if normalization:
                return combined.normalize()
            else:
                return combined

    def condition(self, hypothesis, normalization=True):
        """
        Conditions the mass function with 'hypothesis'.

        'normalization' determines whether the resulting conjunctive combination is normalized (must be specified as a keyword argument).

        Shorthand for self.combine_conjunctive(MassFunction({hypothesis:1.0}), normalization).
        """
        m = MassFunction({MassFunction._convert(hypothesis): 1.0})
        return self.combine_conjunctive(m, normalization=normalization)

    def conflict(
        self,
        mass_function: "MassFunction[K] | Iterable[MassFunction[K]]",
        sample_count: int | None = None,
    ) -> float:
        """
        Calculates the weight of conflict between two or more mass functions.

        If 'mass_function' is not of type MassFunction, it is assumed to be an iterable containing multiple mass functions.

        The weight of conflict is computed as the (natural) logarithm of the normalization constant in Dempster's rule of combination.
        Returns infinity in case the mass functions are flatly contradicting.
        """
        # compute full conjunctive combination (could be more efficient)
        m = self.combine_conjunctive(
            mass_function, normalization=False, sample_count=sample_count
        )
        empty = m[frozenset()]
        m_sum = fsum(m.values())
        diff = m_sum - empty
        if diff == 0.0:
            return float("inf")
        else:
            return -log(diff)

    def normalize(self):
        """
        Normalizes the mass function in-place.

        Sets the mass value of the empty set to 0 and scales all other values such that their sum equals 1.
        For convenience, the method returns 'self'.
        """
        if frozenset() in self:
            del self[frozenset()]
        mass_sum = fsum(self.values())
        if mass_sum != 1.0:
            for h, v in self.items():
                self[h] = v / mass_sum
        return self

    def prune(self):
        """
        Removes all non-focal (0 mass) hypotheses in-place.

        For convenience, the method returns 'self'.
        """
        remove = [h for (h, v) in self.items() if v == 0.0]
        for h in remove:
            del self[h]
        return self

    def markov_deterministic(
        self,
        transition_model: "Callable[[K], MassFunction[K]]",
    ):
        updated = MassFunction[K]()
        for k, v in self.items():
            predicted: MassFunction[K] | None = None
            for e in k:
                if predicted is None:
                    predicted = transition_model(e)
                else:
                    predicted = predicted.combine_disjunctive(transition_model(e))
            kp: frozenset[K]
            vp: float
            for kp, vp in (predicted or {}).items():
                updated[kp] += v * vp
        return updated

    def markov_monto_carlo(
        self,
        transition_model: "Callable[[K, int], MassFunction[K]]",
        sample_count: int,
    ):
        updated = MassFunction[K]()
        s: frozenset[K]
        for s, n in self.sample(sample_count, as_dict=True).items():
            unions: list[list[K]] = [[] for _ in range(n)]
            for e in s:
                ts = transition_model(e, n)
                for i, t in enumerate(ts):
                    unions[i].extend(t)
            for u in unions:
                updated[u] += 1.0 / sample_count
        return updated

    def markov(
        self,
        transition_model: "Callable[[K], MassFunction[K]] | Callable[[K, int], MassFunction[K]]",
        sample_count: int | None = None,
    ):
        """
        Computes the mass function induced by a prior belief (self) and a transition model.

        The transition model expresses a joint belief over the frame of this mass function and a new frame.
        The belief over the frame of this mass function is implicitly assumed to be vacuous.
        The transition model is a function returning the conditional belief over the new frame (as a mass function
        if sample_count=None) while taking a singleton hypothesis of the current frame as input.
        The disjunctive rule of combination is then used to construct the mass function over the new frame.

        If 'sample_count' is not None, the true mass function is approximated using the specified number of samples.
        In this case, 'transition_model' is expected to take a second argument stating how many samples from the corresponding conditional mass function should be returned.
        The return value in this case is expected to be an iterable over sampled hypotheses from the new frame.

        This method can be used to implement the prediction step for estimation in a hidden Markov process (hence the name).
        Under this interpretation, the transition model expresses the mass distribution over successor states given the current state.
        """
        if sample_count is None:  # deterministic
            return self.markov_deterministic(transition_model)  # type: ignore
        else:  # Monte-Carlo
            return self.markov_monto_carlo(transition_model, sample_count)  # type: ignore

    def map(self, function):
        """
        Maps each hypothesis to a new hypothesis using 'function' and returns the new mass function.

        'function' is a function taking a hypothesis as its only input and returning a new hypothesis
        (i.e., a sequence that can be converted to a 'frozenset').

        Here are some example use cases:

        1. Vacuous extension to a multi-dimensional frame of discernment (m is defined over
        the frame A while the new mass function is defined over the Cartesian product AxB):

            B = {'x', 'y', 'z'}
            m.map(lambda h: itertools.product(h, B))

        2. Projection to a lower dimensional frame (m is defined over AxBxC such that each hypothesis is
        a set of tuples where each tuple consists of 3 elements; the new mass function is defined over BxC):

            m.map(lambda h: (t[1:] for t in h))
        """
        m = MassFunction()
        for h, v in self.items():
            m[self._convert(function(h))] += v
        return m

    def pignistic(self):
        """Computes the pignistic transformation and returns it as a new mass function consisting only of singletons."""
        p = MassFunction()
        for h, v in self.items():
            if v > 0.0:
                size = float(len(h))
                for s in h:
                    p[(s,)] += v / size
        return p.normalize()

    def local_conflict(self) -> float:
        """
        Computes the local conflict measure.

        For more information, see Pal et al. 1993. Uncertainty measures for evidential reasoning II:
        A new measure of total uncertainty. International Journal of Approximate Reasoning.

        Only works for normalized mass functions.
        If the mass function is unnormalized, the method returns float('nan')

        In case the mass function is a probability function (containing only singleton hypotheses),
        it reduces to the classical entropy measure.
        """
        if self[frozenset()] > 0.0:
            return float("nan")
        c = 0.0
        for h, v in self.items():
            if v > 0.0:
                c += v * log2(len(h) / v)
        return c

    def hartley_measure(self):
        """
        Computes the Hartley-like measure in order to quantify the amount of imprecision.

        For more information, see:
        G. J. Klir (1999), "Uncertainty and information measures for imprecise probabilities: An overview",
        International Symposium on Imprecise Probabilities and Their Applications.
        """
        return fsum([v * log2(len(h)) for h, v in self.items()])

    def norm(self, m, p=2):
        """
        Computes the p-norm between two mass functions (default is p=2).

        Both mass functions are treated as vectors of mass values.
        """
        d = fsum([(v - m[h]) ** p for (h, v) in self.items()])
        for h, v in m.items():
            if h not in self:
                d += v**p
        return d ** (1.0 / p)

    def is_compatible(self, m: "MassFunction") -> bool:
        """
        Checks whether another mass function is compatible with this one.

        Compatibility means that the mass value of each hypothesis in 'm' is less than
        or equal to the corresponding plausibility given by this mass function.
        """
        return all(self.pl(h) >= v for (h, v) in m.items())

    @overload
    def sample(
        self,
        n: int,
        quantization: bool = ...,
        as_dict: Literal[False] = False,
    ) -> list[frozenset[K]]: ...

    @overload
    def sample(
        self,
        n: int,
        quantization: bool = ...,
        as_dict: Literal[True] = True,
    ) -> dict[frozenset[K], int]: ...

    def sample(
        self, n: int, quantization: bool = True, as_dict: bool = False
    ) -> list[frozenset[K]] | dict[frozenset[K], int]:
        """
        Returns n random samples from the mass distribution.

        Hypotheses are drawn with a probability proportional to their mass values (with replacement).

        If 'quantization' is True (default), the method performs a quantization of the mass values.
        This means the frequency of a hypothesis h in the sample set is at least int(self[h] * n / t) where t is the sum of all mass values.
        The remaining sample slots (if any) are filled up according to the remainders of the fractions computed in the first step.

        The parameter 'as_dict' determines the type of the returned value.
        If 'as_dict' is False (default), a list of length n is returned.
        Otherwise, the result is a dictionary specifying the number of samples for each hypothesis.
        """
        if not isinstance(n, int):
            raise TypeError("n must be int")
        samples = {h: 0 for h in self} if as_dict else []
        mass_sum = fsum(self.values())
        if quantization:
            remainders = []
            remaining_sample_count = n
            for h, v in self.items():
                fraction = n * v / mass_sum
                quotient = int(fraction)
                if quotient > 0:
                    if as_dict:
                        assert isinstance(samples, dict)
                        samples[h] = quotient
                    else:
                        assert isinstance(samples, list)
                        samples.extend([h] * quotient)
                remainders.append((h, fraction - quotient))
                remaining_sample_count -= quotient
            remainders.sort(reverse=True, key=lambda hv: hv[1])
            for h, _ in remainders[:remaining_sample_count]:
                if as_dict:
                    assert isinstance(samples, dict)
                    samples[h] += 1
                else:
                    assert isinstance(samples, list)
                    samples.append(h)
        else:
            rv = [uniform(0.0, mass_sum) for _ in range(n)]
            hypotheses = sorted(self.items(), reverse=True, key=lambda hv: hv[1])
            for i in range(n):
                mass = 0.0
                for h, v in hypotheses:
                    mass += v
                    if mass >= rv[i]:
                        if as_dict:
                            assert isinstance(samples, dict)
                            samples[h] += 1
                        else:
                            assert isinstance(samples, list)
                            samples.append(h)
                        break
        if isinstance(samples, list):
            shuffle(samples)
            return samples
        assert isinstance(samples, dict)
        return samples

    def is_probabilistic(self) -> bool:
        """
        Checks whether the mass function is a probability function.

        Returns True if and only if all hypotheses are singletons (normalization is ignored).
        """
        return all(len(h) == 1 for h in self.keys())

    def sample_probability_distributions(self, n: int) -> list["MassFunction[K]"]:
        """
        Randomly generates n compatible probability distributions from the mass function.

        The result is a list of n independently sampled probability distributions expressed as mass functions.
        This can be useful for estimating various statistical measures like the minimum or maximum entropy consistent with the mass distribution.
        """
        samples = [MassFunction[K]() for _ in range(n)]
        for i in range(n):
            for h, v in self.items():
                if len(h) == 1:
                    samples[i][h] += v
                else:
                    rv = [random() for _ in range(len(h))]
                    total = fsum(rv)
                    for k, s in enumerate(h):
                        samples[i][{s}] += rv[k] * v / total
        return samples

    def max_bel(self) -> frozenset[K] | None:
        """
        Returns the singleton with the highest belief.

        In case there are multiple singletons with maximum belief, only one of them is returned.
        Returns None, if the mass function does not contain any hypotheses.
        """
        return self._max_singleton(self.bel)

    def max_pl(self) -> frozenset[K] | None:
        """
        Returns the singleton with the highest plausibility.

        In case there are multiple singletons with maximum plausibility, only one of them is returned.
        Returns None, if the mass function does not contain any hypotheses.
        """
        return self._max_singleton(self.pl)

    def _max_singleton(self, f):
        st = self.singletons()
        if st:
            value_list = [(f(s), s) for s in st]
            shuffle(value_list)
            return max(value_list)[1]
        else:
            return None

    def to_dict(self) -> dict[K, float]:
        """Convert a mass function only consisting of singletons to a dictionary by removing each enclosing frozenset."""
        if not self.is_probabilistic():
            raise NotProbabilisticError("mass function must only contain singletons")
        return {next(iter(h)): v for h, v in self.items()}

    @staticmethod
    def from_dict(d: "MassFunction[K] | Mapping[K, float]") -> "MassFunction[K]":
        """Convert a dictionary to a mass function by enclosing each key with a frozenset."""
        if isinstance(d, MassFunction):
            return d
        else:
            # src: dict[frozenset[K], float]
            src = {frozenset((h,)): v for h, v in d.items()}
            # MassFunction wants: Mapping[Iterable[K], float]
            # which is compatible?! but we still get an error
            return MassFunction(src)  # type: ignore

    @staticmethod
    def from_possibility(poss):
        """
        Constructs a consonant mass function from a possibility distribution.

        For more information, see:
        D. Dubois, H. Prade (1982), "On several representations of an uncertainty body of evidence",
        Fuzzy Information and Decision Processes, 167-181.
        """
        if isinstance(poss, MassFunction):
            poss = poss.to_dict()  # remove enclosing sets
        H, P = zip(
            *sorted(poss.items(), key=lambda e: e[1], reverse=True)
        )  # sort possibility values in descending order
        m = MassFunction()
        m[H] = P[-1]
        for i in range(len(H) - 1):
            m[H[: i + 1]] = P[i] - P[i + 1]
        return m

    @staticmethod
    def pignistic_inverse(p):
        """
        Constructs a consonant mass function from a pignistic probability distribution by applying the inverse pignistic transformation.

        For more information, see:
        D. Dubois, H. Prade, P. Smets (2008), "A definition of subjective possibility",
        International Journal of Approximate Reasoning 48 (2), 352-364.
        """
        p = MassFunction.from_dict(p)
        poss = MassFunction({h1: fsum([min(p[h1], p[h2]) for h2 in p]) for h1 in p})
        return MassFunction.from_possibility(poss)

    @staticmethod
    def _to_array_index(hypothesis, frame):
        """Map a hypothesis to an array index given a frame of discernment."""
        index = 0
        for i, s in enumerate(frame):
            if s in hypothesis:
                index += 2**i
        return index

    @staticmethod
    def _from_array_index(index, frame):
        """Map an array index to a hypothesis given a frame of discernment."""
        hypothesis = set()
        for i, s in enumerate(frame):
            if 2**i & index:
                hypothesis.add(s)
        return frozenset(hypothesis)

    def to_array(self, frame):
        """
        Convert the mass function to a NumPy array.

        Hypotheses are mapped to array indices using '_to_array_index'.
        The resulting array has 2^n entries where n is the size of the frame of discernment.
        """
        a = numpy.zeros(2 ** len(frame))
        for h, v in self.items():
            a[MassFunction._to_array_index(h, frame)] = v
        return a

    @staticmethod
    def from_array(a, frame):
        """
        Convert a NumPy array to a mass function given a frame of discernment.

        Array indices are mapped to hypotheses using '_from_array_index'.
        """
        m = MassFunction()
        for i, v in enumerate(a):
            if v > 0.0:
                m[MassFunction._from_array_index(i, frame)] = v
        return m

    @staticmethod
    def _confidence_intervals(histogram, alpha):
        """Compute Goodman confidence intervals."""
        p_lower = {}
        p_upper = {}
        a = chi2.ppf(1.0 - alpha / len(histogram), 1)
        n = float(sum(histogram.values()))
        for h, n_h in histogram.items():
            delta_h = a * (a + 4.0 * n_h * (n - n_h) / n)
            p_lower[h] = (a + 2.0 * n_h - sqrt(delta_h)) / (2.0 * (n + a))
            p_upper[h] = (a + 2.0 * n_h + sqrt(delta_h)) / (2.0 * (n + a))
        return p_lower, p_upper

    @staticmethod
    def from_samples(histogram, method="idm", alpha=0.05, s=1.0):
        """
        Generate a mass function from an empirical probability distribution that was obtained from a limited number of samples.
        This makes the expected deviation of the empirical distribution from the true distribution explicit.

        'histogram' represents the empirical distribution. It is a dictionary mapping each possible event to the respective
        number of observations (represented as integers).

        'method' determines the algorithm used for generating the mass function.
        Except for method 'bayesian', all algorithms are based on the idea that the true probabilities lie within confidence intervals
        represented by the mass function with confidence level 1 - 'alpha'.

        The following modes are supported:

        'idm': Imprecise Dirichlet model. A small amount of mass (controlled by 's') is assigned to the entire frame.
        For more information on 'idm', see:
        P. Walley (1996), "Inferences from multinomial data: learning about a bag of marbles",
        Journal of the Royal Statistical Society. Series B (Methodological), 3-57.

        'maxbel': Maximize the total belief by solving a linear program. (Attention: this becomes very computationally expensive
        for larger numbers of events.)

        'maxbel-ordered': Similar to 'maxbel' except that the events are assumed to have a natural order (e.g., intervals), in which case
        the mass function can be computed analytically and thus much faster.

        For more information on 'maxbel' and 'maxbel-ordered', see:
        T. Denoeux (2006), "Constructing belief functions from sample data using multinomial confidence regions",
        International Journal of Approximate Reasoning 42, 228-252.

        'mcd': Compute the least committed consonant mass function whose pignistic transformation lies within the confidence interval
        induced by 'alpha'. Like 'maxbel', it is based on solving a linear program and quickly becomes computationally expensive.

        'mcd-approximate': An approximation of 'mcd' that can be computed much more efficiently.

        For more information on these two methods, see:
        A. Aregui, T. Denoeux (2008), "Constructing consonant belief functions from sample data using confidence sets of pignistic probabilities",
        International Journal of Approximate Reasoning 49, 575-594.

        'bayesian': Construct a Bayesian mass function based on the relative frequencies. In addition, additive smoothing is applied (controlled by 's').

        In case the sample number is 0, returns a vacuous mass function (or uniform distribution for 'bayesian').

        (Requires SciPy for computing confidence intervals and solving linear programs.)
        """
        if not isinstance(histogram, dict):
            raise TypeError("histogram must be of type dict")
        for v in histogram.values():
            if not isinstance(v, int):
                raise TypeError("all histogram values must be of type int")
        if not histogram:
            return MassFunction()
        if (
            sum(histogram.values()) == 0
        ):  # return vacuous/uniform belief if there are no samples
            vac = MassFunction({tuple(histogram.keys()): 1})
            if method == "bayesian":
                return vac.pignistic()
            else:
                return vac
        if method == "bayesian":
            return MassFunction({(h,): v + s for h, v in histogram.items()}).normalize()
        elif method == "idm":
            return MassFunction._from_samples_idm(histogram, s)
        elif method == "maxbel":
            return MassFunction._from_samples_maxbel(histogram, alpha)
        elif method == "maxbel-ordered":
            return MassFunction._from_samples_maxbel(histogram, alpha, ordered=True)
        elif method == "mcd":
            return MassFunction._from_samples_mcd(histogram, alpha)
        elif method == "mcd-approximate":
            return MassFunction._from_samples_mcd(histogram, alpha, approximate=True)
        raise ValueError(f"unknown method: {method}")

    @staticmethod
    def _from_samples_idm(histogram, s):
        """
        Reference:
        P. Walley (1996), "Inferences from multinomial data: learning about a bag of marbles",
        Journal of the Royal Statistical Society. Series B (Methodological), 3-57.
        """
        total = sum(histogram.values())
        m = MassFunction()
        for h, c in histogram.items():
            m[(h,)] = float(c) / (total + s)
        m[MassFunction._convert(histogram.keys())] = float(s) / (total + s)
        return m

    @staticmethod
    def _from_samples_maxbel(histogram, alpha, ordered=False):
        """
        Reference:
        T. Denoeux (2006), "Constructing belief functions from sample data using multinomial confidence regions",
        International Journal of Approximate Reasoning 42, 228-252.
        """
        p_lower, p_upper = MassFunction._confidence_intervals(histogram, alpha)

        def p_lower_set(hs):
            l = u = 0
            for h in H:
                if h in hs:
                    l += p_lower[h]
                else:
                    u += p_upper[h]
            return max(l, 1 - u)

        if ordered:
            H = sorted(histogram.keys())
            m = MassFunction()
            for i1, h1 in enumerate(H):
                m[(h1,)] = p_lower[h1]
                for i2, h2 in enumerate(H[i1 + 1 :]):
                    i2 += i1 + 1
                    if i2 == i1 + 1:
                        v = p_lower_set(H[i1 : i2 + 1]) - p_lower[h1] - p_lower[h2]
                    else:
                        v = (
                            p_lower_set(H[i1 : i2 + 1])
                            - p_lower_set(H[i1 + 1 : i2 + 1])
                            - p_lower_set(H[i1:i2])
                            + p_lower_set(H[i1 + 1 : i2])
                        )
                    if v > 0:
                        m[H[i1 : i2 + 1]] = v
            return m
        else:
            H = list(histogram.keys())
            L = 2 ** len(H)

            def singletons(index):
                return [i for i in range(len(H)) if 2**i & index]

            # Objective: maximize sum_i m_i * 2^(-|i|)
            # linprog minimizes, so use negative coefficients
            c = numpy.array(
                [-(2.0 ** (-len(singletons(i)))) for i in range(L)],
                dtype=float,
            )

            # Constraint (24): bel(A_i) <= p_lower_set(A_i)
            A_ub = numpy.zeros((L, L), dtype=float)
            b_ub = numpy.zeros(L, dtype=float)

            for i in range(L):
                for j in range(L):
                    if (j & i) == j:
                        A_ub[i, j] = 1.0
                b_ub[i] = p_lower_set(MassFunction._from_array_index(i, H))

            # Constraint (25): sum_A m(A) = 1
            A_eq = numpy.ones((1, L), dtype=float)
            b_eq = numpy.array([1.0], dtype=float)

            result = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=(0.0, None),  # Constraint (26): m(A) >= 0
                method="highs",
            )

            if not result.success:
                raise ValueError(f"maxbel linear program failed: {result.message}")

            m_optimal = numpy.maximum(result.x, 0.0)
            m_optimal[m_optimal < LP_ZERO_TOLERANCE] = 0.0

            return MassFunction.from_array(m_optimal, H)

    @staticmethod
    def _from_samples_mcd(histogram, alpha, approximate=False):
        """
        Reference:
        A. Aregui, T. Denoeux (2008), "Constructing consonant belief functions from sample data using confidence
        sets of pignistic probabilities", International Journal of Approximate Reasoning 49, 575-594.
        """
        p_lower, p_upper = MassFunction._confidence_intervals(histogram, alpha)
        H = list(histogram.keys())

        if approximate:
            # approximate possibility distribution
            poss = {
                h1: min(1, fsum([min(p_upper[h1], p_upper[h2]) for h2 in H]))
                for h1 in H
            }
        else:
            # optimal possibility distribution (based on linear programming)
            poss = {h: 0.0 for h in H}
            n = len(H)

            # Constraint (26)
            bounds = [(p_lower[h], p_upper[h]) for h in H]

            # Constraint (27): sum(p) == 1
            A_eq = numpy.ones((1, n), dtype=float)
            b_eq = numpy.array([1.0], dtype=float)

            for k, h_k in enumerate(H):
                S_k = {l for l in range(n) if p_lower[H[l]] >= p_upper[h_k]}
                S_k.add(k)
                I_k = {l for l in range(n) if p_upper[H[l]] < p_lower[h_k]}
                P_k = set(range(n)).difference(S_k.union(I_k))

                for A in powerset(P_k):
                    G = S_k.union(A)
                    G_c = set(range(n)).difference(G)

                    # Objective: maximize sum_{i in G_c} p_i + len(G) * p_k
                    # linprog minimizes, so use negative coefficients
                    c = numpy.zeros(n, dtype=float)
                    for i in G_c:
                        c[i] -= 1.0
                    c[k] -= float(len(G))

                    rows = []
                    rhs = []

                    # Constraint (30): p_i >= p_k for i in G
                    # as: p_k - p_i <= 0
                    for i in G:
                        if i == k:
                            continue
                        row = numpy.zeros(n, dtype=float)
                        row[k] = 1.0
                        row[i] = -1.0
                        rows.append(row)
                        rhs.append(0.0)

                    # Constraint (31): p_k >= p_i for i in G_c
                    # as: p_i - p_k <= 0
                    for i in G_c:
                        row = numpy.zeros(n, dtype=float)
                        row[i] = 1.0
                        row[k] = -1.0
                        rows.append(row)
                        rhs.append(0.0)

                    A_ub = numpy.array(rows, dtype=float) if rows else None
                    b_ub = numpy.array(rhs, dtype=float) if rows else None

                    result = linprog(
                        c,
                        A_ub=A_ub,
                        b_ub=b_ub,
                        A_eq=A_eq,
                        b_eq=b_eq,
                        bounds=bounds,
                        method="highs",
                    )

                    if result.success:
                        poss[h_k] = max(poss[h_k], -float(result.fun))
                    elif result.status == 2:
                        # Infeasible subset, skip it.
                        continue
                    else:
                        raise ValueError(
                            f"mcd linear program failed for event {h_k!r}: {result.message}"
                        )

            # Possibility values should lie in [0, 1].
            for h in poss:
                if poss[h] < 0.0:
                    poss[h] = 0.0
                elif poss[h] > 1.0:
                    poss[h] = 1.0

        return MassFunction.from_possibility(poss)


def powerset(iterable: Iterable[K]):
    """
    Returns an iterator over the power set of 'set'.

    'set' is an arbitrary iterator over hashable elements.
    All returned subsets are of type 'frozenset'.
    """
    items = tuple(iterable)
    return map(
        frozenset,
        chain.from_iterable(combinations(items, r) for r in range(len(items) + 1)),
    )


def gbt_m(hypothesis, likelihoods, normalization=True):
    """
    Computes the mass value of 'hypothesis' using the generalized Bayesian theorem.

    Equivalent to MassFunction.gbt(likelihoods, normalization)[hypothesis].
    """
    if isinstance(likelihoods, dict):
        likelihoods = list(likelihoods.items())
    q = gbt_q(hypothesis, likelihoods, normalization)
    return q * reduce(
        mul, [1.0 - l[1] for l in likelihoods if l[0] not in hypothesis], 1.0
    )


def gbt_bel(hypothesis, likelihoods, normalization=True):
    """
    Computes the belief of 'hypothesis' using the generalized Bayesian theorem.

    Equivalent to MassFunction.gbt(likelihoods, normalization).bel(hypothesis).
    """
    if isinstance(likelihoods, dict):
        likelihoods = list(likelihoods.items())
    eta = _gbt_normalization(likelihoods) if normalization else 1.0
    exc = reduce(mul, [1.0 - l[1] for l in likelihoods if l[0] not in hypothesis], 1.0)
    all_hyp = reduce(mul, [1.0 - l[1] for l in likelihoods], 1.0)
    return eta * (exc - all_hyp)


def gbt_pl(hypothesis, likelihoods, normalization=True):
    """
    Computes the plausibility of 'hypothesis' using the generalized Bayesian theorem.

    Equivalent to MassFunction.gbt(likelihoods, normalization).pl(hypothesis).
    """
    if isinstance(likelihoods, dict):
        likelihoods = list(likelihoods.items())
    eta = _gbt_normalization(likelihoods) if normalization else 1.0
    return eta * (
        1.0 - reduce(mul, [1.0 - l[1] for l in likelihoods if l[0] in hypothesis], 1.0)
    )


def gbt_q(hypothesis, likelihoods, normalization=True):
    """
    Computes the commonality of 'hypothesis' using the generalized Bayesian theorem.

    Equivalent to MassFunction.gbt(likelihoods, normalization).q(hypothesis).
    """
    if isinstance(likelihoods, dict):
        likelihoods = list(likelihoods.items())
    eta = _gbt_normalization(likelihoods) if normalization else 1.0
    return eta * reduce(mul, [l[1] for l in likelihoods if l[0] in hypothesis], 1.0)


def gbt_pignistic(singleton, likelihoods):
    """
    Computes the pignistic probability of 'singleton' for the belief function obtained
    by applying the generalized Bayesian theorem to 'likelihoods'.

    This function has time complexity O(len(likelihoods)**2) and is equivalent to the following
    expression (which has exponential complexity):
    MassFunction.gbt(likelihoods).pignistic()[(singleton,)]
    """
    if isinstance(likelihoods, dict):
        likelihoods = list(likelihoods.items())
    singleton_lh = None
    lh_values = []
    for h, v in likelihoods:
        if h == singleton:
            singleton_lh = v
        else:
            lh_values.append(v)
    if singleton_lh is None:
        raise ValueError(f"singleton {singleton!r} is not contained in likelihoods")
    m_sum = _gbt_pignistic_recursive(lh_values, 0)
    eta = _gbt_normalization(likelihoods)
    return sum([eta * v * singleton_lh / (c + 1.0) for c, v in enumerate(m_sum)])


def _gbt_pignistic_recursive(likelihoods, i):
    """
    Helper function for recursively computing the pignistic probability corresponding to the GBT.

    This function computes the sum over all mass values (obtained via the GBT) and groups them by the
    cardinalities of the corresponding sets.
    """
    if i == len(likelihoods) - 1:
        m_sum = [0.0] * (len(likelihoods) + 1)
        m_sum[0] = 1.0 - likelihoods[i]
        m_sum[1] = likelihoods[i]
        return m_sum
    else:
        m_sum = _gbt_pignistic_recursive(likelihoods, i + 1)
        m_sum_inc = [0.0] * (len(likelihoods) + 1)
        m_sum_exc = [0.0] * (len(likelihoods) + 1)
        for k in range(len(likelihoods) + 1):
            if k < len(likelihoods):
                m_sum_inc[k + 1] = m_sum[k] * likelihoods[i]
            m_sum_exc[k] = m_sum[k] * (1.0 - likelihoods[i])
            m_sum[k] = m_sum_inc[k] + m_sum_exc[k]
        return m_sum


def _gbt_normalization(likelihoods):
    """Helper function for computing the GBT normalization constant."""
    return 1.0 / (1.0 - reduce(mul, [1.0 - l[1] for l in likelihoods], 1.0))
