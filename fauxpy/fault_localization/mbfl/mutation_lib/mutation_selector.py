import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type


class MutationSelector(ABC):
    """
    Selects which (module_path, line_number) pairs to mutate when the
    number of candidate pairs exceeds the configured budget.
    """

    @abstractmethod
    def select(
        self, module_line_pair_list: List[Tuple[str, int]], budget: int
    ) -> List[Tuple[str, int]]:
        """
        Args:
            module_line_pair_list (List[Tuple[str, int]]): All candidate
                (module_path, line_number) pairs.
            budget (int): The maximum number of pairs to return.

        Returns:
            List[Tuple[str, int]]: The selected subset of
                module_line_pair_list, of length min(budget, len(module_line_pair_list)).
        """
        raise NotImplementedError


class RandomMutationSelector(MutationSelector):
    """Selects pairs uniformly at random."""

    def select(
        self, module_line_pair_list: List[Tuple[str, int]], budget: int
    ) -> List[Tuple[str, int]]:
        num_pairs_to_select = min(budget, len(module_line_pair_list))
        return random.sample(module_line_pair_list, num_pairs_to_select)


_MUTATION_SELECTOR_REGISTRY: Dict[str, Type[MutationSelector]] = {
    "random": RandomMutationSelector,
}


def get_available_mutation_selection_strategies() -> List[str]:
    """
    Returns:
        List[str]: The names of all registered mutation selection strategies.
    """
    return list(_MUTATION_SELECTOR_REGISTRY.keys())


def get_mutation_selector(strategy_name: str) -> MutationSelector:
    """
    Looks up and instantiates the mutation selector registered under strategy_name.

    Args:
        strategy_name (str): The name of the registered strategy.

    Returns:
        MutationSelector: An instance of the corresponding selector.

    Raises:
        ValueError: If strategy_name is not registered.
    """
    if strategy_name not in _MUTATION_SELECTOR_REGISTRY:
        available_strategies = ", ".join(get_available_mutation_selection_strategies())
        raise ValueError(
            f"Unknown mutation selection strategy '{strategy_name}'. "
            f"Available strategies: {available_strategies}."
        )
    return _MUTATION_SELECTOR_REGISTRY[strategy_name]()
