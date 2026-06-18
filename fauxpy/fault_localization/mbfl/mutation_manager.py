import logging
import random
from pathlib import Path
from typing import List, Optional

from pyllmut import (
    ModelType,
    MutantGenerator,
    MutantInfo,
    MutationReport,
    PromptInfo,
    ResponseInfo,
)

from fauxpy import constants
from fauxpy.session_lib import naming_lib
from fauxpy.session_lib.fl_type import MutationStrategy

from ...session_lib.fauxpy_printer import fl_print
from .db_manager import MbflDbManager
from .mutation_lib.cosmic_ray import CosmicRayMutantGenerator
from .mutation_lib.mutant import Mutant
from .mutation_lib.mutation_selector import (
    RandomMutationSelector,
    get_mutation_selector,
)


class MutationManager:
    """
    Manages the generation of mutants for given lines in modules,
    utilizing various mutation strategies.
    """

    def __init__(
        self,
        db_manager: MbflDbManager,
        mutation_strategy: MutationStrategy,
        budget: Optional[int] = None,
        mutation_selection_strategy: Optional[str] = None,
    ):
        """
        Initializes the MutationManager with specified database manager and mutation strategy.

        Args:
            db_manager (MbflDbManager): An instance for managing database operations.
            mutation_strategy (MutationStrategy): The strategy to be used for mutant generation.
            budget (Optional[int]): Maximum number of mutants to generate. None means no limit.
            mutation_selection_strategy (Optional[str]): Name of the registered MutationSelector
                strategy used to pick which (module, line) pairs to mutate when budget is set.
                Only consulted for the Traditional mutation strategy; other mutation strategies
                always use random selection. Ignored (and may be left None) when budget is None.
        """
        self._db_manager = db_manager
        self._mutation_strategy = mutation_strategy
        self._budget = budget
        self._mutation_selector = (
            get_mutation_selector(mutation_selection_strategy)
            if mutation_selection_strategy is not None
            else None
        )

    @staticmethod
    def _set_mutant_ids(mutant_list):
        """
        Assigns unique identifiers to each mutant in the provided list.

        Args:
            mutant_list (List[Mutant]): A list of mutants to be assigned IDs.
        """
        for i in range(len(mutant_list)):
            mutant_list[i].set_id(f"M{i}")

    def get_all_mutants_for_failing_line_number_list(
        self, failing_line_number_list
    ) -> List:
        """
        Generates mutants for the given statements. Each statement contains
         information about the module and the line number the statement belongs to.
         When budget is set, randomly selects that number of (module, line) pairs for mutation.

        Args:
            failing_line_number_list (List[str]): A list of statements to generate mutants for.

        Returns:
            List[Mutant]: A list of mutants corresponding to the given statements.
        """
        mutant_list = []

        for statement_name in failing_line_number_list:
            path, line_number = naming_lib.convert_statement_name_to_components(
                statement_name
            )
            self._db_manager.insert_failing_line_number_components(path, line_number)

        failing_module_path_list = (
            self._db_manager.select_distinct_failing_module_paths()
        )

        # Build a list of (module_path, line_number) pairs
        module_line_pairs = []
        for module_path in failing_module_path_list:
            line_number_list = (
                self._db_manager.select_failing_line_numbers_for_module_path(
                    module_path
                )
            )
            for line_number in line_number_list:
                module_line_pairs.append((module_path, line_number))

        # If budget is set, select (module, line) pairs to mutate. The configurable
        # selection strategy (--strategy) only applies to the Traditional mutation
        # strategy; other mutation strategies always fall back to random selection.
        if self._budget is not None and self._budget > 0:
            total_pairs = len(module_line_pairs)
            num_pairs_to_select = min(self._budget, total_pairs)
            if total_pairs > 0:
                fl_print.normal("\nApplying budget constraint:")
                fl_print.normal(f"  Budget limit: {self._budget}")
                fl_print.normal(f"  Total available pairs: {total_pairs}")
                fl_print.normal(f"  Pairs to select: {num_pairs_to_select}")
                selector = (
                    self._mutation_selector
                    if self._mutation_strategy == MutationStrategy.Traditional
                    and self._mutation_selector is not None
                    else RandomMutationSelector()
                )
                selected_pairs = selector.select(module_line_pairs, self._budget)
                fl_print.normal(
                    f"   Selected {len(selected_pairs)} out of {total_pairs} pairs"
                )
                module_line_pairs = selected_pairs
        else:
            fl_print.normal(
                f"\nNo budget constraint. Using all {len(module_line_pairs)} (module, line) pairs."
            )

        # Group selected pairs by module for efficient processing
        module_to_lines = {}
        for module_path, line_number in module_line_pairs:
            if module_path not in module_to_lines:
                module_to_lines[module_path] = []
            module_to_lines[module_path].append(line_number)

        # Generate mutants for selected modules and lines
        for module_path, line_number_list in module_to_lines.items():
            current_module_mutant_list = self._get_module_mutant_list(
                module_path, line_number_list
            )

            mutant_list += current_module_mutant_list

        # A single (module, line) pair can still expand into many mutants
        # (one per applicable mutation operator), so also cap the total
        # number of generated mutants to the budget.
        if (
            self._budget is not None
            and self._budget > 0
            and len(mutant_list) > self._budget
        ):
            fl_print.normal(
                f"\nGenerated {len(mutant_list)} mutants, exceeding the budget of {self._budget}."
            )
            mutant_list = random.sample(mutant_list, self._budget)
            fl_print.normal(
                f"  Randomly selected {self._budget} mutants to respect the budget."
            )

        self._set_mutant_ids(mutant_list)

        return mutant_list

    def _get_module_mutant_list(self, module_path: str, line_number_list: List[int]):
        """
        Generates a list of mutants for a given module and its line numbers,
        based on the configured mutation strategy.

        Args:
            module_path (str): The path of the module to generate mutants for.
            line_number_list (List[int]): A list of line numbers within the module.

        Returns:
            List[Mutant]: A list of generated mutants for the module.
        """

        fl_print.normal(
            f"Generating mutants using the mutation strategy {self._mutation_strategy.name} for the following module:"
        )
        fl_print.normal(f"  {module_path}")

        if self._mutation_strategy == MutationStrategy.Traditional:
            module_mutant_list = self._get_traditional_module_mutant_list(
                module_path, line_number_list
            )
        elif self._mutation_strategy in [
            MutationStrategy.GPT4oMini,
            MutationStrategy.GPT4o,
        ]:
            module_mutant_list = self._get_pyllmut_module_mutant_list(
                module_path, line_number_list
            )
        elif self._mutation_strategy in [
            MutationStrategy.TraditionalWithGPT4oMini,
            MutationStrategy.TraditionalWithGPT4o,
        ]:
            module_mutant_list = self._get_traditional_with_pyllmut_module_mutant_list(
                module_path, line_number_list
            )
        else:
            raise Exception(
                f"Mutation strategy {self._mutation_strategy} is not supported."
            )

        fl_print.normal(f"Number of generated mutants: {len(module_mutant_list)}")

        return module_mutant_list

    @staticmethod
    def _get_traditional_module_mutant_list(
        module_path: str, line_number_list: List[int]
    ) -> List[Mutant]:
        """
        Generates mutants for a module using traditional mutation operators (i.e., Cosmic Ray).

        Args:
            module_path (str): The path of the module to generate mutants for.
            line_number_list (List[int]): A list of line numbers within the module to mutate.

        Returns:
            List[Mutant]: A list of mutants generated using traditional methods.
        """
        mutant_generator = CosmicRayMutantGenerator()
        module_mutant_list = mutant_generator.get_mutants_for_module_and_lines(
            module_path=module_path,
            line_numbers=line_number_list,
            operator_mutation_target_unique=True,
        )
        return module_mutant_list

    def _get_pyllmut_module_mutant_list(
        self, module_path: str, line_number_list: List[int]
    ) -> List[Mutant]:
        """
        Generates mutants for a module using PyLLMut.

        Args:
            module_path (str): The path of the module to generate mutants for.
            line_number_list (List[int]): A list of line numbers within the module to mutate.

        Returns:
            List[Mutant]: A list of mutants generated using PyLLMut.
        """

        assert self._mutation_strategy != MutationStrategy.Traditional

        model_type_map = {
            MutationStrategy.GPT4oMini: ModelType.GPT4oMini,
            MutationStrategy.TraditionalWithGPT4oMini: ModelType.GPT4oMini,
            MutationStrategy.GPT4o: ModelType.GPT4o,
            MutationStrategy.TraditionalWithGPT4o: ModelType.GPT4o,
        }

        module_content = Path(module_path).read_text()
        generator = MutantGenerator(
            module_content=module_content,
            line_number_list=line_number_list,
            mutants_per_line_count=constants.MUTANTS_PER_LINE_COUNT,
            timeout_seconds_per_line=constants.TIMEOUT_SECONDS_PER_LINE,
            model_type=model_type_map[self._mutation_strategy],
        )
        mutation_report = generator.generate()

        timeout_line_list = [
            x.get_line_number() for x in mutation_report.get_timeout_info_list()
        ]
        if len(timeout_line_list) > 0:
            message_for_timeout = (
                f"A timeout occurred for the following lines while using the {self._mutation_strategy.name} strategy. "
                f"Consider increasing the timeout duration or reducing the number of mutants requested per line.\n"
                f"Timeout occurred for lines: {timeout_line_list}"
            )
            fl_print.normal(message_for_timeout)
            logging.warning(message_for_timeout)

        valid_mutant_list = mutation_report.get_valid_mutant_list()

        module_mutant_list = []
        for valid_mutant_info in valid_mutant_list:
            current_mutant = Mutant(
                module_path=module_path,
                operator_name=model_type_map[self._mutation_strategy].name,
                occurrence=-1,
                start_pos=(valid_mutant_info.get_line_number(), -1),
                end_pos=(valid_mutant_info.get_line_number(), -1),
                module_content=valid_mutant_info.get_mutated_module_content(),
                module_diff=valid_mutant_info.get_diff_content().splitlines(),
            )

            module_mutant_list.append(current_mutant)

        # The following is for additional analysis and is not required for the fault localization session.
        self._store_in_db_pyllmut_results(mutation_report)

        return module_mutant_list

    def _get_traditional_with_pyllmut_module_mutant_list(
        self, module_path: str, line_number_list: List[int]
    ) -> List[Mutant]:
        """
        Generates mutants for a module using a combination of traditional mutation operators and gpt-4o-mini.

        Args:
            module_path (str): The path of the module to generate mutants for.
            line_number_list (List[int]): A list of line numbers within the module to mutate.

        Returns:
            List[Mutant]: A list of mutants generated using a combination of traditional mutation operators and gpt-4o-mini.
        """
        assert self._mutation_strategy != MutationStrategy.Traditional

        fl_print.normal("Lines to generate mutants for:", line_number_list)
        traditional_mutant_list: List[Mutant] = (
            self._get_traditional_module_mutant_list(module_path, line_number_list)
        )
        covered_line_number_list = list(
            set([x.get_line_number() for x in traditional_mutant_list])
        )
        covered_line_number_list.sort()
        fl_print.normal(
            "Lines covered by traditional mutation operators:", covered_line_number_list
        )
        uncovered_line_number_list = list(
            set(line_number_list) - set(covered_line_number_list)
        )
        uncovered_line_number_list.sort()
        fl_print.normal(
            "Lines uncovered by traditional mutation operators:",
            uncovered_line_number_list,
        )

        llm_mutant_list = self._get_pyllmut_module_mutant_list(
            module_path, uncovered_line_number_list
        )

        fl_print.normal("Number of traditional mutants", len(traditional_mutant_list))
        fl_print.normal("Number of LLM mutants", len(llm_mutant_list))
        module_mutant_list = traditional_mutant_list + llm_mutant_list

        return module_mutant_list

    def _store_in_db_pyllmut_results(self, mutation_report: MutationReport):
        """
        Stores results from PyLLMut into the database for further analysis.

        Args:
            mutation_report (MutationReport): An object containing the mutation results from PyLLMut.

        Note:
            This method is used for additional analysis and is not required for the fault localization session.
        """

        # The following is for additional analysis and is not required for the fault localization session.
        all_mutant_list: List[MutantInfo] = mutation_report.get_mutant_list()
        for mutant_info_item in all_mutant_list:
            self._db_manager.insert_pyllmut_mutant_info(
                prompt_content=mutant_info_item.get_prompt_content(),
                line_number=mutant_info_item.get_line_number(),
                sent_token_count=mutant_info_item.get_sent_token_count(),
                response_content=mutant_info_item.get_response_content(),
                received_token_count=mutant_info_item.get_received_token_count(),
                diff_content=mutant_info_item.get_diff_content(),
                pre_code_model=mutant_info_item.get_pre_code_model(),
                after_code_model=mutant_info_item.get_after_code_model(),
                pre_code_refined=mutant_info_item.get_pre_code_refined(),
                after_code_refined=mutant_info_item.get_after_code_refined(),
                mutant_type=mutant_info_item.get_mutant_type().name,
            )

        # The following is for additional analysis and is not required for the fault localization session.
        bad_response_info_list: List[ResponseInfo] = (
            mutation_report.get_bad_response_info_list()
        )
        for bad_response_info_item in bad_response_info_list:
            self._db_manager.insert_pyllmut_bad_response_info(
                prompt_content=bad_response_info_item.get_prompt_content(),
                line_number=bad_response_info_item.get_line_number(),
                sent_token_count=bad_response_info_item.get_sent_token_count(),
                response_content=bad_response_info_item.get_response_content(),
                received_token_count=bad_response_info_item.get_received_token_count(),
            )

        # The following is for additional analysis and is not required for the fault localization session.
        timeout_info_list: List[PromptInfo] = mutation_report.get_timeout_info_list()
        for timeout_info_item in timeout_info_list:
            self._db_manager.insert_pyllmut_timeout_info(
                prompt_content=timeout_info_item.get_prompt_content(),
                line_number=timeout_info_item.get_line_number(),
            )
