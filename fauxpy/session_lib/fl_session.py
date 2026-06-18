from abc import abstractmethod, ABC


class FlSession(ABC):
    @abstractmethod
    def run_test_call(self, item):
        """
        Runs before the execution of the current test.
        """
        pass

    @abstractmethod
    def run_test_make_report(self, item, call):
        """
        Runs after the execution of the current test.
        """
        pass

    @abstractmethod
    def terminal_summary(self, terminal_reporter, exit_status):
        """
        Runs after the execution of all tests.
        """
        pass

    def get_extra_metrics(self) -> dict:
        """
        Returns additional session-specific metrics to be merged into the
        delta time report (e.g., MBFL's mutant generation/validation timing).
        Sessions that have nothing extra to report can rely on this default.
        """
        return {}
