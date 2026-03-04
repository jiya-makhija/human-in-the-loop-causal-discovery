from dataclasses import dataclass


@dataclass
class RunLogger:
    """
    Simple print-based logger.
    Keeps counters to show what's happening.
    """
    verbose: bool = True
    edges_added: int = 0
    edges_skipped_cycle: int = 0
    edges_skipped_dup: int = 0

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg)