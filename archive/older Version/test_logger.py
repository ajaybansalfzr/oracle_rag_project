# scripts/test_logger.py
from scripts.utils.utils import get_logger

logger = get_logger("MyTestApp")


def run_test():
    logger.info("This is an informational message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")


if __name__ == "__main__":
    run_test()
    print("\nTest complete. Check console for logger output.")
