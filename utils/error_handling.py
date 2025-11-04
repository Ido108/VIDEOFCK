"""
Error handling utilities with exponential backoff and retry logic.
"""
import time
import logging
from functools import wraps
from typing import Callable, Any, Optional, Tuple, Type
from constants import API_MAX_RETRIES, API_RETRY_DELAY_SECONDS

logger = logging.getLogger(__name__)


class RetryError(Exception):
    """Raised when all retry attempts fail."""
    pass


def exponential_backoff_retry(
    max_retries: int = API_MAX_RETRIES,
    base_delay: float = API_RETRY_DELAY_SECONDS,
    exponential_base: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Decorator that implements exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        max_delay: Maximum delay between retries
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function called on each retry attempt

    Returns:
        Decorated function with retry logic

    Example:
        @exponential_backoff_retry(max_retries=3, base_delay=1.0)
        def api_call():
            response = requests.get("https://api.example.com")
            return response.json()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries - 1:
                        # Calculate delay with exponential backoff
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )

                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {delay:.2f}s..."
                        )

                        # Call retry callback if provided
                        if on_retry:
                            on_retry(attempt + 1, e)

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed for {func.__name__}: {str(e)}"
                        )

            # All retries exhausted
            raise RetryError(
                f"Failed after {max_retries} attempts. Last error: {str(last_exception)}"
            ) from last_exception

        return wrapper
    return decorator


def with_timeout(timeout_seconds: float):
    """
    Decorator that adds timeout capability to functions.

    Note: This uses threading and may not work with all functions.
    For API calls, prefer using library-specific timeout parameters.

    Args:
        timeout_seconds: Maximum execution time in seconds

    Returns:
        Decorated function with timeout
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import threading

            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)

            if thread.is_alive():
                logger.error(f"Function {func.__name__} timed out after {timeout_seconds}s")
                raise TimeoutError(
                    f"Function {func.__name__} exceeded timeout of {timeout_seconds}s"
                )

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute a function and return a default value on error.

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        default: Default value to return on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for the function

    Returns:
        Function result or default value on error

    Example:
        result = safe_execute(risky_function, arg1, arg2, default=None)
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Error executing {func.__name__}: {str(e)}", exc_info=True)
        return default


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for API calls.

    Prevents cascading failures by temporarily disabling calls after
    a threshold of consecutive failures.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that triggers the circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "open":
            if (
                self.last_failure_time
                and time.time() - self.last_failure_time >= self.recovery_timeout
            ):
                self.state = "half-open"
                logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception("Circuit breaker is OPEN - calls are blocked")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == "half-open":
            self.state = "closed"
            logger.info("Circuit breaker closed after successful call")

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.error(
                f"Circuit breaker opened after {self.failure_count} consecutive failures"
            )


# Pre-configured retry decorators for common scenarios
api_retry = exponential_backoff_retry(
    max_retries=API_MAX_RETRIES,
    base_delay=API_RETRY_DELAY_SECONDS,
    exceptions=(Exception,)
)

network_retry = exponential_backoff_retry(
    max_retries=4,
    base_delay=2.0,
    max_delay=16.0,
    exceptions=(ConnectionError, TimeoutError)
)
