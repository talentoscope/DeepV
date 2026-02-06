"""
Structured logging system for DeepV.

Provides unified, configurable logging with performance timing, structured context,
and proper error handling across the entire pipeline.
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    from omegaconf import OmegaConf
    logging_config = OmegaConf.load(Path(__file__).parent.parent.parent / "config" / "logging" / "default.yaml")
except:
    # Fallback config
    class FallbackConfig:
        level = "INFO"
        format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        file_enabled = False
        file_path = "logs/deepv.log"
        json_format = False
        performance_timing = True
        context_fields = ["module", "operation", "batch_size"]
    logging_config = FallbackConfig()


class StructuredLogger(logging.Logger):
    """Enhanced logger with structured logging, timing, and context support."""

    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)
        self.context: Dict[str, Any] = {}

    def with_context(self, **context) -> 'StructuredLogger':
        """Return a logger with additional context fields."""
        new_logger = StructuredLogger(self.name, self.level)
        new_logger.context = {**self.context, **context}
        new_logger.handlers = self.handlers[:]
        new_logger.propagate = self.propagate
        return new_logger

    @contextmanager
    def timing(self, operation: str, log_level: int = logging.INFO):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.log(log_level, f"Operation '{operation}' completed in {duration:.3f}s",
                    extra={"operation": operation, "duration": duration, "timing": True})

    def log_performance(self, operation: str, metrics: Dict[str, Union[int, float]],
                       log_level: int = logging.INFO):
        """Log performance metrics."""
        self.log(log_level, f"Performance metrics for '{operation}': {metrics}",
                extra={"operation": operation, "metrics": metrics, "performance": True})

    def log_error(self, error: Exception, operation: str = None, **extra_context):
        """Log an exception with context."""
        context = {"exception_type": type(error).__name__, "exception_message": str(error)}
        if operation:
            context["operation"] = operation
        context.update(extra_context)

        self.error(f"Exception in {operation or 'unknown operation'}: {error}",
                  extra=context, exc_info=True)

    def log_pipeline_step(self, step: str, status: str = "started",
                         progress: Optional[float] = None, **extra):
        """Log pipeline step progress."""
        message = f"Pipeline step '{step}' {status}"
        if progress is not None:
            message += f" ({progress:.1%})"

        core_data = {
            "pipeline_step": step,
            "status": status,
            "progress": progress,
        }

        # Put caller-supplied extra fields under a `context` key to avoid
        # overwriting reserved LogRecord attributes like 'module'. This keeps
        # structured data available while remaining compatible with the
        # Python logging system.
        payload = {**core_data, "context": extra}

        self.info(message, extra=payload)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        # Extract extra fields
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'message'
            }:
                extra_fields[key] = value

        log_entry = {
            "timestamp": self.formatTime(record, self.default_time_format),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if extra_fields:
            log_entry["extra"] = extra_fields

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class StructuredFormatter(logging.Formatter):
    """Enhanced formatter with context support."""

    def __init__(self, fmt: str = None, include_context: bool = True):
        if fmt is None:
            fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        super().__init__(fmt)
        self.include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        # Add context fields to the record
        if hasattr(record, 'context') and record.context and self.include_context:
            context_str = " ".join(f"[{k}={v}]" for k, v in record.context.items())
            record.msg = f"{record.msg} {context_str}"

        return super().format(record)


def create_structured_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    include_context: bool = True
) -> StructuredLogger:
    """
    Create a structured logger with configurable output.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        json_format: Whether to use JSON formatting
        include_context: Whether to include context in log messages

    Returns:
        Configured StructuredLogger instance
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create logger
    logging.setLoggerClass(StructuredLogger)
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = StructuredFormatter(include_context=include_context)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_pipeline_logger(module: str = "deepv") -> StructuredLogger:
    """Get a logger configured for the DeepV pipeline."""
    return create_structured_logger(
        name=f"deepv.{module}",
        level=logging_config.level if hasattr(logging_config, 'level') else "INFO",
        log_file=logging_config.file_path if hasattr(logging_config, 'file_path') else None,
        json_format=getattr(logging_config, 'json_format', False),
        include_context=getattr(logging_config, 'include_context', True)
    )


# Convenience functions for common logging patterns
def log_pipeline_start(logger: StructuredLogger, pipeline_name: str, **context):
    """Log the start of a pipeline operation."""
    logger.with_context(**context).log_pipeline_step(pipeline_name, "started")

def log_pipeline_progress(logger: StructuredLogger, pipeline_name: str, progress: float, **context):
    """Log pipeline progress."""
    logger.with_context(**context).log_pipeline_step(pipeline_name, "in_progress", progress)

def log_pipeline_complete(logger: StructuredLogger, pipeline_name: str, **context):
    """Log pipeline completion."""
    logger.with_context(**context).log_pipeline_step(pipeline_name, "completed")

def log_performance_metrics(logger: StructuredLogger, operation: str, **metrics):
    """Log performance metrics."""
    logger.log_performance(operation, metrics)