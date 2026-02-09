#!/usr/bin/env python3
"""
Custom Warnings Module

Custom warning classes for DeepV operations.
Provides specialized warning types for different categories of issues.

Features:
- UndefinedWarning for undefined behavior
- Custom warning hierarchy
- Specialized warning types

Used throughout the codebase for appropriate warning classification.
"""

class UndefinedWarning(RuntimeWarning):
    pass
