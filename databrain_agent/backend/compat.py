"""Compatibility shims for LangChain version differences."""
import sys

# Try to create a compatibility shim for langchain_core.pydantic_v1 if it doesn't exist
try:
    from langchain_core import pydantic_v1
    # If it exists, we're good
except ImportError:
    # If langchain_core.pydantic_v1 doesn't exist, try to create a compatibility shim
    try:
        # Try pydantic.v1 first (available in pydantic v2)
        import pydantic.v1 as pydantic_v1_module
        # Create a fake module to satisfy imports
        import types
        pydantic_v1 = types.ModuleType('pydantic_v1')
        
        # Copy common classes from pydantic.v1
        for attr in ['BaseModel', 'Field', 'validator', 'root_validator', 'Extra', 'ValidationError', 'BaseSettings']:
            if hasattr(pydantic_v1_module, attr):
                setattr(pydantic_v1, attr, getattr(pydantic_v1_module, attr))
        
        # Add to sys.modules so imports work
        sys.modules['langchain_core.pydantic_v1'] = pydantic_v1
    except ImportError:
        # Fallback: use pydantic v2 directly and map to v1 equivalents
        try:
            import pydantic
            import types
            pydantic_v1 = types.ModuleType('pydantic_v1')
            
            # Map pydantic v2 classes to v1 equivalents
            pydantic_v1.BaseModel = pydantic.BaseModel
            pydantic_v1.Field = pydantic.Field
            
            # Try to get Extra enum
            try:
                from pydantic import ConfigDict
                # For pydantic v2, Extra is replaced by ConfigDict
                # Create a simple Extra-like class
                class Extra:
                    allow = "allow"
                    ignore = "ignore"
                    forbid = "forbid"
                pydantic_v1.Extra = Extra
            except ImportError:
                pydantic_v1.Extra = None
            
            # Add validator decorators (v2 uses field_validator)
            try:
                from pydantic import field_validator, model_validator
                pydantic_v1.validator = field_validator
                pydantic_v1.root_validator = model_validator
            except ImportError:
                # Fallback if field_validator doesn't exist
                def validator(*args, **kwargs):
                    def decorator(func):
                        return func
                    return decorator
                pydantic_v1.validator = validator
                pydantic_v1.root_validator = validator
            
            # Add ValidationError
            try:
                from pydantic import ValidationError
                pydantic_v1.ValidationError = ValidationError
            except ImportError:
                pydantic_v1.ValidationError = Exception
            
            # Add BaseSettings if available
            try:
                from pydantic_settings import BaseSettings
                pydantic_v1.BaseSettings = BaseSettings
            except ImportError:
                pydantic_v1.BaseSettings = pydantic.BaseModel
            
            # Add to sys.modules
            sys.modules['langchain_core.pydantic_v1'] = pydantic_v1
        except Exception as e:
            # If all else fails, just create an empty module
            import types
            pydantic_v1 = types.ModuleType('pydantic_v1')
            sys.modules['langchain_core.pydantic_v1'] = pydantic_v1
