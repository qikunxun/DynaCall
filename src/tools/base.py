"""Base implementation for tools or skills."""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from functools import partial, wraps
from inspect import signature
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type, Union, get_type_hints

from pydantic import BaseModel, Field, create_model, ConfigDict


class SchemaAnnotationError(TypeError):
    """Raised when 'args_schema' is missing or has an incorrect type annotation."""
    pass


class ToolException(Exception):
    """An optional exception that tool throws when execution error occurs.

    When this exception is thrown, the agent will not stop working,
    but will handle the exception according to the handle_tool_error
    variable of the tool, and the processing result will be returned
    to the agent as observation, and printed in red on the console.
    """
    pass


class ToolBase(ABC):
    """Base class for all tools."""

    name: str
    description: str
    args_schema: Optional[Type[BaseModel]] = None
    return_direct: bool = False
    stringify_rule: Optional[Callable[..., str]] = None

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool synchronously."""
        pass

    @abstractmethod
    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool asynchronously."""
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make the tool callable."""
        return self.run(*args, **kwargs)

    async def __acall__(self, *args: Any, **kwargs: Any) -> Any:
        """Make the tool async callable."""
        return await self.arun(*args, **kwargs)


def _create_subset_model(
    name: str, model: BaseModel, field_names: List[str]
) -> Type[BaseModel]:
    """Create a pydantic model with only a subset of model's fields."""
    fields = {}
    for field_name in field_names:
        field = model.model_fields[field_name]
        annotation = field.annotation
        if annotation is None:
            annotation = Any
        fields[field_name] = (annotation, field)
    return create_model(name, **fields)


def _get_filtered_args(
    inferred_model: Type[BaseModel],
    func: Callable,
) -> Dict:
    """Get the arguments from a function's signature."""
    schema = inferred_model.model_json_schema()["properties"]
    valid_keys = signature(func).parameters
    return {k: schema[k] for k in valid_keys if k not in ("run_manager", "callbacks")}


def create_schema_from_function(
    model_name: str,
    func: Callable,
) -> Type[BaseModel]:
    """Create a pydantic schema from a function's signature."""
    
    # Get type hints from function
    type_hints = get_type_hints(func)
    
    # Get function signature
    sig = signature(func)
    
    # Create fields for the schema
    fields = {}
    for param_name, param in sig.parameters.items():
        # Skip *args and **kwargs
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
            
        # Get type annotation
        param_type = type_hints.get(param_name, Any)
        
        # Handle optional parameters
        if param.default != param.empty:
            fields[param_name] = (Optional[param_type], Field(default=param.default))
        else:
            fields[param_name] = (param_type, ...)
    
    # Create the model
    return create_model(
        f"{model_name}Schema",
        __config__=ConfigDict(extra="forbid"),
        **fields
    )


class Tool(ToolBase):
    """Tool that takes in function or coroutine directly."""

    def __init__(
        self,
        name: str,
        func: Optional[Callable[..., Any]] = None,
        coroutine: Optional[Callable[..., Awaitable[Any]]] = None,
        description: str = "",
        args_schema: Optional[Type[BaseModel]] = None,
        return_direct: bool = False,
        stringify_rule: Optional[Callable[..., str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize tool."""
        self.name = name
        self.func = func
        self.coroutine = coroutine
        self.description = description
        self.args_schema = args_schema
        self.return_direct = return_direct
        self.stringify_rule = stringify_rule
        
        # Initialize args from schema or function
        if self.args_schema is not None:
            self._args = self.args_schema.model_json_schema()["properties"]
        elif func is not None:
            # Infer schema from function if not provided
            try:
                self.args_schema = create_schema_from_function(name, func)
                self._args = self.args_schema.model_json_schema()["properties"]
            except Exception:
                # Fallback to simple string input
                self._args = {"tool_input": {"type": "string"}}
        else:
            self._args = {"tool_input": {"type": "string"}}

    @property
    def args(self) -> dict:
        """The tool's input arguments."""
        return self._args

    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        """Convert tool input to arguments and kwargs."""
        if isinstance(tool_input, str):
            return (tool_input,), {}
        elif isinstance(tool_input, dict):
            if self.args_schema:
                # Validate against schema
                validated = self.args_schema(**tool_input)
                return (), validated.model_dump()
            else:
                # For simple tools, treat dict as kwargs
                return (), tool_input
        else:
            raise ToolException(f"Unsupported tool input type: {type(tool_input)}")

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool synchronously."""
        if self.func:
            return self.func(*args, **kwargs)
        elif self.coroutine:
            # Run async function synchronously
            return asyncio.run(self.coroutine(*args, **kwargs))
        else:
            raise NotImplementedError("Tool does not support sync execution")

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool asynchronously."""
        if self.coroutine:
            return await self.coroutine(*args, **kwargs)
        elif self.func:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, partial(self.func, *args, **kwargs))
        else:
            raise NotImplementedError("Tool does not support async execution")

    def invoke(self, input: Union[str, Dict], **kwargs: Any) -> Any:
        """Invoke the tool with input."""
        args, kwargs_from_input = self._to_args_and_kwargs(input)
        kwargs.update(kwargs_from_input)
        return self.run(*args, **kwargs)

    async def ainvoke(self, input: Union[str, Dict], **kwargs: Any) -> Any:
        """Invoke the tool asynchronously with input."""
        args, kwargs_from_input = self._to_args_and_kwargs(input)
        kwargs.update(kwargs_from_input)
        return await self.arun(*args, **kwargs)


class StructuredTool(Tool):
    """Tool that can operate on any number of inputs with structured schema."""

    def __init__(
        self,
        name: str,
        func: Optional[Callable[..., Any]] = None,
        coroutine: Optional[Callable[..., Awaitable[Any]]] = None,
        description: str = "",
        args_schema: Optional[Type[BaseModel]] = None,
        return_direct: bool = False,
        stringify_rule: Optional[Callable[..., str]] = None,
        infer_schema: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize structured tool."""
        
        # Determine source function for schema inference
        source_function = func if func is not None else coroutine
        if source_function is None:
            raise ValueError("Function and/or coroutine must be provided")
            
        # Set description from docstring if not provided
        if not description and source_function.__doc__:
            description = source_function.__doc__.strip()
            
        # Create schema if needed
        _args_schema = args_schema
        if _args_schema is None and infer_schema:
            _args_schema = create_schema_from_function(f"{name}Schema", source_function)
            
        # Format description with signature
        sig = signature(source_function)
        formatted_description = f"{name}{sig} - {description}"
            
        super().__init__(
            name=name,
            func=func,
            coroutine=coroutine,
            description=formatted_description,
            args_schema=_args_schema,
            return_direct=return_direct,
            stringify_rule=stringify_rule,
            **kwargs,
        )

    @classmethod
    def from_function(
        cls,
        func: Optional[Callable] = None,
        coroutine: Optional[Callable[..., Awaitable[Any]]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        args_schema: Optional[Type[BaseModel]] = None,
        infer_schema: bool = True,
        **kwargs: Any,
    ) -> StructuredTool:
        """Create tool from a given function."""
        
        if func is not None:
            source_function = func
        elif coroutine is not None:
            source_function = coroutine
        else:
            raise ValueError("Function and/or coroutine must be provided")
            
        name = name or source_function.__name__
        description = description or source_function.__doc__
        
        if description is None:
            raise ValueError(
                "Function must have a docstring if description not provided."
            )
            
        return cls(
            name=name,
            func=func,
            coroutine=coroutine,
            description=description,
            args_schema=args_schema,
            return_direct=return_direct,
            infer_schema=infer_schema,
            **kwargs,
        )


def tool(
    *args: Union[str, Callable],
    return_direct: bool = False,
    args_schema: Optional[Type[BaseModel]] = None,
    infer_schema: bool = True,
) -> Callable:
    """Decorator to create tools from functions."""
    
    def _make_with_name(tool_name: str) -> Callable:
        def _make_tool(dec_func: Callable) -> Tool:
            if inspect.iscoroutinefunction(dec_func):
                coroutine = dec_func
                func = None
            else:
                coroutine = None
                func = dec_func

            if infer_schema or args_schema is not None:
                return StructuredTool.from_function(
                    func,
                    coroutine,
                    name=tool_name,
                    return_direct=return_direct,
                    args_schema=args_schema,
                    infer_schema=infer_schema,
                )
                
            # If someone doesn't want a schema applied
            if func and func.__doc__ is None:
                raise ValueError(
                    "Function must have a docstring if "
                    "description not provided and infer_schema is False."
                )
                
            return Tool(
                name=tool_name,
                func=func,
                coroutine=coroutine,
                description=f"{tool_name} tool",
                return_direct=return_direct,
            )

        return _make_tool

    # Handle different decorator usages
    if len(args) == 1 and isinstance(args[0], str):
        # @tool("search")
        return _make_with_name(args[0])
    elif len(args) == 1 and callable(args[0]):
        # @tool
        return _make_with_name(args[0].__name__)(args[0])
    elif len(args) == 0:
        # @tool(return_direct=True)
        def _partial(func: Callable) -> Tool:
            return _make_with_name(func.__name__)(func)
        return _partial
    else:
        raise ValueError("Too many arguments for tool decorator")
