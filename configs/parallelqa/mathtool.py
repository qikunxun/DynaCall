"""Math tool that interprets prompts and executes python code to do math."""

from __future__ import annotations

import ast
import math
import re
import asyncio
from typing import Any, Dict, List, Optional


class MinMaxTransformer(ast.NodeTransformer):
    
    def __init__(self):
        self.calculation_steps = []
    
    def visit_Call(self, node):
        
        if isinstance(node.func, ast.Name):
            if node.func.id in ['min', 'max']:
                
                arg_values = []
                for arg in node.args:
                    if isinstance(arg, ast.Constant):
                        arg_values.append(arg.value)
                    else:
                        
                        temp_transformer = MinMaxTransformer()
                        temp_ast = temp_transformer.visit(ast.copy_location(ast.Expression(body=arg), arg))
                        temp_ast = ast.fix_missing_locations(temp_ast)
                        temp_code = compile(temp_ast, "<string>", "eval")
                        arg_value = eval(temp_code)
                        arg_values.append(arg_value)
                
                
                if node.func.id == 'min':
                    result = min(arg_values)
                    min_index = arg_values.index(result)
                    min_arg = ast.unparse(node.args[min_index]) if hasattr(ast, 'unparse') else self._unparse_node(node.args[min_index])
                    step = f"{node.func.id}({', '.join(ast.unparse(arg) if hasattr(ast, 'unparse') else self._unparse_node(arg) for arg in node.args)}) = {min_arg} = {result}"
                else:  # max
                    result = max(arg_values)
                    max_index = arg_values.index(result)
                    max_arg = ast.unparse(node.args[max_index]) if hasattr(ast, 'unparse') else self._unparse_node(node.args[max_index])
                    step = f"{node.func.id}({', '.join(ast.unparse(arg) if hasattr(ast, 'unparse') else self._unparse_node(arg) for arg in node.args)}) = {max_arg} = {result}"
                
                self.calculation_steps.append(step)
                return ast.Constant(value=result)
        
        return self.generic_visit(node)
    
    def _unparse_node(self, node):
        
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.BinOp):
            left = self._unparse_node(node.left)
            right = self._unparse_node(node.right)
            op = self._unparse_operator(node.op)
            return f"({left} {op} {right})"
        elif isinstance(node, ast.UnaryOp):
            operand = self._unparse_node(node.operand)
            op = self._unparse_operator(node.op)
            return f"({op}{operand})"
        elif isinstance(node, ast.Call):
            func_name = self._unparse_node(node.func)
            args = ', '.join(self._unparse_node(arg) for arg in node.args)
            return f"{func_name}({args})"
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return str(ast.dump(node))
    
    def _unparse_operator(self, op):
        
        operators = {
            ast.Add: '+',
            ast.Sub: '-', 
            ast.Mult: '*',
            ast.Div: '/',
            ast.FloorDiv: '//',
            ast.Mod: '%',
            ast.Pow: '**',
            ast.USub: '-',
            ast.UAdd: '+'
        }
        return operators.get(type(op), str(op))


def calculate_with_steps(expression: str) -> str:
    
    try:
        
        if 'min(' in expression or 'max(' in expression:
            transformer = MinMaxTransformer()
            
            
            parsed_expression = ast.parse(expression, mode="eval")
            
            
            transformed_ast = transformer.visit(parsed_expression)
            transformed_ast = ast.fix_missing_locations(transformed_ast)
            
            
            compiled_code = compile(transformed_ast, "<string>", "eval")
            result = eval(compiled_code)
            # Important: return only the scalar result so downstream Calculate
            # receives a clean numeric observation, not a step trace string.
            return str(result)
        else:
            
            result = eval(expression)
            return str(result)
            
    except Exception as e:
        return f"Error calculating expression '{expression}': {str(e)}"


def extract_math_expression(description: str) -> str:
    """Extract math expression from description."""
    return description.strip()


def replace_dependencies(expression: str, dependencies: List[str]) -> str:
    """Replace placeholders with actual values."""
    for i, dep_value in enumerate(dependencies, 1):
        placeholder = f"${i}"
        expression = expression.replace(placeholder, str(dep_value))
    return expression


class MathTool:
    """Simple math tool that solves math problems."""
    
    def __init__(self, llm):
        """Initialize with a language model."""
        self.llm = llm
        
        # Define the math prompt template
        self.math_prompt_template = """Translate a math problem into a expression that can be executed using Python's numexpr library. Use the output of running this code to answer the question.
    You MUST follow the following guidelines:
    - Do not use "where(...)" expressions in your code since it is not supported.
    - Do not use "fmax(...)" expression in your code since it is not supported. Use "max(...)" instead.
    - You MUST never introduce a new variable in any mathmatical expression. The mathematical expression MUST ONLY contain numbers and operations. For instance gazelle_max_speed * 0.12 is NEVER allowed and you must find the numerical value of the gazelle's max speed from the given context.

    Begin.

    Question: What is 37593 * 67?
    ```text
    37593 * 67
    ```
    ...numexpr.evaluate("37593 * 67")...
    ```output
    2518731
    ```
    Answer: 2518731

    Question: 37593^(1/5)
    ```text
    37593**(1/5)
    ```
    ...numexpr.evaluate("37593**(1/5)")...
    ```output
    8.222831614237718
    ```
    Answer: 8.222831614237718

    Question: Answer the Question based on the Context.

    Context: Steven can run at the maximum speed up 123km/h.

    Question: What is the speed of Steven in km/h when he was 10% faster than now?
    ```text
    123 * 1.1
    ```
    ...numexpr.evaluate(123 * 1.1)...
    ```output
    135.3
    ```
    Answer: 135.3

    Question: {question}
    """

    def _evaluate_expression(self, expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            expression = expression.strip()
            # Remove any commas in between digits from the expression
            # e.g. "1,000,000, 2,000,000" -> "1000000, 2000000"
            if '("' in expression and '")' in expression:
                index_l = expression.index('("') + 2
                index_r = expression.index('")')
                expression = expression[index_l:index_r]
            expression = re.sub(r"(?<=\d\*\+),(?=\d\*\+)", "", expression)
            
            # Calculate the result
            result = calculate_with_steps(expression)
        except Exception as e:
            msg = (
                f'"{expression}" is not a valid expression, and raised error: {e}.'
                " You must try again with a valid numerical expression"
            )
            print(msg)
            return msg

        # Remove any leading and trailing brackets from the output
        return re.sub(r"^\[|\]$", "", result)

    def _process_llm_result(self, llm_output: str) -> str:
        """Process LLM output to extract answer."""
        llm_output = llm_output.strip()
        
        if '```' in llm_output:
            index = llm_output.index('```')
            llm_output = llm_output[index:]
        
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            try:
                output = self._evaluate_expression(llm_output)
                answer = "Answer: " + output
            except Exception as e:
                raise ValueError(f"unknown format from LLM: {llm_output}")
        
        return answer

    def run(self, question: str, context: List[str] = None) -> str:
        """Run math calculation on a question with optional context."""
        # Format the question with context if provided
        if context:
            if len(context) == 1:
                context_str = f"Context:\n{context[0]}"
            else:
                context_strs = []
                for i, c in enumerate(context):
                    context_strs.append(f"Context {i}:\n{c}")
                context_str = "\n\n".join(context_strs)
            
            full_prompt = (
                "Answer the Question based on the Context. When you write down a expression, it MUST ONLY consists of numbers and operators. "
                "Here are some guidelines that you will be PANALIZED if you don't follow:\n\n"
                "  - When you are asked for differences, you consider the absolute value of the difference. Difference of two numbers is always positive."
                "For instance, the difference between 1 and 2 is 1, not -1.\n"
                "  - When you are applying operations (e.g. difference, summation, ratio, etc.) between multiple values in the Context, you must unify the units of those numbers. "
                "For instance, you cannot add 1 meter to 1 foot.\n"
                "     - You must pick the values in the same units if all the values are available in the same units.\n"
                "     - If not, you must convert them to the same units before applying the operation.\n"
                "  - You MUST strictly follow the unit (e.g. meter, kilometer, million, etc.) you were asked.\n"
                "     - If the Context has the numbers in same units as the question, you can directly use them.\n"
                "     - If the Context has the numbers in different units than the question, you must convert them to the units asked in the question."
                "For example, if the question asks for the distance between two cities in kilometers, but the Context has the distance in miles, "
                "you must convert the distance to kilometers.\n"
                "  - If you are asked about a particular number in millions, billions, or any other unit, the number should be written without specifying the unit. "
                "For example, if you are asked for 100 millions, it should be written as 100, not 100 million or 100,000,000.\n"
                ' - Never introduce a variable. For instance "gazelle_max_speed * 1.4" is not allowed. Pick up a correct number from the given context.\n'
                "\n"
                f"{context_str}\n\n"
                f"Question: {question}\n\n"
            )
            full_prompt = self.math_prompt_template.format(question=full_prompt)
        else:
            full_prompt = self.math_prompt_template.format(question=question)

        llm_response = self.llm.predict(full_prompt, stop=["```output"])
        answer = self._process_llm_result(llm_response)
        
        # Extract just the numerical answer
        if "Answer:" in answer:
            result = answer.split("Answer:")[1].strip()
            try:
                # Try to convert to float and round
                result_num = float(result)
                result_num = round(result_num, 3)
                return str(result_num)
            except:
                # If not a number, return as-is
                return result
        return answer
    
    async def arun(self, question: str, context: List[str] = None) -> str:
        """Async version of run."""
        # 同样修改异步版本
        if context:
            if len(context) == 1:
                context_str = f"Context:\n{context[0]}"
            else:
                context_strs = []
                for i, c in enumerate(context):
                    context_strs.append(f"Context {i}:\n{c}")
                context_str = "\n\n".join(context_strs)
            
            full_prompt = (
                "Answer the Question based on the Context. When you write down a expression, it MUST ONLY consists of numbers and operators. "
                "Here are some guidelines that you will be PANALIZED if you don't follow:\n\n"
                "  - When you are asked for differences, you consider the absolute value of the difference. Difference of two numbers is always positive."
                "For instance, the difference between 1 and 2 is 1, not -1.\n"
                "  - When you are applying operations (e.g. difference, summation, ratio, etc.) between multiple values in the Context, you must unify the units of those numbers. "
                "For instance, you cannot add 1 meter to 1 foot.\n"
                "     - You must pick the values in the same units if all the values are available in the same units.\n"
                "     - If not, you must convert them to the same units before applying the operation.\n"
                "  - You MUST strictly follow the unit (e.g. meter, kilometer, million, etc.) you were asked.\n"
                "     - If the Context has the numbers in same units as the question, you can directly use them.\n"
                "     - If the Context has the numbers in different units than the question, you must convert them to the units asked in the question."
                "For example, if the question asks for the distance between two cities in kilometers, but the Context has the distance in miles, "
                "you must convert the distance to kilometers.\n"
                "  - If you are asked about a particular number in millions, billions, or any other unit, the number should be written without specifying the unit. "
                "For example, if you are asked for 100 millions, it should be written as 100, not 100 million or 100,000,000.\n"
                ' - Never introduce a variable. For instance "gazelle_max_speed * 1.4" is not allowed. Pick up a correct number from the given context.\n'
                "\n"
                f"{context_str}\n\n"
                f"Question: {question}\n\n"
            )
            full_prompt = self.math_prompt_template.format(question=full_prompt)
        else:
            full_prompt = self.math_prompt_template.format(question=question)

        # 修改这里：直接使用 apredict
        if hasattr(self.llm, 'apredict'):
            llm_response = await self.llm.apredict(full_prompt, stop=["```output"])
        else:
            # 回退方案
            llm_response = f"LLM response placeholder for: {full_prompt[:100]}..."
        
        # Process the response
        answer = self._process_llm_result(llm_response)
        
        # Extract just the numerical answer
        if "Answer:" in answer:
            result = answer.split("Answer:")[1].strip()
            try:
                # Try to convert to float and round
                result_num = float(result)
                result_num = round(result_num, 3)
                return str(result_num)
            except:
                # If not a number, return as-is
                return result
        return answer

def create_math_tool(llm):
    """Factory function to create a math tool."""
    math_tool_instance = MathTool(llm)
    def math_function(question: str, context: List[str] = None) -> str:
        """Math tool function interface."""
        return math_tool_instance.run(question, context)

    # Add async version
    async def async_math_function(question: str, context: List[str] = None) -> str:
        """Async math tool function interface."""
        return await math_tool_instance.arun(question, context)

    # Make the function have both sync and async attributes
    math_function.async_func = async_math_function

    return math_function
