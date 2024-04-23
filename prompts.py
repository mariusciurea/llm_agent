from llama_index.core.prompts import PromptTemplate

instruction_str = """
    1. Convert the query to executable Python code using Pandas.
    2. The final line of string should be a Python expression that can be called with the `eval()` function.
    3. The code should represent a solution to the query.
    4. PRINT ONLY THE EXPRESSION.
    5. Do note quote the expression."""

new_prompt = PromptTemplate(
    """\
    You are working with a Pandas dataframe in Python.
    The name of the dataframe is `df`.
    This is the result of `print(df.head())`:
    {df_str}
    
    Follow these instructions:
    {instruction_str}
    Query: {query_str}
    
    Expression: """
)


context = """Purpose: The primary role of the agent is to assist users by providing accurate
          information about world population statistics and details about a country."""
