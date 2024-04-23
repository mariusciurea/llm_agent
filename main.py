from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import instruction_str, new_prompt, context
from notes_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI


load_dotenv()

data = Path(__file__).parent / 'data'
population = data / 'population.csv'

population_df = pd.read_csv(population)
# print(population_df.head())

population_query_engine = PandasQueryEngine(
    df=population_df,
    verbose=True,
    instruction_str=instruction_str
)


# print(dir(population_query_engine))
population_query_engine.update_prompts({"pandas_prompt": new_prompt})

# population_query_engine.query('What is the population of Romania?')

tools = [
    note_engine,
    QueryEngineTool(query_engine=population_query_engine, metadata=ToolMetadata(
        name="population",
        description="this gives information at the world population and demographics",
    ))
]

llm = OpenAI(model="gpt-3.5-turbo-16k-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input('Enter a prompt: (q to quit): ')) != 'q':
    result = agent.query(prompt)
    print(result)
