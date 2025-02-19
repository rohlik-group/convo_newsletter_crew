from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class EXAInput(BaseModel):
    """Input schema for EXATool."""
    query: str = Field(..., description="The query to process with EXA tool.")


class EXATool(BaseTool):
    name: str = "EXA Tool"
    description: str = "Analyzes and refines complex queries from raw input data; intended for advanced processing when inputs are ambiguous, require deep contextual analysis, or involve multifaceted instructions."
    args_schema: Type[BaseModel] = EXAInput

    def _run(self, query: str) -> str:
        # Use Exa.ai's API for advanced processing of the query
        import os
        from exa_py import Exa
        exa_api_key = os.getenv("EXA_API_KEY")
        if not exa_api_key:
            return "EXA API key not found."

        exa = Exa(exa_api_key)

        response = exa.search_and_contents(
            query,
            type="neural",
            use_autoprompt=True,
            num_results=3,
            highlights=True
        )

        parsed_result = ''.join([
            f'<Title id={idx}>{result.title}</Title>'
            f'<URL id={idx}>{result.url}</URL>'
            f'<Highlight id={idx}>{"  ,  ".join(result.highlights)}</Highlight>'
            for idx, result in enumerate(response.results)
        ])

        return parsed_result


if __name__ == "__main__":
    tool = EXATool()
    test_query = "Sample query for EXA tool."
    result = tool._run(query=test_query)
    print("Result:", result) 