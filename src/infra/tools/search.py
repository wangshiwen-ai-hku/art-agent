import asyncio
import os
from typing import Annotated, List, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from tavily import AsyncTavilyClient

from dotenv import load_dotenv

load_dotenv()

##########################
# Tavily Search Tool Utils
##########################
TAVILY_SEARCH_DESCRIPTION = "一个网络搜索引擎。"

COUNTRY = "china"  # apply this parameter when topic is general
MAX_RESULTS = 5
MAX_CHARS_TO_INCLUDE = 5000


@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    topic: Annotated[Literal["general", "news"], InjectedToolArg] = "general",
) -> str:
    """Fetch and summarize search results from Tavily search API.

    Args:
        queries: List of search queries to execute
    Returns:
        Formatted string containing summarized search results
    """
    # Step 1: Execute search queries asynchronously
    search_results = await tavily_search_async(
        queries,
        max_results=MAX_RESULTS,
        topic=topic,
        include_raw_content=True,
        country=COUNTRY,
    )

    # Step 2: Deduplicate results by URL to avoid processing the same content multiple times
    unique_results = {}
    for response in search_results:
        for result in response["results"]:
            url = result["url"]
            if url not in unique_results:
                unique_results[url] = {**result, "query": response["query"]}

    # Step 3: Process results and prepare for formatting
    final_results = {}

    for url, result in unique_results.items():
        content = result["content"]  # Use original content directly

        final_results[url] = {"title": result["title"], "content": content}

    # Step 4: Format the final output using utility function
    if not final_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    return format_search_results(final_results)


def format_search_results(summarized_results: dict) -> str:
    """Format search results into a string."""
    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- 来源 {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"内容:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"

    return formatted_output


async def tavily_search_async(
    search_queries,
    max_results: int = 5,
    topic: Literal["general", "news"] = "general",
    include_raw_content: bool = True,
    country: str = "china",
):
    """Execute multiple Tavily search queries asynchronously.

    Args:
        search_queries: List of search query strings to execute
        max_results: Maximum number of results per query
        topic: Topic category for filtering results
        include_raw_content: Whether to include full webpage content
        config: Runtime configuration for API key access

    Returns:
        List of search result dictionaries from Tavily API
    """
    # Initialize the Tavily client with API key from config
    tavily_client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    # Create search tasks for parallel execution
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
            country=country,
            exclude_domains=["zh.wikipedia.org", "wikipedia.org", "en.wikipedia.org"],
        )
        for query in search_queries
    ]

    # Execute all search queries in parallel and return results
    search_results = await asyncio.gather(*search_tasks)
    return search_results


async def test_tavily_tools():
    """Test function for tavily tools, similar to metaso tools test"""
    print("=== Testing Tavily Tools ===")

    test_queries = ["牛顿第一定律"]

    # Test basic search with the new signature
    print(f"\n1. Testing tavily_search with queries: {test_queries}")
    try:
        basic_results = await tavily_search.ainvoke(
            {
                "queries": test_queries,
                "topic": "general",
            }
        )
        print(basic_results)
    except Exception as e:
        print(f"Error testing tavily_search: {e}")

    print("\n=== Tavily Tools Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_tavily_tools())
