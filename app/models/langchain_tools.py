from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


class LangChainToolCategory(str, Enum):
    SEARCH = "search"
    CALCULATOR = "calculator"
    FILE = "file"
    WEB = "web"
    DATABASE = "database"
    MEMORY = "memory"
    CODE = "code"
    UTILITY = "utility"


class SearchToolType(str, Enum):
    GOOGLE_SEARCH = "google_search"
    DUCKDUCKGO_SEARCH = "duckduckgo_search"
    WOLFRAM_ALPHA = "wolfram_alpha"
    TAVILY_SEARCH = "tavily_search"


class FileToolType(str, Enum):
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    DELETE_FILE = "delete_file"
    LIST_DIRECTORY = "list_directory"
    COPY_FILE = "copy_file"
    MOVE_FILE = "move_file"


class WebToolType(str, Enum):
    REQUESTS_GET = "requests_get"
    REQUESTS_POST = "requests_post"
    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"


class DatabaseToolType(str, Enum):
    SQL_DATABASE = "sql_database"
    MONGODB = "mongodb"
    VECTOR_STORE = "vector_store"


class CalculatorToolType(str, Enum):
    PYTHON_REPL = "python_repl"
    NUMEXPR = "numexpr"


class LangChainTool(BaseModel):
    """LangChain built-in tool configuration"""
    category: LangChainToolCategory
    tool_name: str
    tool_class: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required_packages: List[str] = Field(default_factory=list)
    configuration: Dict[str, Any] = Field(default_factory=dict)


# Predefined LangChain tools
LANGCHAIN_TOOLS = {
    # Search Tools
    "google_search": LangChainTool(
        category=LangChainToolCategory.SEARCH,
        tool_name="GoogleSearchRun",
        tool_class="langchain_community.tools.google_search.GoogleSearchRun",
        description="Search Google for information",
        parameters={
            "num_results": {"type": "int", "default": 5, "description": "Number of results to return"},
            "country": {"type": "str", "default": "us", "description": "Country code for search"},
            "language": {"type": "str", "default": "en", "description": "Language code"}
        },
        required_packages=["google-search-results"],
        configuration={
            "api_key_required": True,
            "api_key_field": "api_key"
        }
    ),
    "duckduckgo_search": LangChainTool(
        category=LangChainToolCategory.SEARCH,
        tool_name="DuckDuckGoSearchRun",
        tool_class="langchain_community.tools.duckduckgo_search.DuckDuckGoSearchRun",
        description="Search DuckDuckGo for information",
        parameters={
            "num_results": {"type": "int", "default": 5, "description": "Number of results to return"},
            "region": {"type": "str", "default": "us-en", "description": "Region for search"},
            "safesearch": {"type": "str", "default": "moderate", "description": "Safe search level"}
        },
        required_packages=["duckduckgo-search"]
    ),
    "wolfram_alpha": LangChainTool(
        category=LangChainToolCategory.SEARCH,
        tool_name="WolframAlphaQueryRun",
        tool_class="langchain_community.tools.wolfram_alpha.tool.WolframAlphaQueryRun",
        description="Query Wolfram Alpha for computational knowledge",
        parameters={
            "max_results": {"type": "int", "default": 5, "description": "Maximum number of results"}
        },
        required_packages=["wolframalpha"],
        configuration={
            "api_key_required": True,
            "api_key_field": "app_id"
        }
    ),
    
    # Calculator Tools
    "python_repl": LangChainTool(
        category=LangChainToolCategory.CALCULATOR,
        tool_name="PythonREPLTool",
        tool_class="langchain_experimental.tools.python.tool.PythonREPLTool",
        description="Execute Python code and return results",
        parameters={
            "globals": {"type": "dict", "default": {}, "description": "Global variables for the REPL"},
            "locals": {"type": "dict", "default": {}, "description": "Local variables for the REPL"}
        },
        required_packages=["langchain-experimental"]
    ),
    "numexpr": LangChainTool(
        category=LangChainToolCategory.CALCULATOR,
        tool_name="NumExpr",
        tool_class="langchain_experimental.tools.numexpr_tool.NumExpr",
        description="Evaluate mathematical expressions using NumExpr",
        parameters={
            "expression": {"type": "str", "description": "Mathematical expression to evaluate"}
        },
        required_packages=["langchain-experimental", "numexpr"]
    ),
    
    # File Tools
    "read_file": LangChainTool(
        category=LangChainToolCategory.FILE,
        tool_name="ReadFileTool",
        tool_class="langchain_community.tools.file_management.read.ReadFileTool",
        description="Read contents of a file",
        parameters={
            "file_path": {"type": "str", "description": "Path to the file to read"}
        },
        required_packages=["langchain-community"]
    ),
    "write_file": LangChainTool(
        category=LangChainToolCategory.FILE,
        tool_name="WriteFileTool",
        tool_class="langchain_community.tools.file_management.write.WriteFileTool",
        description="Write content to a file",
        parameters={
            "file_path": {"type": "str", "description": "Path to the file to write"},
            "content": {"type": "str", "description": "Content to write to the file"}
        },
        required_packages=["langchain-community"]
    ),
    "list_directory": LangChainTool(
        category=LangChainToolCategory.FILE,
        tool_name="ListDirectoryTool",
        tool_class="langchain_community.tools.file_management.list_dir.ListDirectoryTool",
        description="List contents of a directory",
        parameters={
            "dir_path": {"type": "str", "description": "Path to the directory to list"}
        },
        required_packages=["langchain-community"]
    ),
    
    # Web Tools
    "requests_get": LangChainTool(
        category=LangChainToolCategory.WEB,
        tool_name="RequestsGetTool",
        tool_class="langchain_community.tools.requests.tool.RequestsGetTool",
        description="Make GET request to a URL",
        parameters={
            "url": {"type": "str", "description": "URL to make GET request to"},
            "headers": {"type": "dict", "default": {}, "description": "HTTP headers"},
            "params": {"type": "dict", "default": {}, "description": "URL parameters"}
        },
        required_packages=["langchain-community", "requests"]
    ),
    "requests_post": LangChainTool(
        category=LangChainToolCategory.WEB,
        tool_name="RequestsPostTool",
        tool_class="langchain_community.tools.requests.tool.RequestsPostTool",
        description="Make POST request to a URL",
        parameters={
            "url": {"type": "str", "description": "URL to make POST request to"},
            "headers": {"type": "dict", "default": {}, "description": "HTTP headers"},
            "data": {"type": "dict", "default": {}, "description": "POST data"},
            "json_data": {"type": "dict", "default": {}, "description": "JSON data to send"}
        },
        required_packages=["langchain-community", "requests"]
    ),
    
    # Database Tools
    "sql_database": LangChainTool(
        category=LangChainToolCategory.DATABASE,
        tool_name="QuerySQLDataBaseTool",
        tool_class="langchain_community.tools.sql_database.tool.QuerySQLDataBaseTool",
        description="Query SQL database",
        parameters={
            "query": {"type": "str", "description": "SQL query to execute"}
        },
        required_packages=["langchain-community"],
        configuration={
            "database_connection_required": True,
            "connection_field": "db"
        }
    ),
    
    # Memory Tools
    "memory": LangChainTool(
        category=LangChainToolCategory.MEMORY,
        tool_name="MemoryTool",
        tool_class="langchain.tools.memory.tool.MemoryTool",
        description="Store and retrieve information from memory",
        parameters={
            "key": {"type": "str", "description": "Memory key to store/retrieve"},
            "value": {"type": "str", "description": "Value to store in memory"}
        },
        required_packages=["langchain"]
    )
}


class LangChainToolCreateRequest(BaseModel):
    """Request to create a LangChain tool"""
    tool_name: str
    category: LangChainToolCategory
    configuration: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None
    custom_parameters: Dict[str, Any] = Field(default_factory=dict)


class LangChainToolListResponse(BaseModel):
    """Response for LangChain tool listing"""
    tools: List[LangChainTool]
    categories: List[LangChainToolCategory]
    total_count: int


def get_langchain_tools_by_category(category: LangChainToolCategory) -> List[LangChainTool]:
    """Get LangChain tools by category"""
    return [tool for tool in LANGCHAIN_TOOLS.values() if tool.category == category]


def get_langchain_tool_by_name(tool_name: str) -> Optional[LangChainTool]:
    """Get specific LangChain tool by name"""
    return LANGCHAIN_TOOLS.get(tool_name)


def get_all_langchain_tools() -> List[LangChainTool]:
    """Get all available LangChain tools"""
    return list(LANGCHAIN_TOOLS.values())


def get_langchain_tool_categories() -> List[LangChainToolCategory]:
    """Get all LangChain tool categories"""
    return list(LangChainToolCategory)
