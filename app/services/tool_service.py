from typing import Optional, List, Dict, Any
from uuid import uuid4
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging
import httpx
import json

from app.database import get_database
from app.models.tool import (
    Tool, ToolConfig, ToolType, ToolStatus, ToolCreateRequest, ToolUpdateRequest,
    ToolExecutionRequest, ToolExecutionResponse, APIConfig, 
    FunctionConfig, MCPConfig, WebhookConfig, LangChainConfig, CustomConfig
)
from app.models.langchain_tools import (
    LANGCHAIN_TOOLS, LangChainTool, LangChainToolCategory,
    get_langchain_tool_by_name, get_all_langchain_tools
)

logger = logging.getLogger(__name__)


class ToolService:
    """Service for managing tools and their execution"""
    
    def __init__(self):
        self.db: AsyncIOMotorDatabase = get_database()
    
    async def create_tool(self, request: ToolCreateRequest) -> Tool:
        """Create a new tool"""
        tool_id = str(uuid4())
        now = datetime.now()
        
        tool = Tool(
            id=tool_id,
            user_id=request.user_id,
            name=request.name,
            description=request.description,
            type=request.type,
            status=ToolStatus.ACTIVE,
            config=request.config,
            parameters=request.parameters,
            tags=request.tags,
            created_at=now,
            updated_at=now
        )
        
        # Save to database
        await self.db.get_collection("tools").insert_one(tool.model_dump(by_alias=True))
        
        logger.info(f"Created tool {tool_id} for user {request.user_id}")
        return tool
    
    async def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get tool by ID"""
        doc = await self.db.get_collection("tools").find_one({"_id": str(tool_id)})
        if doc:
            doc["id"] = doc.pop("_id")
            return Tool(**doc)
        return None
    
    async def get_tools_by_user(self, user_id: str) -> List[Tool]:
        """Get all tools for a user"""
        cursor = self.db.get_collection("tools").find({"user_id": user_id}).sort("created_at", -1)
        tools = []
        async for doc in cursor:
            doc["id"] = doc.pop("_id")
            tools.append(Tool(**doc))
        return tools
    
    async def update_tool(self, tool_id: str, request: ToolUpdateRequest) -> Optional[Tool]:
        """Update tool"""
        update_data = {}
        
        if request.name is not None:
            update_data["name"] = request.name
        if request.description is not None:
            update_data["description"] = request.description
        if request.type is not None:
            update_data["type"] = request.type
        if request.config is not None:
            update_data["config"] = request.config.model_dump()
        if request.parameters is not None:
            update_data["parameters"] = [p.model_dump() for p in request.parameters]
        if request.tags is not None:
            update_data["tags"] = request.tags
        
        update_data["updated_at"] = datetime.now()
        
        result = await self.db.get_collection("tools").update_one(
            {"_id": str(tool_id)},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            return await self.get_tool(tool_id)
        
        return None
    
    async def delete_tool(self, tool_id: str) -> bool:
        """Delete tool"""
        result = await self.db.get_collection("tools").delete_one({"_id": str(tool_id)})
        
        if result.deleted_count > 0:
            logger.info(f"Deleted tool {tool_id}")
            return True
        
        return False
    
    async def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResponse:
        """Execute a tool"""
        tool = await self.get_tool(request.tool_id)
        if not tool:
            return ToolExecutionResponse(
                success=False,
                result=None,
                error="Tool not found",
                execution_time=0.0
            )
        
        if tool.status != ToolStatus.ACTIVE:
            return ToolExecutionResponse(
                success=False,
                result=None,
                error=f"Tool is not active (status: {tool.status})",
                execution_time=0.0
            )
        
        start_time = datetime.now()
        
        try:
            if tool.type == ToolType.API:
                result = await self._execute_api_tool(tool, request.parameters)
            elif tool.type == ToolType.FUNCTION:
                result = await self._execute_function_tool(tool, request.parameters)
            elif tool.type == ToolType.MCP:
                result = await self._execute_mcp_tool(tool, request.parameters)
            elif tool.type == ToolType.WEBHOOK:
                result = await self._execute_webhook_tool(tool, request.parameters)
            elif tool.type == ToolType.LANGCHAIN:
                result = await self._execute_langchain_tool(tool, request.parameters)
            elif tool.type == ToolType.CUSTOM:
                result = await self._execute_custom_tool(tool, request.parameters)
            else:
                raise ValueError(f"Unsupported tool type: {tool.type}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolExecutionResponse(
                success=True,
                result=result,
                error=None,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Tool execution failed: {e}")
            
            return ToolExecutionResponse(
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _execute_api_tool(self, tool: Tool, parameters: Dict[str, Any]) -> Any:
        """Execute API tool"""
        if not tool.config.api:
            raise ValueError("API config not found")
        
        config = tool.config.api
        
        # Build URL with parameters
        url = config.base_url
        if parameters:
            # Simple parameter injection - you might want more sophisticated logic
            for key, value in parameters.items():
                url = url.replace(f"{{{key}}}", str(value))
        
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.get(
                url,
                headers=config.headers,
                timeout=config.timeout
            )
            response.raise_for_status()
            return response.json()
    
    async def _execute_function_tool(self, tool: Tool, parameters: Dict[str, Any]) -> Any:
        """Execute function tool"""
        if not tool.config.function:
            raise ValueError("Function config not found")
        
        config = tool.config.function
        
        # Create a safe execution environment
        safe_globals = {
            "__builtins__": {
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "reversed": reversed,
                "enumerate": enumerate,
                "zip": zip,
                "range": range,
                "print": print,
            }
        }
        
        # Set environment variables
        for key, value in config.environment_variables.items():
            safe_globals[key] = value
        
        # Execute the function
        try:
            exec(config.code, safe_globals)
            
            # Look for a main function or execute the code directly
            if "main" in safe_globals:
                return safe_globals["main"](parameters)
            else:
                # If no main function, assume the code returns the result
                return safe_globals.get("result", None)
                
        except Exception as e:
            raise ValueError(f"Function execution failed: {e}")
    
    async def _execute_mcp_tool(self, tool: Tool, parameters: Dict[str, Any]) -> Any:
        """Execute MCP tool"""
        if not tool.config.mcp:
            raise ValueError("MCP config not found")
        
        config = tool.config.mcp
        
        # For now, make a simple HTTP request to the MCP server
        # In a real implementation, you'd use the MCP protocol
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config.server_url}/execute",
                json={
                    "capabilities": config.capabilities,
                    "parameters": parameters
                },
                headers=config.authentication or {}
            )
            response.raise_for_status()
            return response.json()
    
    async def _execute_webhook_tool(self, tool: Tool, parameters: Dict[str, Any]) -> Any:
        """Execute webhook tool"""
        if not tool.config.webhook:
            raise ValueError("Webhook config not found")
        
        config = tool.config.webhook
        
        # Prepare payload
        if config.payload_template:
            payload = self._render_template(config.payload_template, parameters)
        else:
            payload = parameters
        
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=config.method,
                url=config.url,
                json=payload,
                headers=config.headers
            )
            response.raise_for_status()
            return response.json()
    
    def _render_template(self, template: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Render template with parameters"""
        # Simple template rendering - you might want to use Jinja2 or similar
        rendered = template
        for key, value in parameters.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))
        
        try:
            return json.loads(rendered)
        except json.JSONDecodeError:
            return {"raw": rendered}
    
    async def _execute_langchain_tool(self, tool: Tool, parameters: Dict[str, Any]) -> Any:
        """Execute LangChain tool"""
        if not tool.config.langchain:
            raise ValueError("LangChain config not found")
        
        config = tool.config.langchain
        
        try:
            # Import the tool class dynamically
            module_path, class_name = config.tool_class.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            tool_class = getattr(module, class_name)
            
            # Create tool instance with configuration
            tool_instance = tool_class(**config.parameters)
            
            # Execute the tool
            if hasattr(tool_instance, 'arun'):
                result = await tool_instance.arun(**parameters)
            else:
                result = tool_instance.run(**parameters)
            
            return result
            
        except Exception as e:
            raise ValueError(f"LangChain tool execution failed: {e}")
    
    async def _execute_custom_tool(self, tool: Tool, parameters: Dict[str, Any]) -> Any:
        """Execute custom tool"""
        if not tool.config.custom:
            raise ValueError("Custom config not found")
        
        config = tool.config.custom
        
        try:
            if config.language == "python":
                # Execute Python code
                safe_globals = {
                    "__builtins__": {
                        "len": len, "str": str, "int": int, "float": float, "bool": bool,
                        "list": list, "dict": dict, "tuple": tuple, "set": set,
                        "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
                        "sorted": sorted, "reversed": reversed, "enumerate": enumerate,
                        "zip": zip, "range": range, "print": print,
                    }
                }
                
                # Set environment variables
                for key, value in config.environment_variables.items():
                    safe_globals[key] = value
                
                # Set parameters as globals
                safe_globals.update(parameters)
                
                # Execute the custom code
                exec(config.implementation, safe_globals)
                
                # Return result if available
                return safe_globals.get("result", "Custom tool executed successfully")
            
            else:
                raise ValueError(f"Unsupported custom tool language: {config.language}")
                
        except Exception as e:
            raise ValueError(f"Custom tool execution failed: {e}")
    
    async def validate_tool_config(self, tool_type: ToolType, config: Any) -> Dict[str, Any]:
        """Validate tool configuration"""
        errors = []
        warnings = []
        
        if tool_type == ToolType.API:
            if not isinstance(config, APIConfig):
                errors.append("Invalid API config type")
            else:
                # Validate URL
                if not config.base_url:
                    errors.append("Base URL is required")
                
                # Test connection (optional)
                try:
                    async with httpx.AsyncClient(timeout=5) as client:
                        await client.head(config.base_url)
                except Exception:
                    warnings.append("Could not connect to API endpoint")
        
        elif tool_type == ToolType.FUNCTION:
            if not isinstance(config, FunctionConfig):
                errors.append("Invalid function config type")
            else:
                # Validate code syntax
                try:
                    compile(config.code, "<string>", "exec")
                except SyntaxError as e:
                    errors.append(f"Invalid Python syntax: {e}")
        
        elif tool_type == ToolType.MCP:
            if not isinstance(config, MCPConfig):
                errors.append("Invalid MCP config type")
            else:
                if not config.server_url:
                    errors.append("Server URL is required")
        
        elif tool_type == ToolType.WEBHOOK:
            if not isinstance(config, WebhookConfig):
                errors.append("Invalid webhook config type")
            else:
                if not config.url:
                    errors.append("Webhook URL is required")
        
        return {
            "errors": errors,
            "warnings": warnings,
            "valid": len(errors) == 0
        }
    
    async def get_langchain_tools(self, category: Optional[LangChainToolCategory] = None) -> List[LangChainTool]:
        """Get available LangChain tools"""
        if category:
            return [tool for tool in get_all_langchain_tools() if tool.category == category]
        return get_all_langchain_tools()
    
    async def get_langchain_tool(self, tool_name: str) -> Optional[LangChainTool]:
        """Get specific LangChain tool"""
        return get_langchain_tool_by_name(tool_name)
    
    async def create_langchain_tool(self, tool_name: str, user_id: str, configuration: Dict[str, Any] = None) -> Tool:
        """Create a tool from LangChain tool definition"""
        langchain_tool = get_langchain_tool_by_name(tool_name)
        if not langchain_tool:
            raise ValueError(f"LangChain tool '{tool_name}' not found")
        
        # Create tool configuration
        tool_config = {
            "langchain": {
                "tool_name": langchain_tool.tool_name,
                "tool_class": langchain_tool.tool_class,
                "parameters": configuration or {}
            }
        }
        
        # Create tool request
        request = ToolCreateRequest(
            name=langchain_tool.description,
            description=f"LangChain {langchain_tool.category.value} tool: {langchain_tool.tool_name}",
            type=ToolType.LANGCHAIN,
            config=ToolConfig(**tool_config),
            parameters=[
                {
                    "name": param_name,
                    "type": param_info.get("type", "string"),
                    "description": param_info.get("description", ""),
                    "required": param_name in langchain_tool.parameters
                }
                for param_name, param_info in langchain_tool.parameters.items()
            ],
            tags=["langchain", langchain_tool.category.value],
            user_id=user_id
        )
        
        return await self.create_tool(request)
