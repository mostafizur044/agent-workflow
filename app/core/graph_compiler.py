from typing import Dict, Any, List, Optional, Callable
import logging
import asyncio
from datetime import datetime
from enum import Enum
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseLanguageModel

from app.models.agent import Agent
from app.models.model import ModelProvider
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.tool_service import ToolService
from app.services.model_service import ModelService
from app.services.agent_service import AgentService
from app.core.cache import get_cache_manager
from config import settings

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Types of nodes in the agent graph"""
    ENTRY = "entry"
    LLM = "llm"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    TOOL_EXECUTION = "tool_execution"
    MCP_INTEGRATION = "mcp_integration"
    SUB_AGENT_ROUTER = "sub_agent_router"
    CONDITIONAL_ROUTER = "conditional_router"
    PARALLEL_PROCESSOR = "parallel_processor"
    MEMORY_MANAGER = "memory_manager"
    OUTPUT = "output"
    ERROR_HANDLER = "error_handler"


class FlowType(str, Enum):
    """Types of execution flows"""
    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    LOOP = "loop"
    BRANCH = "branch"


class ExecutionContext:
    """Context for graph execution"""
    def __init__(self, agent_id: str, session_id: str, user_id: str):
        self.agent_id = agent_id
        self.session_id = session_id
        self.user_id = user_id
        self.start_time = datetime.now()
        self.execution_stack = []  # For tracking nested agent calls
        self.max_depth = 5  # Prevent infinite recursion
        self.parallel_tasks = {}  # For parallel execution tracking


class GraphCompiler:
    """Compiler for building dynamic LangGraph based on agent configuration"""
    
    def __init__(self):
        self.kb_service = KnowledgeBaseService()
        self.tool_service = ToolService()
        self.model_service = ModelService()
        self.agent_service = AgentService()
        self.checkpointer = MemorySaver()
    
    async def get_or_compile_graph(self, agent: Agent, context: Optional[ExecutionContext] = None):
        """Get cached graph or compile new one with execution context"""
        # Check cache first
        cache_key = f"{agent.id}_{agent.version}"
        if context:
            cache_key += f"_{context.session_id}"
            
        cached_graph = await get_cache_manager().get_agent_graph(str(agent.id), agent.version)
        if cached_graph:
            logger.info(f"Using cached graph for agent {agent.id}")
            return cached_graph
        
        # Compile new graph
        return await self.compile_agent_graph(agent, context)
    
    async def compile_agent_graph(self, agent: Agent, context: Optional[ExecutionContext] = None):
        """Compile agent configuration into LangGraph with advanced features"""
        logger.info(f"Compiling graph for agent {agent.id}")
        
        try:
            # Create state schema based on enabled components
            state_schema = self._create_enhanced_state_schema(agent)
            
            # Create graph with checkpointer for state management
            graph = StateGraph(state_schema, checkpointer=self.checkpointer)
            
            # Build execution plan
            execution_plan = await self._build_execution_plan(agent, context)
            
            # Add nodes based on execution plan
            await self._add_enhanced_nodes(graph, agent, execution_plan)
            
            # Add edges based on execution plan
            await self._add_enhanced_edges(graph, agent, execution_plan)
            
            # Compile the graph with advanced configuration
            compiled_graph = graph.compile()
            
            # Cache the compiled graph
            await get_cache_manager().set_agent_graph(str(agent.id), agent.version, compiled_graph)
            
            logger.info(f"Successfully compiled graph for agent {agent.id}")
            return compiled_graph
            
        except Exception as e:
            logger.error(f"Failed to compile graph for agent {agent.id}: {e}")
            raise
    
    async def _build_execution_plan(self, agent: Agent, context: Optional[ExecutionContext]) -> Dict[str, Any]:
        """Build execution plan based on agent configuration and context"""
        plan = {
            "flow_type": FlowType.SEQUENTIAL,
            "nodes": [],
            "edges": [],
            "parallel_groups": [],
            "conditional_routes": [],
            "error_handling": {},
            "memory_strategy": "buffer" if agent.config.memory.enabled else None
        }
        
        # Determine flow type based on configuration
        if agent.config.sub_agents.enabled and len(agent.config.sub_agents.items) > 1:
            plan["flow_type"] = FlowType.BRANCH
        elif agent.config.tools.enabled and len(agent.config.tools.items) > 3:
            plan["flow_type"] = FlowType.PARALLEL
        
        # Build node sequence
        node_sequence = []
        
        # Always start with entry
        node_sequence.append({
            "type": NodeType.ENTRY,
            "id": "entry",
            "config": {}
        })
        
        # Add LLM if enabled
        if agent.config.llm.enabled:
            node_sequence.append({
                "type": NodeType.LLM,
                "id": "llm",
                "config": agent.config.llm.model_dump()
            })
        
        # Add knowledge retrieval
        if agent.config.knowledge_bases.enabled:
            node_sequence.append({
                "type": NodeType.KNOWLEDGE_RETRIEVAL,
                "id": "knowledge_retrieval",
                "config": agent.config.knowledge_bases.model_dump()
            })
        
        # Add tools (can be parallel or sequential)
        if agent.config.tools.enabled:
            if plan["flow_type"] == FlowType.PARALLEL:
                # Group tools for parallel execution
                parallel_group = []
                for tool_item in agent.config.tools.items:
                    if tool_item.enabled:
                        parallel_group.append({
                            "type": NodeType.TOOL_EXECUTION,
                            "id": f"tool_{tool_item.id}",
                            "config": tool_item.model_dump()
                        })
                plan["parallel_groups"].append(parallel_group)
            else:
                # Sequential tool execution
                node_sequence.append({
                    "type": NodeType.TOOL_EXECUTION,
                    "id": "tool_execution",
                    "config": agent.config.tools.model_dump()
                })
        
        # Add sub-agent router
        if agent.config.sub_agents.enabled:
            node_sequence.append({
                "type": NodeType.SUB_AGENT_ROUTER,
                "id": "sub_agent_router",
                "config": agent.config.sub_agents.model_dump()
            })
        
        # Add memory management
        if agent.config.memory.enabled:
            node_sequence.append({
                "type": NodeType.MEMORY_MANAGER,
                "id": "memory_manager",
                "config": agent.config.memory.model_dump()
            })
        
        # Always end with output
        node_sequence.append({
            "type": NodeType.OUTPUT,
            "id": "output",
            "config": {}
        })
        
        # Add error handler
        node_sequence.append({
            "type": NodeType.ERROR_HANDLER,
            "id": "error_handler",
            "config": {}
        })
        
        plan["nodes"] = node_sequence
        return plan
    
    def _create_enhanced_state_schema(self, agent: Agent) -> Dict[str, Any]:
        """Create enhanced state schema with comprehensive fields"""
        schema = {
            # Core message handling
            "messages": list,
            "current_step": str,
            "metadata": dict,
            
            # Execution context
            "execution_context": dict,
            "session_id": str,
            "user_id": str,
            "agent_id": str,
            
            # Knowledge base integration
            "retrieved_context": dict,
            "knowledge_scores": list,
            "retrieval_metadata": dict,
            
            # Tool execution
            "tool_calls": list,
            "tool_results": list,
            "tool_errors": list,
            "parallel_tool_results": dict,
            
            # Sub-agent communication
            "sub_agent_responses": list,
            "sub_agent_calls": list,
            "agent_communication": dict,
            
            # Memory management
            "conversation_history": list,
            "memory_buffer": list,
            "summarized_context": str,
            
            # Error handling and recovery
            "errors": list,
            "error_context": dict,
            "retry_count": int,
            "fallback_responses": list,
            
            # Advanced features
            "conditional_routes": list,
            "parallel_execution": dict,
            "execution_trace": list,
            "performance_metrics": dict,
            
            # A2A communication
            "incoming_messages": list,
            "outgoing_messages": list,
            "communication_protocol": str,
            "agent_network": dict,
            
            # State management
            "checkpoint_data": dict,
            "rollback_points": list,
            "state_version": int
        }
        
        return schema
    
    async def _add_enhanced_nodes(self, graph: StateGraph, agent: Agent, execution_plan: Dict[str, Any]):
        """Add enhanced nodes to graph based on execution plan"""
        
        # Add nodes from execution plan
        for node_config in execution_plan["nodes"]:
            node_type = node_config["type"]
            node_id = node_config["id"]
            config = node_config["config"]
            
            if node_type == NodeType.ENTRY:
                graph.add_node(node_id, self._create_enhanced_entry_node())
            elif node_type == NodeType.LLM:
                graph.add_node(node_id, self._create_enhanced_llm_node(config, agent))
            elif node_type == NodeType.KNOWLEDGE_RETRIEVAL:
                graph.add_node(node_id, self._create_enhanced_knowledge_retrieval_node(config, agent))
            elif node_type == NodeType.TOOL_EXECUTION:
                graph.add_node(node_id, self._create_enhanced_tool_execution_node(config, agent))
            elif node_type == NodeType.SUB_AGENT_ROUTER:
                graph.add_node(node_id, self._create_enhanced_sub_agent_router_node(config, agent))
            elif node_type == NodeType.MEMORY_MANAGER:
                graph.add_node(node_id, self._create_enhanced_memory_manager_node(config, agent))
            elif node_type == NodeType.OUTPUT:
                graph.add_node(node_id, self._create_enhanced_output_node())
            elif node_type == NodeType.ERROR_HANDLER:
                graph.add_node(node_id, self._create_enhanced_error_handler_node())
            elif node_type == NodeType.CONDITIONAL_ROUTER:
                graph.add_node(node_id, self._create_conditional_router_node(config))
            elif node_type == NodeType.PARALLEL_PROCESSOR:
                graph.add_node(node_id, self._create_parallel_processor_node(config, agent))
        
        # Add parallel processing nodes if needed
        for parallel_group in execution_plan.get("parallel_groups", []):
            for node_config in parallel_group:
                node_id = node_config["id"]
                config = node_config["config"]
                graph.add_node(node_id, self._create_parallel_tool_node(config, agent))
    
    async def _add_enhanced_edges(self, graph: StateGraph, agent: Agent, execution_plan: Dict[str, Any]):
        """Add enhanced edges based on execution plan"""
        
        # Set entry point
        graph.set_entry_point("entry")
        
        flow_type = execution_plan["flow_type"]
        nodes = execution_plan["nodes"]
        
        if flow_type == FlowType.SEQUENTIAL:
            await self._add_sequential_edges(graph, nodes)
        elif flow_type == FlowType.PARALLEL:
            await self._add_parallel_edges(graph, execution_plan)
        elif flow_type == FlowType.BRANCH:
            await self._add_branch_edges(graph, nodes)
        elif flow_type == FlowType.CONDITIONAL:
            await self._add_conditional_edges(graph, nodes, execution_plan.get("conditional_routes", []))
        
        # Always add error handling edges
        await self._add_error_handling_edges(graph, nodes)
        
        # End with output
        graph.add_edge("output", END)
    
    async def _add_sequential_edges(self, graph: StateGraph, nodes: List[Dict[str, Any]]):
        """Add sequential edges"""
        for i in range(len(nodes) - 1):
            current_node = nodes[i]["id"]
            next_node = nodes[i + 1]["id"]
            graph.add_edge(current_node, next_node)
    
    async def _add_parallel_edges(self, graph: StateGraph, execution_plan: Dict[str, Any]):
        """Add parallel execution edges"""
        nodes = execution_plan["nodes"]
        parallel_groups = execution_plan.get("parallel_groups", [])
        
        # Find the node before parallel execution
        parallel_start_node = None
        for i, node in enumerate(nodes):
            if node["type"] == NodeType.TOOL_EXECUTION:
                parallel_start_node = nodes[i - 1]["id"] if i > 0 else "entry"
                break
        
        if parallel_start_node and parallel_groups:
            # Connect to parallel tool nodes
            parallel_tool_ids = []
            for parallel_group in parallel_groups:
                for tool_config in parallel_group:
                    parallel_tool_ids.append(tool_config["id"])
                    graph.add_edge(parallel_start_node, tool_config["id"])
            
            # Connect parallel tools to next sequential node
            next_node = None
            for node in nodes:
                if node["type"] in [NodeType.SUB_AGENT_ROUTER, NodeType.OUTPUT]:
                    next_node = node["id"]
                    break
            
            if next_node:
                for tool_id in parallel_tool_ids:
                    graph.add_edge(tool_id, next_node)
    
    async def _add_branch_edges(self, graph: StateGraph, nodes: List[Dict[str, Any]]):
        """Add branching edges for sub-agent routing"""
        # Similar to sequential but with conditional routing for sub-agents
        await self._add_sequential_edges(graph, nodes)
        
        # Add conditional routing for sub-agent router
        sub_agent_router_node = None
        for node in nodes:
            if node["type"] == NodeType.SUB_AGENT_ROUTER:
                sub_agent_router_node = node["id"]
                break
        
        if sub_agent_router_node:
            # Add conditional edges based on sub-agent triggers
            graph.add_conditional_edges(
                sub_agent_router_node,
                self._route_sub_agents,
                {
                    "continue": "output",
                    "error": "error_handler"
                }
            )
    
    async def _add_conditional_edges(self, graph: StateGraph, nodes: List[Dict[str, Any]], conditional_routes: List[Dict[str, Any]]):
        """Add conditional routing edges"""
        await self._add_sequential_edges(graph, nodes)
        
        # Add conditional routes
        for route in conditional_routes:
            source_node = route["source"]
            conditions = route["conditions"]
            
            graph.add_conditional_edges(
                source_node,
                self._create_conditional_router(conditions),
                route["destinations"]
            )
    
    async def _add_error_handling_edges(self, graph: StateGraph, nodes: List[Dict[str, Any]]):
        """Add error handling edges to all nodes"""
        for node in nodes:
            if node["type"] != NodeType.ERROR_HANDLER:
                graph.add_edge(node["id"], "error_handler")
    
    def _create_enhanced_entry_node(self) -> Callable:
        """Create enhanced entry point node"""
        async def entry_node(state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info("Executing enhanced entry node")
            
            # Initialize execution context
            execution_context = {
                "start_time": datetime.now().isoformat(),
                "node_sequence": ["entry"],
                "checkpoint_created": True
            }
            
            # Initialize performance metrics
            performance_metrics = {
                "node_start_times": {"entry": datetime.now().timestamp()},
                "execution_depth": 0
            }
            
            # Initialize A2A communication
            agent_communication = {
                "protocol_version": "1.0",
                "message_queue": [],
                "communication_handlers": {}
            }
            
            return {
                "current_step": "entry",
                "execution_context": execution_context,
                "performance_metrics": performance_metrics,
                "agent_communication": agent_communication,
                "state_version": 1,
                "checkpoint_data": {"entry_completed": True}
            }
        return entry_node
    
    def _create_enhanced_llm_node(self, llm_config: Dict[str, Any], agent: Agent) -> Callable:
        """Create enhanced LLM node with multi-provider support and A2A communication"""
        async def llm_node(state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info("Executing enhanced LLM node")
            
            try:
                # Get model configuration
                model = await self.model_service.get_model(str(llm_config.get("model_id", "")))
                if not model:
                    # Fallback to OpenAI
                    llm = ChatOpenAI(
                        model=llm_config.get("model", "gpt-3.5-turbo"),
                        temperature=llm_config.get("temperature", 0.7),
                        max_tokens=llm_config.get("max_tokens", 1000),
                        streaming=llm_config.get("streaming", False),
                        openai_api_key=settings.openai_api_key
                    )
                else:
                    # Use configured model
                    llm = await self._create_model_client(model)
                
                # Prepare enhanced messages with A2A communication
                messages = await self._prepare_enhanced_messages(state, llm_config, agent)
                
                # Handle incoming A2A messages
                if state.get("incoming_messages"):
                    for msg in state["incoming_messages"]:
                        messages.append(SystemMessage(content=f"Message from {msg['sender']}: {msg['content']}"))
                
                # Get LLM response with error handling
                start_time = datetime.now()
                response = await llm.ainvoke(messages)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Update state with enhanced information
                new_messages = state.get("messages", [])
                new_messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": execution_time,
                    "model_used": model.model_name if model else llm_config.get("model", "unknown")
                })
                
                # Update performance metrics
                performance_metrics = state.get("performance_metrics", {})
                performance_metrics["llm_execution_time"] = execution_time
                performance_metrics["llm_token_count"] = getattr(response, 'usage', {}).get('total_tokens', 0)
                
                # Prepare outgoing A2A messages if needed
                outgoing_messages = []
                if llm_config.get("enable_a2a_communication", False):
                    outgoing_messages = await self._prepare_outgoing_a2a_messages(response.content, state)
                
                return {
                    "messages": new_messages,
                    "current_step": "llm",
                    "performance_metrics": performance_metrics,
                    "outgoing_messages": outgoing_messages,
                    "execution_trace": state.get("execution_trace", []) + [{
                        "step": "llm",
                        "timestamp": datetime.now().isoformat(),
                        "execution_time": execution_time,
                        "success": True
                    }]
                }
                
            except Exception as e:
                logger.error(f"LLM node execution failed: {e}")
                return await self._handle_llm_error(state, e, llm_config)
        
        return llm_node
    
    async def _create_model_client(self, model) -> BaseLanguageModel:
        """Create model client based on provider"""
        try:
            if model.provider == ModelProvider.OPENAI:
                return ChatOpenAI(
                    model=model.model_name,
                    temperature=model.config.get("temperature", 0.7),
                    max_tokens=model.config.get("max_tokens", 1000),
                    api_key=model.config.get("api_key", settings.openai_api_key)
                )
            # elif model.provider == ModelProvider.ANTHROPIC:
            #     # Import and create Anthropic client
            #     return ChatAnthropic(
            #         model=model.model_name,
            #         temperature=model.config.get("temperature", 0.7),
            #         max_tokens=model.config.get("max_tokens", 1000),
            #         api_key=model.config.get("api_key")
            #     )
            # Add more providers as needed
            else:
                # Fallback to OpenAI
                return ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.7,
                    max_tokens=1000,
                    api_key=settings.openai_api_key
                )
        except Exception as e:
            logger.error(f"Failed to create model client: {e}")
            # Fallback to OpenAI
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000,
                api_key=settings.openai_api_key
            )
    
    async def _prepare_enhanced_messages(self, state: Dict[str, Any], llm_config: Dict[str, Any], agent: Agent) -> List:
        """Prepare enhanced messages with context and A2A communication"""
        messages = []
        
        # Add system prompt with agent context
        system_prompt = llm_config.get("system_prompt", "")
        if agent.description:
            system_prompt += f"\n\nAgent Description: {agent.description}"
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        # Add conversation history if memory is enabled
        if state.get("conversation_history"):
            for msg in state["conversation_history"]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
        # Add retrieved context if available
        if state.get("retrieved_context"):
            context_parts = []
            for kb_name, docs in state["retrieved_context"].items():
                context_parts.append(f"Knowledge Base '{kb_name}':")
                for doc in docs:
                    context_parts.append(f"- {doc['content'][:200]}...")
            if context_parts:
                context_msg = "Retrieved Context:\n" + "\n".join(context_parts)
                messages.append(SystemMessage(content=context_msg))
        
        # Add tool results if available
        if state.get("tool_results"):
            tool_results = []
            for result in state["tool_results"]:
                if result["success"]:
                    tool_results.append(f"Tool '{result['tool_name']}': {str(result['result'])[:200]}...")
            if tool_results:
                tool_msg = "Tool Results:\n" + "\n".join(tool_results)
                messages.append(SystemMessage(content=tool_msg))
        
        # Add sub-agent responses if available
        if state.get("sub_agent_responses"):
            sub_agent_responses = []
            for response in state["sub_agent_responses"]:
                if response["triggered"] and response.get("response"):
                    sub_agent_responses.append(f"Sub-agent '{response['sub_agent_name']}': {response['response']}")
            if sub_agent_responses:
                sub_agent_msg = "Sub-agent Responses:\n" + "\n".join(sub_agent_responses)
                messages.append(SystemMessage(content=sub_agent_msg))
        
        # Add current user message
        if state.get("messages"):
            last_message = state["messages"][-1]
            messages.append(HumanMessage(content=last_message["content"]))
        
        return messages
    
    async def _prepare_outgoing_a2a_messages(self, llm_response: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare outgoing A2A messages based on LLM response"""
        outgoing_messages = []
        
        # Simple logic to detect when to send A2A messages
        # In a real implementation, this would be more sophisticated
        if "forward to" in llm_response.lower() or "notify" in llm_response.lower():
            # Extract potential recipient and message
            outgoing_messages.append({
                "recipient": "agent_network",  # Would be determined by context
                "content": llm_response,
                "message_type": "information",
                "priority": "normal",
                "timestamp": datetime.now().isoformat()
            })
        
        return outgoing_messages
    
    async def _handle_llm_error(self, state: Dict[str, Any], error: Exception, llm_config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LLM execution errors"""
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "llm_config": llm_config,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "current_step": "llm",
            "errors": state.get("errors", []) + [error_context],
            "fallback_responses": state.get("fallback_responses", []) + [
                "I apologize, but I encountered an error while processing your request. Please try again."
            ],
            "execution_trace": state.get("execution_trace", []) + [{
                "step": "llm",
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(error)
            }]
        }
    
    def _create_enhanced_knowledge_retrieval_node(self, config: Dict[str, Any], agent: Agent) -> Callable:
        """Create enhanced knowledge retrieval node with advanced features"""
        async def knowledge_retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info("Executing enhanced knowledge retrieval node")
            
            retrieved_context = {}
            knowledge_scores = []
            retrieval_metadata = {}
            
            # Process each enabled knowledge base
            for kb_item in config.get("items", []):
                if not kb_item.get("enabled", True):
                    continue
                
                try:
                    # Get knowledge base
                    kb = await self.kb_service.get_knowledge_base(kb_item["id"])
                    if not kb:
                        continue
                    
                    # Get last user message for search
                    query = ""
                    if state.get("messages"):
                        last_message = state["messages"][-1]
                        query = last_message.get("content", "")
                    
                    # Enhanced search with reranking
                    search_request = {
                        "query": query,
                        "top_k": kb_item.get("retrieval_config", {}).get("top_k", 5),
                        "score_threshold": kb_item.get("retrieval_config", {}).get("score_threshold", 0.7),
                        "rerank_config": kb_item.get("retrieval_config", {}).get("rerank_config"),
                        "hybrid_search": kb_item.get("retrieval_config", {}).get("hybrid_search", False)
                    }
                    
                    start_time = datetime.now()
                    results = await self.kb_service.search_knowledge_base(kb.id, search_request)
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    # Enhanced context with metadata
                    retrieved_context[kb.name] = [
                        {
                            "content": doc.content,
                            "score": score,
                            "metadata": doc.metadata.model_dump(),
                            "retrieval_time": execution_time
                        }
                        for doc, score in zip(results.documents, results.scores)
                    ]
                    
                    # Track scores for analysis
                    knowledge_scores.extend(results.scores)
                    
                    # Store retrieval metadata
                    retrieval_metadata[kb.name] = {
                        "query": query,
                        "execution_time": execution_time,
                        "documents_found": len(results.documents),
                        "average_score": sum(results.scores) / len(results.scores) if results.scores else 0,
                        "embedding_model": kb.embedding_model.model_name,
                        "chunking_method": kb.chunking_config.method
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to retrieve from knowledge base {kb_item['id']}: {e}")
                    retrieval_metadata[kb_item.get("name", "unknown")] = {
                        "error": str(e),
                        "execution_time": 0,
                        "documents_found": 0
                    }
                    continue
            
            return {
                "retrieved_context": retrieved_context,
                "knowledge_scores": knowledge_scores,
                "retrieval_metadata": retrieval_metadata,
                "current_step": "knowledge_retrieval",
                "execution_trace": state.get("execution_trace", []) + [{
                    "step": "knowledge_retrieval",
                    "timestamp": datetime.now().isoformat(),
                    "success": True,
                    "knowledge_bases_searched": len(retrieved_context)
                }]
            }
        
        return knowledge_retrieval_node
    
    def _create_enhanced_tool_execution_node(self, config: Dict[str, Any], agent: Agent) -> Callable:
        """Create enhanced tool execution node with parallel processing support"""
        async def tool_execution_node(state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info("Executing enhanced tool execution node")
            
            tool_calls = []
            tool_results = []
            tool_errors = []
            parallel_tool_results = {}
            
            # Process each enabled tool
            for tool_item in config.get("items", []):
                if not tool_item.get("enabled", True):
                    continue
                
                try:
                    # Execute tool
                    execution_request = {
                        "tool_id": tool_item["id"],
                        "parameters": tool_item.get("config", {})
                    }
                    
                    start_time = datetime.now()
                    result = await self.tool_service.execute_tool(execution_request)
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    tool_calls.append({
                        "tool_id": str(tool_item["id"]),
                        "tool_name": tool_item["name"],
                        "parameters": execution_request["parameters"],
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    tool_results.append({
                        "tool_id": str(tool_item["id"]),
                        "tool_name": tool_item["name"],
                        "result": result.result,
                        "success": result.success,
                        "execution_time": execution_time,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Store parallel results for coordination
                    parallel_tool_results[str(tool_item["id"])] = {
                        "result": result.result,
                        "success": result.success,
                        "execution_time": execution_time
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to execute tool {tool_item['id']}: {e}")
                    tool_errors.append({
                        "tool_id": str(tool_item["id"]),
                        "tool_name": tool_item["name"],
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
            
            return {
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "tool_errors": tool_errors,
                "parallel_tool_results": parallel_tool_results,
                "current_step": "tool_execution",
                "execution_trace": state.get("execution_trace", []) + [{
                    "step": "tool_execution",
                    "timestamp": datetime.now().isoformat(),
                    "success": len(tool_errors) == 0,
                    "tools_executed": len(tool_results)
                }]
            }
        
        return tool_execution_node
    
    def _create_enhanced_sub_agent_router_node(self, config: Dict[str, Any], agent: Agent) -> Callable:
        """Create enhanced sub-agent router with A2A communication"""
        async def sub_agent_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info("Executing enhanced sub-agent router node")
            
            sub_agent_responses = []
            sub_agent_calls = []
            
            # Process each enabled sub-agent
            for sub_agent_item in config.get("items", []):
                if not sub_agent_item.get("enabled", True):
                    continue
                
                # Evaluate trigger condition
                if self._evaluate_enhanced_trigger_condition(sub_agent_item.get("trigger_condition"), state):
                    try:
                        # Prepare A2A communication
                        a2a_message = {
                            "sender": str(agent.id),
                            "recipient": str(sub_agent_item["id"]),
                            "message_type": "task_delegation",
                            "content": state.get("messages", [{}])[-1].get("content", ""),
                            "context": state.get("retrieved_context", {}),
                            "priority": "normal",
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        sub_agent_calls.append(a2a_message)
                        
                        # Execute sub-agent with enhanced context
                        sub_agent = await self.agent_service.get_agent(sub_agent_item["id"])
                        if sub_agent:
                            # Create execution context for sub-agent
                            sub_context = ExecutionContext(
                                agent_id=sub_agent.id,
                                session_id=state.get("session_id", "default"),
                                user_id=state.get("user_id", "default")
                            )
                            
                            # Compile and execute sub-agent graph
                            sub_graph = await self.get_or_compile_graph(sub_agent, sub_context)
                            
                            # Prepare state for sub-agent
                            sub_state = {
                                **state,
                                "incoming_messages": [a2a_message],
                                "agent_id": str(sub_agent.id),
                                "execution_context": {
                                    **state.get("execution_context", {}),
                                    "parent_agent_id": str(agent.id),
                                    "execution_depth": state.get("execution_context", {}).get("execution_depth", 0) + 1
                                }
                            }
                            
                            sub_result = await sub_graph.ainvoke(sub_state)
                            
                            sub_agent_responses.append({
                                "sub_agent_id": str(sub_agent_item["id"]),
                                "sub_agent_name": sub_agent_item["name"],
                                "response": sub_result,
                                "triggered": True,
                                "execution_time": (datetime.now() - datetime.fromisoformat(a2a_message["timestamp"])).total_seconds(),
                                "a2a_message": a2a_message
                            })
                        
                    except Exception as e:
                        logger.error(f"Failed to execute sub-agent {sub_agent_item['id']}: {e}")
                        sub_agent_responses.append({
                            "sub_agent_id": str(sub_agent_item["id"]),
                            "sub_agent_name": sub_agent_item["name"],
                            "response": None,
                            "error": str(e),
                            "triggered": True
                        })
                else:
                    sub_agent_responses.append({
                        "sub_agent_id": str(sub_agent_item["id"]),
                        "sub_agent_name": sub_agent_item["name"],
                        "triggered": False
                    })
            
            return {
                "sub_agent_responses": sub_agent_responses,
                "sub_agent_calls": sub_agent_calls,
                "agent_communication": {
                    **state.get("agent_communication", {}),
                    "outgoing_messages": state.get("outgoing_messages", []) + sub_agent_calls
                },
                "current_step": "sub_agent_router",
                "execution_trace": state.get("execution_trace", []) + [{
                    "step": "sub_agent_router",
                    "timestamp": datetime.now().isoformat(),
                    "success": True,
                    "sub_agents_triggered": len([r for r in sub_agent_responses if r["triggered"]])
                }]
            }
        
        return sub_agent_router_node
    
    def _create_enhanced_output_node(self) -> Callable:
        """Create enhanced output node with comprehensive response formatting"""
        async def output_node(state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info("Executing enhanced output node")
            
            # Format final response with all context
            output = {
                "messages": state.get("messages", []),
                "current_step": "output",
                "metadata": state.get("metadata", {}),
                "execution_context": state.get("execution_context", {}),
                "performance_metrics": state.get("performance_metrics", {}),
                "execution_trace": state.get("execution_trace", []),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add knowledge retrieval results
            if state.get("retrieved_context"):
                output["retrieved_context"] = state["retrieved_context"]
                output["knowledge_scores"] = state.get("knowledge_scores", [])
                output["retrieval_metadata"] = state.get("retrieval_metadata", {})
            
            # Add tool execution results
            if state.get("tool_results"):
                output["tool_results"] = state["tool_results"]
                output["tool_calls"] = state.get("tool_calls", [])
                output["parallel_tool_results"] = state.get("parallel_tool_results", {})
            
            # Add sub-agent responses
            if state.get("sub_agent_responses"):
                output["sub_agent_responses"] = state["sub_agent_responses"]
                output["sub_agent_calls"] = state.get("sub_agent_calls", [])
            
            # Add A2A communication results
            if state.get("agent_communication"):
                output["agent_communication"] = state["agent_communication"]
            
            # Add error information if any
            if state.get("errors"):
                output["errors"] = state["errors"]
            
            if state.get("fallback_responses"):
                output["fallback_responses"] = state["fallback_responses"]
            
            # Calculate total execution time
            start_time = state.get("execution_context", {}).get("start_time")
            if start_time:
                total_time = (datetime.now() - datetime.fromisoformat(start_time)).total_seconds()
                output["total_execution_time"] = total_time
            
            return output
        
        return output_node
    
    def _create_enhanced_error_handler_node(self) -> Callable:
        """Create enhanced error handler node"""
        async def error_handler_node(state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info("Executing enhanced error handler node")
            
            # Collect all errors
            errors = state.get("errors", [])
            error_context = {
                "total_errors": len(errors),
                "error_types": list(set(error.get("error_type", "unknown") for error in errors)),
                "handled_at": datetime.now().isoformat(),
                "execution_state": state.get("current_step", "unknown")
            }
            
            # Determine recovery strategy
            recovery_strategy = self._determine_recovery_strategy(errors, state)
            
            # Apply recovery
            recovered_state = await self._apply_recovery_strategy(recovery_strategy, state)
            
            return {
                "current_step": "error_handler",
                "error_context": error_context,
                "recovery_strategy": recovery_strategy,
                "errors": errors,
                "fallback_responses": state.get("fallback_responses", []) + [
                    "I encountered some issues while processing your request, but I'll do my best to help you."
                ],
                **recovered_state
            }
        
        return error_handler_node
    
    def _evaluate_enhanced_trigger_condition(self, condition: Optional[str], state: Dict[str, Any]) -> bool:
        """Enhanced trigger condition evaluation with A2A context"""
        if not condition:
            return True
        
        try:
            # Enhanced condition evaluation with more context
            context = {
                **state,
                "message_count": len(state.get("messages", [])),
                "has_knowledge_context": bool(state.get("retrieved_context")),
                "has_tool_results": bool(state.get("tool_results")),
                "execution_depth": state.get("execution_context", {}).get("execution_depth", 0),
                "a2a_messages": len(state.get("incoming_messages", []))
            }
            
            # Replace variables in condition
            for key, value in context.items():
                if isinstance(value, str):
                    condition = condition.replace(f"{{{key}}}", f"'{value}'")
                elif isinstance(value, bool):
                    condition = condition.replace(f"{{{key}}}", str(value).lower())
                else:
                    condition = condition.replace(f"{{{key}}}", str(value))
            
            # Evaluate with safe execution
            return eval(condition, {"__builtins__": {}}, {})
            
        except Exception as e:
            logger.error(f"Failed to evaluate trigger condition '{condition}': {e}")
            return False
    
    def _determine_recovery_strategy(self, errors: List[Dict[str, Any]], state: Dict[str, Any]) -> str:
        """Determine appropriate recovery strategy based on errors"""
        if not errors:
            return "continue"
        
        error_types = [error.get("error_type", "unknown") for error in errors]
        
        if "RateLimitError" in error_types:
            return "retry_with_backoff"
        elif "AuthenticationError" in error_types:
            return "fallback_to_cached"
        elif "NetworkError" in error_types:
            return "retry_once"
        else:
            return "fallback_response"
    
    async def _apply_recovery_strategy(self, strategy: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply recovery strategy"""
        if strategy == "continue":
            return {"retry_count": state.get("retry_count", 0)}
        elif strategy == "retry_with_backoff":
            return {"retry_count": state.get("retry_count", 0) + 1, "backoff_delay": 2}
        elif strategy == "fallback_to_cached":
            return {"using_cached_response": True}
        elif strategy == "retry_once":
            return {"retry_count": state.get("retry_count", 0) + 1}
        else:  # fallback_response
            return {"using_fallback": True}
    
    def _route_sub_agents(self, state: Dict[str, Any]) -> str:
        """Route sub-agents based on state"""
        sub_agent_responses = state.get("sub_agent_responses", [])
        errors = state.get("errors", [])
        
        if errors:
            return "error"
        elif sub_agent_responses:
            return "continue"
        else:
            return "continue"
    
    def _create_conditional_router(self, conditions: List[Dict[str, Any]]) -> Callable:
        """Create conditional router function"""
        async def conditional_router(state: Dict[str, Any]) -> str:
            for condition in conditions:
                if self._evaluate_enhanced_trigger_condition(condition.get("condition"), state):
                    return condition.get("destination", "output")
            return "output"
        return conditional_router
    
    def _create_parallel_tool_node(self, config: Dict[str, Any], agent: Agent) -> Callable:
        """Create parallel tool execution node"""
        async def parallel_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
            # Similar to enhanced tool execution but optimized for parallel processing
            return await self._create_enhanced_tool_execution_node(config, agent)(state)
        return parallel_tool_node
    
    def _create_parallel_processor_node(self, config: Dict[str, Any], agent: Agent) -> Callable:
        """Create parallel processor node for concurrent operations"""
        async def parallel_processor_node(state: Dict[str, Any]) -> Dict[str, Any]:
            # Implement parallel processing logic
            tasks = []
            
            # Create parallel tasks based on config
            for task_config in config.get("tasks", []):
                if task_config["type"] == "tool_execution":
                    task = self._create_enhanced_tool_execution_node(task_config, agent)(state)
                    tasks.append(task)
                elif task_config["type"] == "knowledge_retrieval":
                    task = self._create_enhanced_knowledge_retrieval_node(task_config, agent)(state)
                    tasks.append(task)
            
            # Execute tasks in parallel
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return {"parallel_results": results}
            
            return state
        
        return parallel_processor_node
    
    def _create_conditional_router_node(self, config: Dict[str, Any]) -> Callable:
        """Create conditional router node"""
        async def conditional_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
            # Implement conditional routing logic
            return {"current_step": "conditional_router"}
        return conditional_router_node
    
    def _create_enhanced_memory_manager_node(self, config: Dict[str, Any], agent: Agent) -> Callable:
        """Create enhanced memory manager node"""
        async def memory_manager_node(state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info("Executing enhanced memory manager node")
            
            memory_buffer = state.get("memory_buffer", [])
            conversation_history = state.get("conversation_history", [])
            
            # Add current messages to buffer
            if state.get("messages"):
                memory_buffer.extend(state["messages"])
            
            # Manage memory based on configuration
            window_size = config.get("window_size", 10)
            
            # Keep only recent messages if buffer is too large
            if len(memory_buffer) > window_size:
                memory_buffer = memory_buffer[-window_size:]
            
            # Summarize older context if needed
            summarized_context = ""
            if config.get("summarization", False) and len(memory_buffer) > window_size // 2:
                summarized_context = await self._summarize_context(memory_buffer[:-window_size//2])
                memory_buffer = memory_buffer[-window_size//2:]
            
            return {
                "memory_buffer": memory_buffer,
                "conversation_history": conversation_history,
                "summarized_context": summarized_context,
                "current_step": "memory_manager",
                "execution_trace": state.get("execution_trace", []) + [{
                    "step": "memory_manager",
                    "timestamp": datetime.now().isoformat(),
                    "success": True,
                    "memory_size": len(memory_buffer)
                }]
            }
        
        return memory_manager_node
    
    async def _summarize_context(self, context_messages: List[Dict[str, Any]]) -> str:
        """Summarize context messages"""
        # Simple summarization - in production, use a proper summarization model
        content_parts = []
        for msg in context_messages:
            if isinstance(msg, dict) and "content" in msg:
                content_parts.append(msg["content"])
        
        return " ".join(content_parts)[:500] + "..."
