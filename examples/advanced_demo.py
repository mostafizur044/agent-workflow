"""
Advanced Demo Script for AI Agent Workflow System

This script demonstrates the new enhanced features:
1. Model management with multiple providers
2. Advanced knowledge base with different chunking methods
3. LangChain tools integration
4. Draft/publish workflow
5. Playground testing
6. Enhanced agent configuration
"""

import asyncio
import requests
import json
from uuid import uuid4
import time

# Configuration
BASE_URL = "http://localhost:8000"
USER_ID = "advanced_demo_user"

def create_model():
    """Create a model registration"""
    print("Creating model registration...")
    
    response = requests.post(f"{BASE_URL}/api/models/", json={
        "name": "GPT-4 Turbo Advanced",
        "description": "Advanced GPT-4 model for complex tasks",
        "provider": "openai",
        "model_name": "gpt-4-turbo-preview",
        "model_type": "chat",
        "config": {
            "api_key": "your_openai_key_here",
            "max_tokens": 4096,
            "temperature": 0.7,
            "supports_function_calling": True,
            "supports_vision": True
        },
        "is_shared": False,
        "tags": ["advanced", "gpt4", "function_calling"],
        "user_id": USER_ID
    })
    
    if response.status_code == 201:
        model = response.json()
        print(f"✓ Model created: {model['id']}")
        return model
    else:
        print(f"✗ Failed to create model: {response.text}")
        return None

def test_model():
    """Test the created model"""
    print("Testing model...")
    
    # Get the model first
    models_response = requests.get(f"{BASE_URL}/api/models/", params={"user_id": USER_ID})
    if models_response.status_code == 200:
        models = models_response.json()
        if models:
            model_id = models[0]["id"]
            
            response = requests.post(f"{BASE_URL}/api/models/{model_id}/test", json={
                "test_prompt": "Hello, can you help me with a complex task?",
                "max_tokens": 100
            })
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Model test successful: {result['response'][:100]}...")
                return model_id
            else:
                print(f"✗ Model test failed: {response.text}")
    
    return None

def create_langchain_tool():
    """Create a LangChain tool"""
    print("Creating LangChain tool...")
    
    # Get available LangChain tools
    tools_response = requests.get(f"{BASE_URL}/api/tools/langchain/list")
    if tools_response.status_code == 200:
        tools = tools_response.json()
        search_tools = [t for t in tools["tools"] if t["category"] == "search"]
        
        if search_tools:
            tool_name = search_tools[0]["tool_name"]
            
            response = requests.post(f"{BASE_URL}/api/tools/langchain/{tool_name}/create", 
                                   params={"user_id": USER_ID},
                                   json={
                                       "num_results": 5,
                                       "country": "us"
                                   })
            
            if response.status_code == 200:
                tool = response.json()
                print(f"✓ LangChain tool created: {tool['id']}")
                return tool
            else:
                print(f"✗ Failed to create LangChain tool: {response.text}")
    
    return None

def create_advanced_knowledge_base():
    """Create knowledge base with advanced chunking"""
    print("Creating advanced knowledge base...")
    
    response = requests.post(f"{BASE_URL}/api/knowledge-bases/", json={
        "name": "Advanced FAQ with Smart Chunking",
        "description": "FAQ with semantic chunking and multiple embedding models",
        "user_id": USER_ID,
        "embedding_model": {
            "provider": "openai",
            "model_name": "text-embedding-3-large",
            "dimensions": 3072
        },
        "chunking_config": {
            "method": "semantic",
            "chunk_size": 800,
            "chunk_overlap": 150,
            "separators": ["\n\n", "\n", ". ", " "],
            "length_function": "len"
        }
    })
    
    if response.status_code == 201:
        kb = response.json()
        print(f"✓ Advanced knowledge base created: {kb['id']}")
        return kb
    else:
        print(f"✗ Failed to create knowledge base: {response.text}")
        return None

def test_playground():
    """Test playground features"""
    print("Testing playground features...")
    
    sample_content = """
    Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
    
    Machine learning is a subset of AI that focuses on algorithms that can learn from data. Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.
    
    Natural Language Processing (NLP) is a subfield of AI that focuses on the interaction between computers and humans through natural language. It involves developing algorithms and models that enable computers to understand, interpret, and generate human language.
    """
    
    # Test chunking
    chunking_response = requests.post(f"{BASE_URL}/api/playground/chunking/test", json={
        "content": sample_content,
        "chunking_config": {
            "method": "sentence",
            "chunk_size": 200,
            "chunk_overlap": 50
        }
    })
    
    if chunking_response.status_code == 200:
        result = chunking_response.json()
        print(f"✓ Chunking test successful: {result['chunk_count']} chunks created")
        print(f"  Average chunk size: {result['average_chunk_size']:.1f} characters")
    
    # Test embedding
    embedding_response = requests.post(f"{BASE_URL}/api/playground/embedding/test", json={
        "text": "What is artificial intelligence?",
        "embedding_model": {
            "provider": "openai",
            "model_name": "text-embedding-3-large"
        }
    })
    
    if embedding_response.status_code == 200:
        result = embedding_response.json()
        print(f"✓ Embedding test successful: {result['embedding_dimensions']} dimensions")

def create_agent_with_advanced_features():
    """Create agent with all advanced features"""
    print("Creating advanced agent...")
    
    response = requests.post(f"{BASE_URL}/api/agents/", json={
        "name": "Advanced Customer Support Agent",
        "description": "AI agent with advanced features for customer support",
        "user_id": USER_ID
    })
    
    if response.status_code == 201:
        agent = response.json()
        print(f"✓ Agent created: {agent['id']}")
        return agent
    else:
        print(f"✗ Failed to create agent: {response.text}")
        return None

def configure_agent_advanced(agent_id, model_id, tool_id, kb_id):
    """Configure agent with advanced features"""
    print("Configuring agent with advanced features...")
    
    # Update agent with advanced configuration
    response = requests.put(f"{BASE_URL}/api/agents/{agent_id}", json={
        "config": {
            "llm": {
                "enabled": True,
                "provider": "openai",
                "model": "gpt-4-turbo-preview",
                "temperature": 0.3,
                "max_tokens": 2000,
                "system_prompt": "You are an advanced AI customer support agent with access to multiple tools and knowledge bases. Use your capabilities wisely to provide accurate and helpful responses.",
                "streaming": True
            },
            "knowledge_bases": {
                "enabled": True,
                "items": [
                    {
                        "id": kb_id,
                        "name": "Advanced FAQ",
                        "enabled": True,
                        "priority": 1,
                        "retrieval_config": {
                            "top_k": 3,
                            "score_threshold": 0.8,
                            "search_type": "similarity",
                            "rerank_config": {
                                "enabled": True,
                                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                                "top_k": 10,
                                "return_top_k": 3
                            }
                        }
                    }
                ]
            },
            "tools": {
                "enabled": True,
                "items": [
                    {
                        "id": tool_id,
                        "name": "Web Search Tool",
                        "type": "langchain",
                        "enabled": True,
                        "config": {
                            "num_results": 5
                        }
                    }
                ]
            },
            "memory": {
                "enabled": True,
                "type": "buffer",
                "window_size": 10,
                "summarization": True
            },
            "routing": {
                "type": "conditional",
                "rules": []
            },
            "metadata": {
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "created_by": USER_ID
            }
        }
    })
    
    if response.status_code == 200:
        print("✓ Agent configuration updated with advanced features")
    else:
        print(f"✗ Failed to update agent configuration: {response.text}")

def publish_agent(agent_id):
    """Publish agent using new publish system"""
    print("Publishing agent...")
    
    # Validate before publishing
    validation_response = requests.post(f"{BASE_URL}/api/publish/validate/{agent_id}")
    if validation_response.status_code == 200:
        validation = validation_response.json()
        if validation["valid"]:
            print("✓ Agent validation passed")
        else:
            print(f"✗ Agent validation failed: {validation['errors']}")
            return None
    
    # Publish agent
    publish_response = requests.post(f"{BASE_URL}/api/publish/publish", json={
        "agent_id": agent_id,
        "publish_as_draft": False,
        "force_recompile": True,
        "validation_options": {
            "check_dependencies": True,
            "test_connections": True
        }
    })
    
    if publish_response.status_code == 200:
        result = publish_response.json()
        if result["success"]:
            print(f"✓ Agent published successfully: {result['published_agent_id']}")
            return result["published_agent_id"]
        else:
            print(f"✗ Agent publishing failed: {result['errors']}")
    
    return None

def execute_advanced_agent(published_agent_id):
    """Execute the advanced agent"""
    print("Executing advanced agent...")
    
    response = requests.post(f"{BASE_URL}/api/execution/", json={
        "agent_id": published_agent_id,
        "input": "I need help with a complex technical issue. Can you search for the latest information and provide a detailed solution?",
        "stream": False,
        "user_id": USER_ID,
        "context": {
            "customer_type": "premium",
            "issue_category": "technical"
        }
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Agent execution successful")
        print(f"  Response: {result['output'][:200]}...")
        print(f"  Execution time: {result['execution_time']:.2f}s")
        
        # Show execution trace if available
        if result.get('execution_trace'):
            trace = result['execution_trace']
            print(f"  Steps executed: {len(trace.get('steps', []))}")
        
        return result
    else:
        print(f"✗ Agent execution failed: {response.text}")
        return None

def get_execution_stats(published_agent_id):
    """Get execution statistics"""
    print("Getting execution statistics...")
    
    response = requests.get(f"{BASE_URL}/api/publish/stats/{published_agent_id}")
    if response.status_code == 200:
        stats = response.json()
        print(f"✓ Execution statistics:")
        print(f"  Total executions: {stats['total_executions']}")
        print(f"  Successful: {stats['successful_executions']}")
        print(f"  Failed: {stats['failed_executions']}")
        if stats['average_execution_time']:
            print(f"  Average execution time: {stats['average_execution_time']:.2f}s")
    
    return stats

def main():
    """Main demo function"""
    print("Advanced AI Agent Workflow System Demo")
    print("=" * 60)
    
    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("✗ API is not running. Please start the services first.")
            return
        print("✓ API is running")
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Please start the services first.")
        return
    
    # Step 1: Create and test model
    model = create_model()
    if not model:
        return
    
    model_id = test_model()
    if not model_id:
        return
    
    # Step 2: Create LangChain tool
    tool = create_langchain_tool()
    if not tool:
        return
    
    # Step 3: Create advanced knowledge base
    kb = create_advanced_knowledge_base()
    if not kb:
        return
    
    # Step 4: Test playground features
    test_playground()
    
    # Step 5: Create and configure agent
    agent = create_agent_with_advanced_features()
    if not agent:
        return
    
    configure_agent_advanced(agent["id"], model_id, tool["id"], kb["id"])
    
    # Step 6: Publish agent
    published_agent_id = publish_agent(agent["id"])
    if not published_agent_id:
        return
    
    # Step 7: Execute agent
    execution_result = execute_advanced_agent(published_agent_id)
    if not execution_result:
        return
    
    # Step 8: Get statistics
    get_execution_stats(published_agent_id)
    
    print("\n" + "=" * 60)
    print("Advanced Demo completed successfully!")
    print(f"Agent ID: {agent['id']}")
    print(f"Published Agent ID: {published_agent_id}")
    print(f"Model ID: {model_id}")
    print(f"Tool ID: {tool['id']}")
    print(f"Knowledge Base ID: {kb['id']}")
    print("\nYou can now:")
    print(f"- View API docs: {BASE_URL}/docs")
    print("- Test playground features at /api/playground")
    print("- Monitor execution statistics")
    print("- Add more advanced features to the agent")

if __name__ == "__main__":
    main()
