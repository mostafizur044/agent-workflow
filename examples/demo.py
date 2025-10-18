"""
Demo script for AI Agent Workflow System

This script demonstrates how to:
1. Create an agent
2. Configure components (knowledge base, tools)
3. Publish the agent
4. Execute the agent with streaming
"""

import asyncio
import requests
import json
from uuid import uuid4
import time

# Configuration
BASE_URL = "http://localhost:8000"
USER_ID = "demo_user"

def create_agent():
    """Create a new customer support agent"""
    print("Creating customer support agent...")
    
    response = requests.post(f"{BASE_URL}/api/agents/", json={
        "name": "Customer Support Agent",
        "description": "Handles customer inquiries with knowledge base and tools",
        "user_id": USER_ID
    })
    
    if response.status_code == 201:
        agent = response.json()
        print(f"✓ Agent created: {agent['id']}")
        return agent
    else:
        print(f"✗ Failed to create agent: {response.text}")
        return None

def create_knowledge_base():
    """Create a knowledge base with sample FAQ data"""
    print("Creating knowledge base...")
    
    response = requests.post(f"{BASE_URL}/api/knowledge-bases/", json={
        "name": "Customer FAQ",
        "description": "Frequently asked questions and answers",
        "user_id": USER_ID
    })
    
    if response.status_code == 201:
        kb = response.json()
        print(f"✓ Knowledge base created: {kb['id']}")
        return kb
    else:
        print(f"✗ Failed to create knowledge base: {response.text}")
        return None

def upload_sample_documents(kb_id):
    """Upload sample FAQ documents"""
    print("Uploading sample documents...")
    
    sample_docs = [
        {
            "content": "Our refund policy allows returns within 30 days of purchase. Contact support to initiate a refund.",
            "metadata": {
                "source": "refund_policy.pdf",
                "title": "Refund Policy",
                "tags": ["refund", "returns", "policy"]
            }
        },
        {
            "content": "Shipping is free for orders over $50. Standard shipping takes 3-5 business days, express shipping takes 1-2 business days.",
            "metadata": {
                "source": "shipping_info.pdf",
                "title": "Shipping Information",
                "tags": ["shipping", "delivery", "cost"]
            }
        },
        {
            "content": "You can track your order by logging into your account or using the tracking number sent to your email.",
            "metadata": {
                "source": "order_tracking.pdf",
                "title": "Order Tracking",
                "tags": ["tracking", "order", "status"]
            }
        }
    ]
    
    for doc in sample_docs:
        response = requests.post(f"{BASE_URL}/api/knowledge-bases/{kb_id}/documents", json=doc)
        if response.status_code == 200:
            print(f"✓ Document uploaded: {doc['metadata']['title']}")
        else:
            print(f"✗ Failed to upload document: {response.text}")

def create_tool():
    """Create a sample CRM lookup tool"""
    print("Creating CRM lookup tool...")
    
    response = requests.post(f"{BASE_URL}/api/tools/", json={
        "name": "CRM Lookup",
        "description": "Lookup customer information from CRM",
        "type": "api",
        "config": {
            "api": {
                "base_url": "https://api.example-crm.com/customers/{customer_id}",
                "headers": {
                    "Authorization": "Bearer demo_token",
                    "Content-Type": "application/json"
                },
                "timeout": 30,
                "retries": 3
            }
        },
        "parameters": [
            {
                "name": "customer_id",
                "type": "string",
                "description": "Customer ID to lookup",
                "required": True
            }
        ],
        "tags": ["crm", "customer", "lookup"],
        "user_id": USER_ID
    })
    
    if response.status_code == 201:
        tool = response.json()
        print(f"✓ Tool created: {tool['id']}")
        return tool
    else:
        print(f"✗ Failed to create tool: {response.text}")
        return None

def configure_agent(agent_id, kb_id, tool_id):
    """Configure the agent with knowledge base and tools"""
    print("Configuring agent...")
    
    # Enable knowledge base
    response = requests.patch(f"{BASE_URL}/api/agents/{agent_id}/toggle", json={
        "path": "knowledge_bases.enabled",
        "value": True
    })
    
    if response.status_code == 200:
        print("✓ Knowledge base enabled")
    else:
        print(f"✗ Failed to enable knowledge base: {response.text}")
    
    # Add knowledge base to agent
    response = requests.post(f"{BASE_URL}/api/agents/{agent_id}/knowledge-bases", 
                           params={"kb_id": kb_id, "priority": 1})
    
    if response.status_code == 200:
        print("✓ Knowledge base added to agent")
    else:
        print(f"✗ Failed to add knowledge base: {response.text}")
    
    # Enable tools
    response = requests.patch(f"{BASE_URL}/api/agents/{agent_id}/toggle", json={
        "path": "tools.enabled",
        "value": True
    })
    
    if response.status_code == 200:
        print("✓ Tools enabled")
    else:
        print(f"✗ Failed to enable tools: {response.text}")
    
    # Add tool to agent
    response = requests.post(f"{BASE_URL}/api/agents/{agent_id}/tools", 
                           params={"tool_id": tool_id})
    
    if response.status_code == 200:
        print("✓ Tool added to agent")
    else:
        print(f"✗ Failed to add tool: {response.text}")
    
    # Update LLM configuration
    response = requests.put(f"{BASE_URL}/api/agents/{agent_id}", json={
        "config": {
            "llm": {
                "enabled": True,
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000,
                "system_prompt": "You are a helpful customer support agent. Use the knowledge base to answer questions accurately and use tools when needed to lookup customer information.",
                "streaming": True
            },
            "knowledge_bases": {
                "enabled": True,
                "items": [
                    {
                        "id": kb_id,
                        "name": "Customer FAQ",
                        "enabled": True,
                        "priority": 1,
                        "retrieval_config": {
                            "top_k": 3,
                            "score_threshold": 0.7,
                            "search_type": "similarity"
                        }
                    }
                ]
            },
            "tools": {
                "enabled": True,
                "items": [
                    {
                        "id": tool_id,
                        "name": "CRM Lookup",
                        "type": "api",
                        "enabled": True,
                        "config": {}
                    }
                ]
            },
            "memory": {
                "enabled": True,
                "type": "buffer",
                "window_size": 5,
                "summarization": False
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
        print("✓ Agent configuration updated")
    else:
        print(f"✗ Failed to update agent configuration: {response.text}")

def publish_agent(agent_id):
    """Publish the agent"""
    print("Publishing agent...")
    
    response = requests.post(f"{BASE_URL}/api/agents/{agent_id}/publish")
    
    if response.status_code == 200:
        print("✓ Agent published successfully")
        return True
    else:
        print(f"✗ Failed to publish agent: {response.text}")
        return False

def execute_agent_sync(agent_id):
    """Execute agent synchronously"""
    print("\nExecuting agent (synchronous)...")
    
    response = requests.post(f"{BASE_URL}/api/execution/", json={
        "agent_id": agent_id,
        "input": "What is your refund policy?",
        "stream": False,
        "user_id": USER_ID
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Agent response: {result['output']}")
        return result
    else:
        print(f"✗ Failed to execute agent: {response.text}")
        return None

def execute_agent_stream(agent_id):
    """Execute agent with streaming"""
    print("\nExecuting agent (streaming)...")
    
    response = requests.post(f"{BASE_URL}/api/execution/stream", json={
        "agent_id": agent_id,
        "input": "How can I track my order?",
        "stream": True,
        "user_id": USER_ID
    }, stream=True)
    
    if response.status_code == 200:
        print("Streaming response:")
        for line in response.iter_lines():
            if line:
                try:
                    # Parse SSE format
                    if line.startswith(b'data: '):
                        data = line[6:].decode('utf-8')
                        if data.strip():
                            chunk = json.loads(data)
                            if chunk['type'] == 'step':
                                print(f"  Step: {chunk['data']}")
                            elif chunk['type'] == 'complete':
                                print(f"  ✓ Execution completed")
                            elif chunk['type'] == 'error':
                                print(f"  ✗ Error: {chunk['data']}")
                except json.JSONDecodeError:
                    pass
    else:
        print(f"✗ Failed to execute agent stream: {response.text}")

def main():
    """Main demo function"""
    print("AI Agent Workflow System Demo")
    print("=" * 50)
    
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
    
    # Create components
    agent = create_agent()
    if not agent:
        return
    
    kb = create_knowledge_base()
    if not kb:
        return
    
    upload_sample_documents(kb['id'])
    
    tool = create_tool()
    if not tool:
        return
    
    # Configure and publish agent
    configure_agent(agent['id'], kb['id'], tool['id'])
    
    if not publish_agent(agent['id']):
        return
    
    # Execute agent
    execute_agent_sync(agent['id'])
    execute_agent_stream(agent['id'])
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print(f"Agent ID: {agent['id']}")
    print(f"Knowledge Base ID: {kb['id']}")
    print(f"Tool ID: {tool['id']}")
    print("\nYou can now:")
    print(f"- View API docs: {BASE_URL}/docs")
    print(f"- Test the agent via API or web interface")
    print(f"- Add more knowledge bases, tools, or sub-agents")

if __name__ == "__main__":
    main()
