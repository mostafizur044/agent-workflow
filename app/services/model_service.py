from typing import Optional, List, Dict, Any
from uuid import uuid4
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging

from app.database import get_database
from app.models.model import (
    Model, ModelCreateRequest, ModelUpdateRequest, ModelTestRequest,
    ModelTestResponse, ModelProvider, ModelType, PresetModel, PRESET_MODELS
)
from config import settings

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing AI models and providers"""
    
    def __init__(self):
        self.db: AsyncIOMotorDatabase = get_database()
    
    async def create_model(self, request: ModelCreateRequest) -> Model:
        """Create a new model registration"""
        model_id = str(uuid4())
        now = datetime.now()
        
        model = Model(
            id=model_id,
            user_id=request.user_id,
            name=request.name,
            description=request.description,
            provider=request.provider,
            model_name=request.model_name,
            model_type=request.model_type,
            config=request.config,
            is_shared=request.is_shared,
            tags=request.tags,
            created_at=now,
            updated_at=now
        )
        
        # Save to database
        await self.db.get_collection("models").insert_one(model.model_dump(by_alias=True))
        
        logger.info(f"Created model {model_id} for user {request.user_id}")
        return model
    
    async def get_model(self, model_id: str) -> Optional[Model]:
        """Get model by ID"""
        doc = await self.db.get_collection("models").find_one({"_id": str(model_id)})
        if doc:
            doc["id"] = doc.pop("_id")
            return Model(**doc)
        return None
    
    async def get_models_by_user(self, user_id: str) -> List[Model]:
        """Get all models for a user"""
        cursor = self.db.get_collection("models").find({
            "$or": [
                {"user_id": user_id},
                {"is_shared": True}
            ]
        }).sort("created_at", -1)
        
        models = []
        async for doc in cursor:
            doc["id"] = doc.pop("_id")
            models.append(Model(**doc))
        return models
    
    async def get_models_by_provider(self, provider: ModelProvider) -> List[Model]:
        """Get models by provider"""
        cursor = self.db.get_collection("models").find({
            "provider": provider.value,
            "is_active": True
        }).sort("created_at", -1)
        
        models = []
        async for doc in cursor:
            doc["id"] = doc.pop("_id")
            models.append(Model(**doc))
        return models
    
    async def get_models_by_type(self, model_type: ModelType) -> List[Model]:
        """Get models by type"""
        cursor = self.db.get_collection("models").find({
            "model_type": model_type.value,
            "is_active": True
        }).sort("created_at", -1)
        
        models = []
        async for doc in cursor:
            doc["id"] = doc.pop("_id")
            models.append(Model(**doc))
        return models
    
    async def update_model(self, model_id: str, request: ModelUpdateRequest) -> Optional[Model]:
        """Update model"""
        update_data = {}
        
        if request.name is not None:
            update_data["name"] = request.name
        if request.description is not None:
            update_data["description"] = request.description
        if request.model_name is not None:
            update_data["model_name"] = request.model_name
        if request.config is not None:
            update_data["config"] = request.config
        if request.is_active is not None:
            update_data["is_active"] = request.is_active
        if request.is_shared is not None:
            update_data["is_shared"] = request.is_shared
        if request.tags is not None:
            update_data["tags"] = request.tags
        
        update_data["updated_at"] = datetime.now()
        
        result = await self.db.get_collection("models").update_one(
            {"_id": str(model_id)},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            return await self.get_model(model_id)
        
        return None
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete model"""
        result = await self.db.get_collection("models").delete_one({"_id": str(model_id)})
        
        if result.deleted_count > 0:
            logger.info(f"Deleted model {model_id}")
            return True
        
        return False
    
    async def test_model(self, model_id: str, request: ModelTestRequest) -> ModelTestResponse:
        """Test a model with a sample prompt"""
        model = await self.get_model(model_id)
        if not model:
            return ModelTestResponse(
                success=False,
                error="Model not found",
                execution_time=0.0
            )
        
        start_time = datetime.now()
        
        try:
            # Create model client based on provider
            client = await self._create_model_client(model)
            
            if model.model_type == ModelType.CHAT:
                response = await self._test_chat_model(client, model, request)
            elif model.model_type == ModelType.COMPLETION:
                response = await self._test_completion_model(client, model, request)
            elif model.model_type == ModelType.EMBEDDING:
                response = await self._test_embedding_model(client, model, request)
            else:
                raise ValueError(f"Unsupported model type: {model.model_type}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ModelTestResponse(
                success=True,
                response=response.get("content", str(response)),
                execution_time=execution_time,
                tokens_used=response.get("tokens_used")
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Model test failed: {e}")
            
            return ModelTestResponse(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _create_model_client(self, model: Model):
        """Create appropriate model client based on provider"""
        if model.provider == ModelProvider.OPENAI:
            return await self._create_openai_client(model)
        elif model.provider == ModelProvider.AZURE:
            return await self._create_azure_client(model)
        elif model.provider == ModelProvider.ANTHROPIC:
            return await self._create_anthropic_client(model)
        elif model.provider == ModelProvider.GOOGLE:
            return await self._create_google_client(model)
        elif model.provider == ModelProvider.COHERE:
            return await self._create_cohere_client(model)
        elif model.provider == ModelProvider.OPENROUTER:
            return await self._create_openrouter_client(model)
        elif model.provider == ModelProvider.LOCAL:
            return await self._create_local_client(model)
        else:
            raise ValueError(f"Unsupported provider: {model.provider}")
    
    async def _create_openai_client(self, model: Model):
        """Create OpenAI client"""
        try:
            import openai
            
            client_config = {
                "api_key": model.config.get("api_key") or settings.openai_api_key,
            }
            
            if model.config.get("base_url"):
                client_config["base_url"] = model.config["base_url"]
            
            if model.config.get("organization"):
                client_config["organization"] = model.config["organization"]
            
            return openai.AsyncOpenAI(**client_config)
        except ImportError:
            raise ValueError("OpenAI library not installed")
    
    async def _create_azure_client(self, model: Model):
        """Create Azure OpenAI client"""
        try:
            import openai
            
            return openai.AsyncAzureOpenAI(
                api_key=model.config.get("api_key"),
                api_version=model.config.get("api_version", "2024-02-15-preview"),
                azure_endpoint=model.config.get("azure_endpoint"),
                azure_ad_token=model.config.get("azure_ad_token")
            )
        except ImportError:
            raise ValueError("OpenAI library not installed")
    
    async def _create_anthropic_client(self, model: Model):
        """Create Anthropic client"""
        try:
            import anthropic
            
            return anthropic.AsyncAnthropic(
                api_key=model.config.get("api_key"),
                base_url=model.config.get("base_url")
            )
        except ImportError:
            raise ValueError("Anthropic library not installed")
    
    async def _create_google_client(self, model: Model):
        """Create Google client"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=model.config.get("api_key"))
            return genai
        except ImportError:
            raise ValueError("Google Generative AI library not installed")
    
    async def _create_cohere_client(self, model: Model):
        """Create Cohere client"""
        try:
            import cohere
            
            return cohere.AsyncClient(
                api_key=model.config.get("api_key"),
                base_url=model.config.get("base_url")
            )
        except ImportError:
            raise ValueError("Cohere library not installed")
    
    async def _create_openrouter_client(self, model: Model):
        """Create OpenRouter client"""
        try:
            import openai
            
            return openai.AsyncOpenAI(
                api_key=model.config.get("api_key"),
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": model.config.get("website", "http://localhost:3000"),
                    "X-Title": model.config.get("app_name", "Agent Workflow")
                }
            )
        except ImportError:
            raise ValueError("OpenAI library not installed")
    
    async def _create_local_client(self, model: Model):
        """Create local model client"""
        # This would depend on the specific local model implementation
        # For now, return a placeholder
        return {"type": "local", "path": model.config.get("local_path")}
    
    async def _test_chat_model(self, client, model: Model, request: ModelTestRequest) -> Dict[str, Any]:
        """Test chat model"""
        if model.provider == ModelProvider.OPENAI or model.provider == ModelProvider.AZURE or model.provider == ModelProvider.OPENROUTER:
            response = await client.chat.completions.create(
                model=model.model_name,
                messages=[{"role": "user", "content": request.test_prompt}],
                max_tokens=request.max_tokens or model.config.get("max_tokens", 100),
                temperature=model.config.get("temperature", 0.7)
            )
            
            return {
                "content": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens if response.usage else None
            }
        
        elif model.provider == ModelProvider.ANTHROPIC:
            response = await client.messages.create(
                model=model.model_name,
                max_tokens=request.max_tokens or model.config.get("max_tokens", 100),
                temperature=model.config.get("temperature", 0.7),
                messages=[{"role": "user", "content": request.test_prompt}]
            )
            
            return {
                "content": response.content[0].text,
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens if response.usage else None
            }
        
        elif model.provider == ModelProvider.GOOGLE:
            gen_model = client.GenerativeModel(model.model_name)
            response = await gen_model.generate_content_async(
                request.test_prompt,
                generation_config={
                    "max_output_tokens": request.max_tokens or model.config.get("max_tokens", 100),
                    "temperature": model.config.get("temperature", 0.7)
                }
            )
            
            return {
                "content": response.text,
                "tokens_used": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else None
            }
        
        else:
            raise ValueError(f"Chat testing not implemented for provider: {model.provider}")
    
    async def _test_completion_model(self, client, model: Model, request: ModelTestRequest) -> Dict[str, Any]:
        """Test completion model"""
        if model.provider == ModelProvider.OPENAI or model.provider == ModelProvider.AZURE or model.provider == ModelProvider.OPENROUTER:
            response = await client.completions.create(
                model=model.model_name,
                prompt=request.test_prompt,
                max_tokens=request.max_tokens or model.config.get("max_tokens", 100),
                temperature=model.config.get("temperature", 0.7)
            )
            
            return {
                "content": response.choices[0].text,
                "tokens_used": response.usage.total_tokens if response.usage else None
            }
        else:
            raise ValueError(f"Completion testing not implemented for provider: {model.provider}")
    
    async def _test_embedding_model(self, client, model: Model, request: ModelTestRequest) -> Dict[str, Any]:
        """Test embedding model"""
        if model.provider == ModelProvider.OPENAI or model.provider == ModelProvider.AZURE or model.provider == ModelProvider.OPENROUTER:
            response = await client.embeddings.create(
                model=model.model_name,
                input=request.test_prompt
            )
            
            return {
                "content": f"Embedding generated with {len(response.data[0].embedding)} dimensions",
                "tokens_used": response.usage.total_tokens if response.usage else None,
                "embedding": response.data[0].embedding
            }
        else:
            raise ValueError(f"Embedding testing not implemented for provider: {model.provider}")
    
    async def get_preset_models(self) -> List[PresetModel]:
        """Get preset model configurations"""
        return PRESET_MODELS
    
    async def create_from_preset(self, preset_name: str, user_id: str, custom_config: Dict[str, Any] = None) -> Optional[Model]:
        """Create model from preset configuration"""
        preset = next((p for p in PRESET_MODELS if p.name == preset_name), None)
        if not preset:
            return None
        
        config = preset.config.copy()
        if custom_config:
            config.update(custom_config)
        
        request = ModelCreateRequest(
            name=preset.name,
            description=preset.description,
            provider=preset.provider,
            model_name=preset.model_name,
            model_type=preset.model_type,
            config=config,
            tags=["preset", preset.provider.value],
            user_id=user_id
        )
        
        return await self.create_model(request)
    
    async def validate_model_config(self, provider: ModelProvider, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model configuration"""
        errors = []
        warnings = []
        
        # Check required fields based on provider
        if provider == ModelProvider.OPENAI:
            if not config.get("api_key") and not settings.openai_api_key:
                warnings.append("API key not provided")
        
        elif provider == ModelProvider.AZURE:
            required_fields = ["api_key", "azure_endpoint", "deployment_name"]
            for field in required_fields:
                if not config.get(field):
                    errors.append(f"Required field '{field}' not provided")
        
        elif provider == ModelProvider.ANTHROPIC:
            if not config.get("api_key"):
                warnings.append("API key not provided")
        
        elif provider == ModelProvider.GOOGLE:
            if not config.get("api_key"):
                warnings.append("API key not provided")
        
        elif provider == ModelProvider.COHERE:
            if not config.get("api_key"):
                warnings.append("API key not provided")
        
        elif provider == ModelProvider.OPENROUTER:
            if not config.get("api_key"):
                errors.append("API key is required for OpenRouter")
        
        elif provider == ModelProvider.LOCAL:
            if not config.get("local_path"):
                errors.append("Local path is required for local models")
        
        return {
            "errors": errors,
            "warnings": warnings,
            "valid": len(errors) == 0
        }
