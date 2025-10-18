from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional
import logging

from app.services.model_service import ModelService
from app.models.model import (
    Model, ModelCreateRequest, ModelUpdateRequest, ModelTestRequest,
    ModelTestResponse, ModelProvider, ModelType, PresetModel
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency injection
def get_model_service() -> ModelService:
    return ModelService()


@router.post("/", response_model=Model, status_code=status.HTTP_201_CREATED)
async def create_model(
    request: ModelCreateRequest,
    model_service: ModelService = Depends(get_model_service)
):
    """Create a new model registration"""
    try:
        # Validate model configuration
        validation = await model_service.validate_model_config(request.provider, request.config)
        if not validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model configuration validation failed: {validation['errors']}"
            )
        
        model = await model_service.create_model(request)
        return model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create model"
        )


@router.get("/", response_model=List[Model])
async def get_models(
    user_id: str,
    provider: Optional[ModelProvider] = None,
    model_type: Optional[ModelType] = None,
    model_service: ModelService = Depends(get_model_service)
):
    """Get models for a user, optionally filtered by provider or type"""
    try:
        if provider and model_type:
            # Filter by both provider and type
            all_models = await model_service.get_models_by_user(user_id)
            models = [m for m in all_models if m.provider == provider and m.model_type == model_type]
        elif provider:
            models = await model_service.get_models_by_provider(provider)
            # Filter to user's models and shared models
            models = [m for m in models if m.user_id == user_id or m.is_shared]
        elif model_type:
            models = await model_service.get_models_by_type(model_type)
            # Filter to user's models and shared models
            models = [m for m in models if m.user_id == user_id or m.is_shared]
        else:
            models = await model_service.get_models_by_user(user_id)
        
        return models
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve models"
        )


@router.get("/{model_id}", response_model=Model)
async def get_model(
    model_id: str,
    model_service: ModelService = Depends(get_model_service)
):
    """Get model by ID"""
    model = await model_service.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    return model


@router.put("/{model_id}", response_model=Model)
async def update_model(
    model_id: str,
    request: ModelUpdateRequest,
    model_service: ModelService = Depends(get_model_service)
):
    """Update model"""
    model = await model_service.update_model(model_id, request)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    return model


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str,
    model_service: ModelService = Depends(get_model_service)
):
    """Delete model"""
    success = await model_service.delete_model(model_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )


@router.post("/{model_id}/test", response_model=ModelTestResponse)
async def test_model(
    model_id: str,
    request: ModelTestRequest,
    model_service: ModelService = Depends(get_model_service)
):
    """Test a model with a sample prompt"""
    try:
        response = await model_service.test_model(model_id, request)
        return response
    except Exception as e:
        logger.error(f"Failed to test model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to test model"
        )


@router.get("/presets/list", response_model=List[PresetModel])
async def get_preset_models(
    model_service: ModelService = Depends(get_model_service)
):
    """Get available preset model configurations"""
    try:
        presets = await model_service.get_preset_models()
        return presets
    except Exception as e:
        logger.error(f"Failed to get preset models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve preset models"
        )


@router.post("/presets/{preset_name}/create", response_model=Model)
async def create_from_preset(
    preset_name: str,
    user_id: str,
    custom_config: Optional[dict] = None,
    model_service: ModelService = Depends(get_model_service)
):
    """Create model from preset configuration"""
    try:
        model = await model_service.create_from_preset(preset_name, user_id, custom_config or {})
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Preset model not found"
            )
        return model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create model from preset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create model from preset"
        )


@router.post("/validate", response_model=dict)
async def validate_model_config(
    provider: ModelProvider,
    config: dict,
    model_service: ModelService = Depends(get_model_service)
):
    """Validate model configuration"""
    try:
        validation = await model_service.validate_model_config(provider, config)
        return validation
    except Exception as e:
        logger.error(f"Failed to validate model config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate model configuration"
        )
