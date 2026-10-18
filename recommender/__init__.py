"""VeriTrust-Rec Summer Edition core recommendation package."""

from .pipeline import recommend_from_query, run_core_pipeline, run_user_pipeline
from .schemas import (
    StructuredIntent,
    SummerPipelineResult,
    SummerQueryResult,
    UserFacingResult,
)

__all__ = [
    "StructuredIntent", "SummerPipelineResult", "SummerQueryResult",
    "UserFacingResult", "recommend_from_query", "run_core_pipeline",
    "run_user_pipeline",
]
