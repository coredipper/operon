from typing import Type, TypeVar, Generic, Optional
from pydantic import BaseModel, ValidationError
from ..core.types import FoldedProtein
import json

T = TypeVar('T', bound=BaseModel)

class Chaperone:
    """
    Output Validator: Forces the output to 'fold' into a Schema.
    """
    def fold(self, raw_peptide_chain: str, target_schema: Type[T]) -> FoldedProtein[T]:
        try:
            # 1. Primary Structure Check (JSON)
            json_data = json.loads(raw_peptide_chain)
            
            # 2. Tertiary Structure Check (Schema)
            structure = target_schema.model_validate(json_data)
            return FoldedProtein(valid=True, structure=structure, raw_peptide_chain=raw_peptide_chain)
            
        except (ValidationError, json.JSONDecodeError) as e:
            return FoldedProtein(valid=False, raw_peptide_chain=raw_peptide_chain, error_trace=str(e))
