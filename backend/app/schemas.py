"""Request models for the API."""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Hour(BaseModel):
    """One hourly record. Every measurement is optional (None = not measured this hour)."""
    model_config = ConfigDict(extra="ignore")
    HR: Optional[float] = Field(None, description="Heart rate, bpm")
    O2Sat: Optional[float] = Field(None, description="Oxygen saturation, %")
    Temp: Optional[float] = Field(None, description="Temperature, °C")
    SBP: Optional[float] = Field(None, description="Systolic blood pressure, mmHg")
    MAP: Optional[float] = Field(None, description="Mean arterial pressure, mmHg")
    DBP: Optional[float] = Field(None, description="Diastolic blood pressure, mmHg")
    Resp: Optional[float] = Field(None, description="Respiration rate, breaths/min")
    WBC: Optional[float] = Field(None, description="White blood cells, 10^3/uL")
    Lactate: Optional[float] = Field(None, description="Lactate, mmol/L")
    Creatinine: Optional[float] = Field(None, description="Creatinine, mg/dL")
    Platelets: Optional[float] = Field(None, description="Platelets, 10^3/uL")
    BUN: Optional[float] = Field(None, description="Blood urea nitrogen, mg/dL")
    ICULOS: Optional[float] = Field(None, description="Hours since ICU admission (optional)")
    Age: Optional[float] = Field(None, description="Age in years (optional, overrides request age)")


class PredictRequest(BaseModel):
    hours: List[Hour] = Field(..., min_length=1, max_length=500, description="Hourly records in time order")
    age: Optional[float] = Field(None, ge=0, le=120, description="Patient age in years")
    first_icu_hour: int = Field(1, ge=0, le=100000, description="ICU hour of the first record (ICULOS)")
