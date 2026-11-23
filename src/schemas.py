"""Contratos de entrada e saída da API."""

from pydantic import BaseModel, Field


class ClienteEntrada(BaseModel):
    """Features brutas esperadas pelo pipeline treinado."""

    tenure: int = Field(
        ..., ge=0, le=100, description="Meses de permanência do cliente.", examples=[12]
    )
    MonthlyCharges: float = Field(
        ..., ge=0, le=200, description="Cobrança mensal em dólares.", examples=[79.85]
    )
    TotalCharges: float = Field(
        ..., ge=0, le=10000, description="Total já cobrado em dólares.", examples=[958.2]
    )
    Contract: str = Field(
        ..., description="Tipo de contrato do cliente.", examples=["Month-to-month"]
    )
    InternetService: str = Field(
        ..., description="Tipo de serviço de internet.", examples=["Fiber optic"]
    )
    OnlineSecurity: str = Field(
        ..., description="Situação do serviço de segurança online.", examples=["No"]
    )
    TechSupport: str = Field(
        ..., description="Situação do suporte técnico.", examples=["No"]
    )
    PaymentMethod: str = Field(
        ..., description="Método de pagamento.", examples=["Electronic check"]
    )


class PredicaoSaida(BaseModel):
    """Resposta de inferência."""

    classe_predita: str = Field(..., description="Classe prevista pelo classificador.")
    probabilidade: float = Field(
        ..., ge=0.0, le=1.0, description="Probabilidade associada à classe prevista."
    )
