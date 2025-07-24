"""Configuration for Logistic Regression model using Pydantic."""

from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class LogisticRegressionConfig(BaseModel):
    """Configuration for Logistic Regression model following InSilicoVA pattern.

    This configuration class defines all parameters needed for the Logistic Regression
    model, including regularization options, solver selection, and class balancing.
    """

    # Regularization parameters
    penalty: Optional[Literal["l1", "l2", "elasticnet", None]] = Field(
        default="l2", description="Regularization penalty type"
    )
    C: float = Field(
        default=1.0, gt=0, description="Inverse of regularization strength"
    )
    l1_ratio: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="ElasticNet mixing parameter (0=L2, 1=L1)",
    )

    # Solver parameters
    solver: Literal[
        "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
    ] = Field(
        default="saga",
        description="Algorithm to use in the optimization problem",
    )
    max_iter: int = Field(
        default=100, ge=1, description="Maximum number of iterations"
    )
    tol: float = Field(
        default=1e-4, gt=0, description="Tolerance for stopping criteria"
    )

    # Multi-class parameters
    multi_class: Literal["auto", "ovr", "multinomial"] = Field(
        default="auto",
        description="Multi-class classification strategy",
    )

    # Class imbalance handling
    class_weight: Union[str, Dict[int, float], None] = Field(
        default="balanced",
        description="Weights associated with classes",
    )

    # Fitting parameters
    fit_intercept: bool = Field(
        default=True, description="Whether to calculate the intercept"
    )
    intercept_scaling: float = Field(
        default=1.0,
        gt=0,
        description="Scaling of the intercept (only for liblinear solver)",
    )

    # Performance parameters
    n_jobs: Optional[int] = Field(
        default=None,
        description="Number of CPU cores used during cross-validation",
    )
    random_state: int = Field(
        default=42, description="Random seed for reproducibility"
    )
    verbose: int = Field(default=0, ge=0, description="Verbosity level")

    # Convergence parameters
    warm_start: bool = Field(
        default=False,
        description="Reuse solution of previous call to fit as initialization",
    )
    max_fun: int = Field(
        default=15000,
        ge=1,
        description="Maximum number of function evaluations (for L-BFGS)",
    )

    @field_validator("solver")
    def validate_solver_penalty_compatibility(cls, solver: str, info: Any) -> str:
        """Validate compatibility between solver and penalty.

        Different solvers support different penalty types:
        - 'liblinear': supports L1, L2
        - 'lbfgs', 'newton-cg', 'newton-cholesky', 'sag': support L2, None
        - 'saga': supports all penalties (L1, L2, elasticnet, None)
        """
        penalty = info.data.get("penalty", "l2")

        if penalty == "l1":
            if solver not in ["liblinear", "saga"]:
                raise ValueError(
                    f"Solver '{solver}' does not support L1 penalty. "
                    "Use 'liblinear' or 'saga' solver."
                )
        elif penalty == "elasticnet":
            if solver != "saga":
                raise ValueError(
                    f"Solver '{solver}' does not support ElasticNet penalty. "
                    "Only 'saga' solver supports ElasticNet."
                )
        elif penalty is None:
            if solver == "liblinear":
                raise ValueError(
                    "Solver 'liblinear' does not support no penalty (penalty=None)."
                )

        return solver

    @field_validator("multi_class")
    def validate_multi_class_solver_compatibility(cls, multi_class: str, info: Any) -> str:
        """Validate compatibility between multi_class and solver.

        - 'liblinear' does not support 'multinomial' multi_class
        """
        solver = info.data.get("solver", "saga")

        if multi_class == "multinomial" and solver == "liblinear":
            raise ValueError(
                "Solver 'liblinear' does not support 'multinomial' multi_class. "
                "It will use 'ovr' (one-vs-rest) instead."
            )

        return multi_class

    @field_validator("l1_ratio")
    def validate_l1_ratio(cls, l1_ratio: Optional[float], info: Any) -> Optional[float]:
        """Validate l1_ratio is only set when penalty='elasticnet'."""
        penalty = info.data.get("penalty", "l2")

        if l1_ratio is not None and penalty != "elasticnet":
            raise ValueError(
                "l1_ratio is only used when penalty='elasticnet'. "
                f"Current penalty is '{penalty}'."
            )

        return l1_ratio
    
    @model_validator(mode='after')
    def validate_elasticnet_l1_ratio(self) -> "LogisticRegressionConfig":
        """Validate that elasticnet penalty has l1_ratio specified."""
        if self.penalty == "elasticnet" and self.l1_ratio is None:
            raise ValueError(
                "l1_ratio must be specified when penalty='elasticnet'. "
                "Set a value between 0 and 1."
            )
        return self
    

    @field_validator("class_weight")
    def validate_class_weight(
        cls, v: Union[str, Dict[int, float], None]
    ) -> Union[str, Dict[int, float], None]:
        """Validate class_weight parameter."""
        if v is None:
            return v

        if isinstance(v, str):
            if v != "balanced":
                raise ValueError(
                    "class_weight string must be 'balanced' or None"
                )
        elif isinstance(v, dict):
            # Ensure all values are positive
            for class_label, weight in v.items():
                if weight <= 0:
                    raise ValueError(
                        f"Class weight for {class_label} must be positive"
                    )

        return v

    @field_validator("n_jobs")
    def validate_n_jobs(cls, n_jobs: Optional[int]) -> Optional[int]:
        """Validate n_jobs parameter."""
        if n_jobs is not None and n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        return n_jobs

    model_config = {"validate_assignment": True, "extra": "forbid"}