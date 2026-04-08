from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    CLEAR             = "clear"
    HOLD              = "hold"
    DENY              = "deny"
    ESCALATE          = "escalate"
    REQUEST_DOCUMENT  = "request_document"
    QUERY_INTERPOL    = "query_interpol"
    VERIFY_BIOMETRICS = "verify_biometrics"


class DocumentType(str, Enum):
    PASSPORT             = "passport"
    VISA                 = "visa"
    BOARDING_PASS        = "boarding_pass"
    TRAVEL_PERMIT        = "travel_permit"
    EMERGENCY_TRAVEL_DOC = "emergency_travel_doc"
    RESIDENCE_PERMIT     = "residence_permit"


class PassengerStatus(str, Enum):
    WAITING       = "waiting"
    IN_PROCESSING = "in_processing"
    CLEARED       = "cleared"
    HELD          = "held"
    DENIED        = "denied"
    ESCALATED     = "escalated"


class RiskLevel(str, Enum):
    CLEAN    = "clean"
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# ─── Sub-models ───────────────────────────────────────────────────────────────

class Document(BaseModel):
    doc_type: DocumentType
    doc_number: str
    issuing_country: str
    expiry_date: Optional[str] = None
    issue_date: Optional[str] = None
    name_on_doc: str
    anomaly: Optional[str] = None
    visa_type: Optional[str] = None
    visa_entries: Optional[str] = None
    destination_countries: Optional[List[str]] = None


class TravelHistory(BaseModel):
    country: str
    entry_date: str
    exit_date: Optional[str] = None
    duration_days: Optional[int] = None
    visa_compliant: bool = True


class PassengerProfile(BaseModel):
    """
    Sanitised passenger view for the agent.
    Biometrics and watchlist results are HIDDEN until queried via
    verify_biometrics / query_interpol actions.
    """
    passenger_id: str
    name: str
    nationality: str
    date_of_birth: str
    gender: str
    destination: str
    flight_number: str
    travel_purpose: str
    documents: List[Document]
    travel_history: List[TravelHistory] = []
    special_circumstances: List[str] = []
    flags: List[str] = []
    queried_biometrics: Optional[Dict[str, Any]] = None
    queried_watchlist: Optional[Dict[str, Any]] = None


class _PassengerInternalData(BaseModel):
    """Ground-truth data — never sent to agent. Only used by environment internally."""
    passenger_id: str
    is_authentic: bool = True
    face_match_score: float = 0.95
    fingerprint_match: bool = True
    watchlist_matched: bool = False
    watchlist_score: float = 0.0
    watchlist_reason: Optional[str] = None
    ground_truth_decision: str
    ground_truth_reason: str
    risk_level: RiskLevel
    nationality: str
    gender: str


class ImmigrationObservation(BaseModel):
    """What the agent sees at each step."""
    current_passenger: Optional[PassengerProfile] = None
    queue_length: int = 0
    queue_summary: List[Dict[str, Any]] = []
    time_remaining: int = 0
    step_count: int = 0
    max_steps: int = 100
    processing_result: str = ""
    auto_flags: List[str] = []
    episode_id: str = ""
    task_id: str = ""
    documents_requested: List[str] = []
    secondary_screening_available: bool = True
    fairness_score: float = 1.0
    api_calls_used: List[str] = []
    api_calls_remaining: int = 4


class ImmigrationAction(BaseModel):
    """What the agent can do."""
    action_type: ActionType
    passenger_id: str
    reason: str = ""
    document_requested: Optional[str] = None


class ImmigrationReward(BaseModel):
    """Detailed reward breakdown."""
    total: float
    decision_reward: float = 0.0
    speed_bonus: float = 0.0
    api_cost: float = 0.0
    fairness_penalty: float = 0.0
    loop_penalty: float = 0.0
    breakdown: Dict[str, float] = {}
    explanation: str = ""


class EpisodeState(BaseModel):
    """Full internal state — returned by state() endpoint."""
    episode_id: str
    task_id: str
    seed: int
    step_count: int
    max_steps: int
    time_elapsed: int
    time_limit: int
    passengers_processed: int
    passengers_total: int
    cumulative_reward: float
    action_history: List[Dict[str, Any]] = []
    decision_log: List[Dict[str, Any]] = []
    current_passenger_id: Optional[str] = None
    done: bool = False
    fairness_tracker: Dict[str, List[str]] = {}
    demographic_log: List[Dict[str, Any]] = []


class StepResult(BaseModel):
    """Return value of step()."""
    observation: ImmigrationObservation
    reward: ImmigrationReward
    done: bool
    info: Dict[str, Any] = {}


class ResetResult(BaseModel):
    """Return value of reset()."""
    observation: ImmigrationObservation
    episode_id: str
    task_id: str
