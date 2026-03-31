from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import date


# ─── Enums ────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    CLEAR = "clear"
    HOLD = "hold"
    DENY = "deny"
    REQUEST_DOCUMENT = "request_document"
    ESCALATE = "escalate"
    ASSIGN_COUNTER = "assign_counter"  # used in multi-counter mode


class DocumentType(str, Enum):
    PASSPORT = "passport"
    VISA = "visa"
    BOARDING_PASS = "boarding_pass"
    TRAVEL_PERMIT = "travel_permit"
    EMERGENCY_TRAVEL_DOC = "emergency_travel_doc"
    RESIDENCE_PERMIT = "residence_permit"


class PassengerStatus(str, Enum):
    WAITING = "waiting"
    IN_PROCESSING = "in_processing"
    CLEARED = "cleared"
    HELD = "held"
    DENIED = "denied"
    ESCALATED = "escalated"


class RiskLevel(str, Enum):
    CLEAN = "clean"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ─── Sub-models ───────────────────────────────────────────────────────────────

class Document(BaseModel):
    doc_type: DocumentType
    doc_number: str
    issuing_country: str
    expiry_date: Optional[str] = None           # ISO format YYYY-MM-DD
    issue_date: Optional[str] = None
    name_on_doc: str
    is_authentic: bool = True                   # ground truth (hidden from agent)
    anomaly: Optional[str] = None               # e.g. "name_mismatch", "expired"
    visa_type: Optional[str] = None             # tourist, work, student, transit
    visa_entries: Optional[str] = None          # single, multiple
    destination_countries: Optional[List[str]] = None


class TravelHistory(BaseModel):
    country: str
    entry_date: str
    exit_date: Optional[str] = None
    duration_days: Optional[int] = None
    visa_compliant: bool = True


class BiometricData(BaseModel):
    face_match_score: float = Field(ge=0.0, le=1.0)   # 1.0 = perfect match
    fingerprint_match: bool = True
    iris_scan_match: Optional[bool] = None


class WatchlistMatch(BaseModel):
    matched: bool = False
    match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    match_reason: Optional[str] = None


class PassengerProfile(BaseModel):
    passenger_id: str
    name: str
    nationality: str
    date_of_birth: str
    gender: str
    destination: str
    flight_number: str
    travel_purpose: str                         # tourism, business, transit, work, study
    documents: List[Document]
    travel_history: List[TravelHistory] = []
    biometrics: BiometricData
    watchlist_match: WatchlistMatch
    special_circumstances: List[str] = []      # e.g. "medical_emergency", "unaccompanied_minor"
    ground_truth_decision: str                  # clear / hold / deny / escalate
    ground_truth_reason: str
    risk_level: RiskLevel
    flags: List[str] = []                       # auto-detected anomalies shown to agent


# ─── OpenEnv Core Models ──────────────────────────────────────────────────────

class ImmigrationObservation(BaseModel):
    """What the agent sees at each step."""
    current_passenger: Optional[PassengerProfile] = None
    queue_length: int = 0
    queue_summary: List[Dict[str, Any]] = []    # brief info on waiting passengers
    time_remaining: int = 0                     # seconds left in episode
    step_count: int = 0
    max_steps: int = 100
    processing_result: str = ""                 # feedback from last action
    auto_flags: List[str] = []                  # system-detected anomalies
    episode_id: str = ""
    task_id: str = ""
    documents_requested: List[str] = []         # docs agent has already requested
    secondary_screening_available: bool = True
    fairness_score: float = 1.0                 # drops if agent is inconsistent


class ImmigrationAction(BaseModel):
    """What the agent can do."""
    action_type: ActionType
    passenger_id: str
    reason: str = ""                            # agent's stated justification
    document_requested: Optional[str] = None    # if action_type = request_document
    counter_id: Optional[int] = None            # if action_type = assign_counter


class ImmigrationReward(BaseModel):
    """Detailed reward breakdown."""
    total: float
    decision_reward: float = 0.0
    speed_bonus: float = 0.0
    escalation_quality: float = 0.0
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
    fairness_tracker: Dict[str, List[str]] = {}  # profile_hash -> [decisions]


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
