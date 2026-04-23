from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from server.config import Outcome, MentalStatus

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    class Action(BaseModel):
        pass
    class Observation(BaseModel):
        done: bool = False
        reward: float = 0.0
    class State(BaseModel):
        episode_id: Optional[str] = None
        step_count: int = 0


class PatientState(BaseModel):
    """Full internal state for one patient. Agents see filtered views of this."""
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    patient_id: str
    bed_number: int
    age: int
    admission_reason: str

    heart_rate: float
    systolic_bp: float
    respiratory_rate: float
    temperature: float
    oxygen_saturation: float
    mental_status: MentalStatus = MentalStatus.ALERT

    lactate: Optional[float] = None
    wbc: Optional[float] = None
    procalcitonin: Optional[float] = None
    creatinine: Optional[float] = None
    blood_culture_result: Optional[str] = None

    pending_labs: Dict[str, int] = Field(default_factory=dict)

    current_medications: List[str] = Field(default_factory=list)
    immunocompromised: bool = False

    infection_present: bool = False
    infection_severity: float = 0.0
    sepsis_onset_tick: Optional[int] = None
    is_false_alarm_patient: bool = False

    antibiotics_administered: Optional[str] = None
    antibiotic_tick: Optional[int] = None
    icu_admitted: bool = False
    outcome: Outcome = Outcome.STABLE
    critical_ticks: int = 0


class AgentFlag(BaseModel):
    """A flag raised by one agent, visible to others this tick."""
    source_role: str
    patient_id: str
    flag_type: str
    urgency: Literal["routine", "urgent", "critical"] = "routine"
    rationale: str = ""
    tick: int


class NurseObservation(Observation):
    tick: int
    max_ticks: int
    task_name: str
    assigned_patient_ids: List[str]
    patient_vitals: List[Dict[str, Any]]
    pharmacist_flags_this_tick: List[AgentFlag]
    lab_flags_this_tick: List[AgentFlag]
    physician_trust: float
    last_action_result: Optional[str] = None
    cumulative_reward: float = 0.0
    normalized_score: Optional[float] = None


class LabObservation(Observation):
    tick: int
    max_ticks: int
    task_name: str
    lab_results: List[Dict[str, Any]]
    pending_labs: List[Dict[str, Any]]
    last_action_result: Optional[str] = None
    cumulative_reward: float = 0.0
    normalized_score: Optional[float] = None


class PharmacistObservation(Observation):
    tick: int
    max_ticks: int
    task_name: str
    patient_medications: List[Dict[str, Any]]
    antibiogram: Dict[str, float]
    lab_flags_this_tick: List[AgentFlag]
    last_action_result: Optional[str] = None
    cumulative_reward: float = 0.0
    normalized_score: Optional[float] = None


class PhysicianObservation(Observation):
    tick: int
    max_ticks: int
    task_name: str
    nurse_escalations_this_tick: List[AgentFlag]
    lab_flags_this_tick: List[AgentFlag]
    pharmacist_flags_this_tick: List[AgentFlag]
    known_patient_summaries: List[Dict[str, Any]]
    physician_trust: float
    last_action_result: Optional[str] = None
    cumulative_reward: float = 0.0
    normalized_score: Optional[float] = None


class NurseAction(Action):
    operation: Literal[
        "escalate_to_physician", "request_lab_test",
        "administer_medication", "flag_concern", "noop",
    ]
    patient_id: Optional[str] = None
    urgency: Optional[Literal["routine", "urgent", "critical"]] = None
    test_type: Optional[str] = None
    rationale: str = ""


class LabAction(Action):
    operation: Literal[
        "release_result", "flag_critical",
        "recommend_followup_test", "noop",
    ]
    patient_id: Optional[str] = None
    test: Optional[str] = None
    reason: str = ""


class PharmacistAction(Action):
    operation: Literal[
        "flag_interaction", "flag_immunosuppression",
        "recommend_antibiotic", "check_dosing", "noop",
    ]
    patient_id: Optional[str] = None
    drug: Optional[str] = None
    rationale: str = ""


class PhysicianAction(Action):
    operation: Literal[
        "order_antibiotics", "order_lab_test",
        "admit_to_icu", "request_consult", "do_nothing",
    ]
    patient_id: Optional[str] = None
    drug: Optional[str] = None
    test: Optional[str] = None
    specialty: Optional[str] = None


class StepRequest(BaseModel):
    """Bundle of 4 parallel actions submitted to /step."""
    nurse: NurseAction
    lab: LabAction
    pharmacist: PharmacistAction
    physician: PhysicianAction


class SepsisState(State):
    task_name: str
    task_description: str
    tick: int
    max_ticks: int
    patients: List[Dict[str, Any]]
    active_flags: List[AgentFlag]
    physician_trust: float
    cumulative_rewards: Dict[str, float]
    total_team_reward: float
    normalized_score: Optional[float] = None
    patients_treated_in_time: int = 0
    patients_with_sepsis: int = 0
    false_alarm_count: int = 0
    lives_saved: int = 0
    lives_lost: int = 0
    total_escalations: int = 0
    correct_escalations: int = 0
