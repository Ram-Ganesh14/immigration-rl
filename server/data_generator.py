"""
Synthetic passenger generator.
Seeded so reset(seed=N) always produces the same episode.
"""

import random
import hashlib
import uuid
from datetime import date, timedelta
from typing import List, Optional
from models.models import (
    PassengerProfile, Document, TravelHistory, BiometricData,
    WatchlistMatch, DocumentType, RiskLevel
)

# ─── Static reference data ────────────────────────────────────────────────────

NATIONALITIES = [
    "Indian", "American", "British", "German", "French", "Brazilian",
    "Nigerian", "Chinese", "Japanese", "Australian", "Canadian",
    "Mexican", "South Korean", "Italian", "Spanish", "Emirati",
    "Saudi", "Pakistani", "Bangladeshi", "Egyptian"
]

DESTINATIONS = [
    "New York", "London", "Dubai", "Singapore", "Paris",
    "Toronto", "Sydney", "Frankfurt", "Tokyo", "Mumbai"
]

FLIGHTS = ["EK203", "BA117", "AA450", "LH701", "SQ321",
           "AI101", "QR572", "EY204", "TK001", "UA889"]

TRAVEL_PURPOSES = ["tourism", "business", "transit", "work", "study"]

FIRST_NAMES = [
    "Amir", "Sofia", "James", "Priya", "Wei", "Fatima", "Carlos",
    "Emma", "Raj", "Yuki", "Chen", "Ahmed", "Maria", "David",
    "Amara", "Lucas", "Nadia", "Omar", "Sarah", "Kenji"
]

LAST_NAMES = [
    "Khan", "Patel", "Smith", "Mueller", "Santos", "Dubois",
    "Okafor", "Zhang", "Tanaka", "Nguyen", "Al-Rashid", "Garcia",
    "Rossi", "Park", "Ibrahim", "Wilson", "Sharma", "Costa",
    "Ivanov", "Nakamura"
]

# Countries that require transit visas for specific layovers (simplified)
TRANSIT_VISA_REQUIRED = {
    "Pakistani": ["UK", "Germany"],
    "Nigerian": ["UK", "France"],
    "Bangladeshi": ["UK", "Germany"],
    "Egyptian": ["UK"],
}

# Visa types that don't match certain travel purposes
VISA_PURPOSE_MISMATCH = {
    "work": ["tourist_visa", "transit_visa"],
    "study": ["tourist_visa", "transit_visa", "business_visa"],
}


# ─── Watchlist (fake names/IDs) ───────────────────────────────────────────────

WATCHLIST = [
    {"name": "Ahmed Al-Rashid", "passport_prefix": "WL", "reason": "fraud_alert"},
    {"name": "Viktor Kozlov",    "passport_prefix": "RU", "reason": "overstay_history"},
    {"name": "Jin Wei",          "passport_prefix": "WL", "reason": "document_forgery"},
    {"name": "Carlos Mendez",    "passport_prefix": "WL", "reason": "criminal_record"},
    {"name": "Priya Sharma",     "passport_prefix": "WL", "reason": "multiple_identity"},
]


def fuzzy_name_match(name1: str, name2: str) -> float:
    """Simple character-overlap similarity."""
    n1, n2 = name1.lower().replace(" ", ""), name2.lower().replace(" ", "")
    if n1 == n2:
        return 1.0
    common = sum(1 for c in n1 if c in n2)
    return common / max(len(n1), len(n2))


def check_watchlist(name: str, passport_number: str) -> WatchlistMatch:
    best_score = 0.0
    best_reason = None
    for entry in WATCHLIST:
        score = fuzzy_name_match(name, entry["name"])
        if passport_number.startswith(entry["passport_prefix"]):
            score = min(1.0, score + 0.3)
        if score > best_score:
            best_score = score
            best_reason = entry["reason"]
    matched = best_score >= 0.75
    return WatchlistMatch(
        matched=matched,
        match_score=round(best_score, 2),
        match_reason=best_reason if matched else None
    )


# ─── Date helpers ─────────────────────────────────────────────────────────────

def today_str() -> str:
    return date.today().isoformat()


def future_date(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def past_date(days: int) -> str:
    return (date.today() - timedelta(days=days)).isoformat()


# ─── Passenger factory ────────────────────────────────────────────────────────

class PassengerGenerator:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def _random_name(self) -> str:
        return f"{self.rng.choice(FIRST_NAMES)} {self.rng.choice(LAST_NAMES)}"

    def _random_passport_number(self, prefix: str = "") -> str:
        prefix = prefix or self.rng.choice(["PA", "PB", "PC", "PD", "PE"])
        return prefix + str(self.rng.randint(1000000, 9999999))

    def _random_dob(self) -> str:
        age = self.rng.randint(18, 70)
        return past_date(age * 365)

    def generate_clean_passenger(self) -> PassengerProfile:
        """A passenger with valid documents and no issues."""
        name = self._random_name()
        nationality = self.rng.choice(NATIONALITIES)
        destination = self.rng.choice(DESTINATIONS)
        purpose = self.rng.choice(["tourism", "business"])
        passport_num = self._random_passport_number()

        passport = Document(
            doc_type=DocumentType.PASSPORT,
            doc_number=passport_num,
            issuing_country=nationality,
            expiry_date=future_date(self.rng.randint(200, 1800)),
            issue_date=past_date(self.rng.randint(100, 3000)),
            name_on_doc=name,
            is_authentic=True,
        )
        visa = Document(
            doc_type=DocumentType.VISA,
            doc_number=f"V{self.rng.randint(100000, 999999)}",
            issuing_country=destination.split()[0],
            expiry_date=future_date(self.rng.randint(30, 365)),
            name_on_doc=name,
            is_authentic=True,
            visa_type="tourist_visa" if purpose == "tourism" else "business_visa",
            visa_entries="multiple",
            destination_countries=[destination],
        )
        boarding = Document(
            doc_type=DocumentType.BOARDING_PASS,
            doc_number=f"BP{self.rng.randint(10000, 99999)}",
            issuing_country="",
            name_on_doc=name,
            is_authentic=True,
        )
        biometrics = BiometricData(
            face_match_score=round(self.rng.uniform(0.91, 0.99), 2),
            fingerprint_match=True,
        )
        wl = check_watchlist(name, passport_num)

        return PassengerProfile(
            passenger_id=str(uuid.uuid4())[:8],
            name=name,
            nationality=nationality,
            date_of_birth=self._random_dob(),
            gender=self.rng.choice(["M", "F"]),
            destination=destination,
            flight_number=self.rng.choice(FLIGHTS),
            travel_purpose=purpose,
            documents=[passport, visa, boarding],
            biometrics=biometrics,
            watchlist_match=wl,
            ground_truth_decision="clear",
            ground_truth_reason="All documents valid, no flags.",
            risk_level=RiskLevel.CLEAN,
            flags=[],
        )

    def generate_expired_passport(self) -> PassengerProfile:
        """Passport expired — must be denied."""
        p = self.generate_clean_passenger()
        p.documents[0].expiry_date = past_date(self.rng.randint(1, 180))
        p.documents[0].is_authentic = True
        p.documents[0].anomaly = "expired_passport"
        p.ground_truth_decision = "deny"
        p.ground_truth_reason = "Passport expired."
        p.risk_level = RiskLevel.LOW
        p.flags = ["PASSPORT_EXPIRED"]
        return p

    def generate_name_mismatch(self) -> PassengerProfile:
        """Name on passport doesn't match boarding pass — hold for verification."""
        p = self.generate_clean_passenger()
        alt_name = self._random_name()
        p.documents[2].name_on_doc = alt_name  # boarding pass has different name
        p.documents[2].anomaly = "name_mismatch"
        p.ground_truth_decision = "hold"
        p.ground_truth_reason = "Name on boarding pass doesn't match passport."
        p.risk_level = RiskLevel.MEDIUM
        p.flags = ["NAME_MISMATCH_BOARDING_PASS"]
        return p

    def generate_watchlist_hit(self) -> PassengerProfile:
        """Passenger matches watchlist — must escalate."""
        p = self.generate_clean_passenger()
        entry = self.rng.choice(WATCHLIST)
        p.name = entry["name"]
        p.documents[0].name_on_doc = entry["name"]
        p.documents[0].doc_number = entry["passport_prefix"] + str(self.rng.randint(1000000, 9999999))
        p.watchlist_match = WatchlistMatch(
            matched=True,
            match_score=0.92,
            match_reason=entry["reason"]
        )
        p.ground_truth_decision = "escalate"
        p.ground_truth_reason = f"Watchlist match: {entry['reason']}"
        p.risk_level = RiskLevel.HIGH
        p.flags = ["WATCHLIST_MATCH"]
        return p

    def generate_visa_purpose_mismatch(self) -> PassengerProfile:
        """Tourist visa but travelling for work — deny."""
        p = self.generate_clean_passenger()
        p.travel_purpose = "work"
        p.documents[1].visa_type = "tourist_visa"
        p.documents[1].anomaly = "visa_purpose_mismatch"
        p.ground_truth_decision = "deny"
        p.ground_truth_reason = "Tourist visa invalid for work travel."
        p.risk_level = RiskLevel.MEDIUM
        p.flags = ["VISA_PURPOSE_MISMATCH"]
        return p

    def generate_overstay_history(self) -> PassengerProfile:
        """Passenger has prior overstay history — hold for secondary."""
        p = self.generate_clean_passenger()
        p.travel_history = [
            TravelHistory(
                country="Germany",
                entry_date=past_date(400),
                exit_date=past_date(200),  # stayed 200 days on 90-day visa
                duration_days=200,
                visa_compliant=False
            )
        ]
        p.ground_truth_decision = "hold"
        p.ground_truth_reason = "Prior overstay of 110 days in Germany."
        p.risk_level = RiskLevel.MEDIUM
        p.flags = ["OVERSTAY_HISTORY"]
        return p

    def generate_low_biometric_match(self) -> PassengerProfile:
        """Face doesn't match passport photo — escalate."""
        p = self.generate_clean_passenger()
        p.biometrics.face_match_score = round(self.rng.uniform(0.45, 0.62), 2)
        p.biometrics.fingerprint_match = False
        p.ground_truth_decision = "escalate"
        p.ground_truth_reason = "Biometric mismatch — possible identity fraud."
        p.risk_level = RiskLevel.HIGH
        p.flags = ["BIOMETRIC_MISMATCH"]
        return p

    def generate_emergency_travel_doc(self) -> PassengerProfile:
        """
        Edge case (Task 3): Expired passport BUT valid emergency travel document.
        Agent must correctly use the emergency doc, not deny based on passport.
        """
        p = self.generate_clean_passenger()
        p.documents[0].expiry_date = past_date(10)
        p.documents[0].anomaly = "expired_passport"
        p.special_circumstances = ["emergency_travel_doc_holder"]

        etd = Document(
            doc_type=DocumentType.EMERGENCY_TRAVEL_DOC,
            doc_number=f"ETD{self.rng.randint(10000, 99999)}",
            issuing_country=p.nationality,
            expiry_date=future_date(30),
            name_on_doc=p.name,
            is_authentic=True,
        )
        p.documents.append(etd)
        p.ground_truth_decision = "clear"
        p.ground_truth_reason = "Expired passport overridden by valid emergency travel document."
        p.risk_level = RiskLevel.LOW
        p.flags = ["PASSPORT_EXPIRED", "EMERGENCY_TRAVEL_DOC_PRESENT"]
        return p

    def generate_unaccompanied_minor(self) -> PassengerProfile:
        """Minor travelling alone — must escalate regardless of docs."""
        p = self.generate_clean_passenger()
        age = self.rng.randint(8, 15)
        p.date_of_birth = past_date(age * 365)
        p.special_circumstances = ["unaccompanied_minor"]
        p.ground_truth_decision = "escalate"
        p.ground_truth_reason = "Unaccompanied minor — child protection protocol required."
        p.risk_level = RiskLevel.MEDIUM
        p.flags = ["UNACCOMPANIED_MINOR"]
        return p

    def generate_forged_document(self) -> PassengerProfile:
        """Subtly forged passport — hard to detect without close inspection."""
        p = self.generate_clean_passenger()
        p.documents[0].is_authentic = False
        p.documents[0].anomaly = "forged_hologram"
        # Biometric still matches (forger used real person's photo)
        p.biometrics.face_match_score = round(self.rng.uniform(0.88, 0.94), 2)
        # Flag is subtle — agent needs to request document inspection
        p.ground_truth_decision = "escalate"
        p.ground_truth_reason = "Forged passport hologram detected on close inspection."
        p.risk_level = RiskLevel.CRITICAL
        p.flags = ["DOCUMENT_INSPECTION_REQUIRED"]  # subtle flag — not outright FORGED
        return p

    def build_queue(self, n: int, difficulty: str = "easy") -> List[PassengerProfile]:
        """
        Build a queue of n passengers based on difficulty.
        easy: mostly clean, 1 obvious issue
        medium: 30% flagged, mixed issues
        hard: 40% flagged, includes edge cases
        """
        queue = []

        if difficulty == "easy":
            clean_count = max(1, n - 1)
            for _ in range(clean_count):
                queue.append(self.generate_clean_passenger())
            queue.append(self.generate_expired_passport())

        elif difficulty == "medium":
            generators = [
                self.generate_clean_passenger,
                self.generate_clean_passenger,
                self.generate_expired_passport,
                self.generate_name_mismatch,
                self.generate_visa_purpose_mismatch,
                self.generate_overstay_history,
            ]
            for i in range(n):
                gen = generators[i % len(generators)]
                queue.append(gen())

        elif difficulty == "hard":
            generators = [
                self.generate_clean_passenger,
                self.generate_clean_passenger,
                self.generate_watchlist_hit,
                self.generate_low_biometric_match,
                self.generate_emergency_travel_doc,
                self.generate_forged_document,
                self.generate_unaccompanied_minor,
                self.generate_overstay_history,
                self.generate_visa_purpose_mismatch,
                self.generate_name_mismatch,
            ]
            for i in range(n):
                gen = generators[i % len(generators)]
                queue.append(gen())

        self.rng.shuffle(queue)
        return queue
