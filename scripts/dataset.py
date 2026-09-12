"""
Task 1 — Deterministic appointment dataset generator.

Design choices (stated for grader reproducibility):
  seed              = 42
  size              = 100 records
  category weights  = General Medicine 35 %, Cardiology 20 %,
                      Dermatology 15 %, Pediatrics 15 %, Orthopedics 15 %
  status weights    = Completed 35 %, Scheduled 30 %, Cancelled 15 %,
                      Rescheduled 10 %, No-Show 10 %
  fee range         = ₹300–₹2500 (realistic Indian private-clinic band)
  follow_up prob    = 20 % (target: land between 10 % and 30 %)
  days_since range  = 0–30

Run:  python scripts/dataset.py
"""

import random
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import (
    CATEGORIES,
    CATEGORY_WEIGHTS,
    DATASET_SEED,
    DATASET_SIZE,
    DAYS_SINCE_CREATED_MAX,
    FEE_MAX_INR,
    FEE_MIN_INR,
    FOLLOW_UP_PROBABILITY,
    STATUSES,
    STATUS_WEIGHTS,
)

# ---------------------------------------------------------------------------
# Per-category fee sub-ranges (realistic Indian private-clinic pricing)
# ---------------------------------------------------------------------------
CATEGORY_FEE_RANGE: dict[str, tuple[int, int]] = {
    "General Medicine": (300, 700),
    "Cardiology":       (1000, 2500),
    "Dermatology":      (600, 1500),
    "Pediatrics":       (400, 900),
    "Orthopedics":      (800, 2000),
}


def generate_appointments(
    seed: int = DATASET_SEED,
    size: int = DATASET_SIZE,
) -> list[dict]:
    """Return a deterministic list of appointment records."""
    rng = random.Random(seed)

    records: list[dict] = []
    for i in range(1, size + 1):
        category: str = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
        status: str = rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
        fee_lo, fee_hi = CATEGORY_FEE_RANGE[category]
        # Round to nearest 50 for realistic-looking fees
        raw_fee = rng.randint(fee_lo // 50, fee_hi // 50) * 50
        consultation_fee = max(FEE_MIN_INR, min(raw_fee, FEE_MAX_INR))
        days_since = rng.randint(0, DAYS_SINCE_CREATED_MAX)
        follow_up = rng.random() < FOLLOW_UP_PROBABILITY

        records.append(
            {
                "record_id": f"APT-{i:04d}",
                "category": category,
                "status": status,
                "consultation_fee_inr": consultation_fee,
                "days_since_created": days_since,
                "follow_up_required": follow_up,
            }
        )
    return records


# Exported module-level constant — import this in other modules.
APPOINTMENTS: list[dict] = generate_appointments()


def _validate_and_report(records: list[dict]) -> None:
    """Print validation report to stdout and raise on constraint violations."""
    print("=" * 60)
    print("Practo Appointment Dataset — Validation Report")
    print("=" * 60)
    print(f"Total records : {len(records)}")
    print()

    # --- Category counts ---
    print("Records per category:")
    cat_counts: dict[str, int] = {}
    for r in records:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1
    for cat in CATEGORIES:
        count = cat_counts.get(cat, 0)
        flag = "  [FAIL - need >= 3]" if count < 3 else ""
        print(f"  {cat:<22} {count:>4}{flag}")
    print()

    # --- Status counts ---
    print("Records per status:")
    status_counts: dict[str, int] = {}
    for r in records:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    for st in STATUSES:
        count = status_counts.get(st, 0)
        flag = "  [FAIL - need >= 1]" if count < 1 else ""
        print(f"  {st:<22} {count:>4}{flag}")
    print()

    # --- Follow-up percentage ---
    follow_up_count = sum(1 for r in records if r["follow_up_required"])
    follow_up_pct = follow_up_count / len(records) * 100
    in_range = 10.0 <= follow_up_pct <= 30.0
    flag = "" if in_range else "  [FAIL - must be 10%-30%]"
    print(
        f"follow_up_required=True : {follow_up_count} / {len(records)}"
        f" = {follow_up_pct:.1f} %{flag}"
    )
    print()

    # --- Fee range sanity ---
    fees = [r["consultation_fee_inr"] for r in records]
    print(f"Fee range (INR)         : min={min(fees)}, max={max(fees)}, "
          f"mean={sum(fees)/len(fees):.0f}")
    print()

    # --- First 5 records sample ---
    print("Sample (first 5 records):")
    header = f"{'record_id':<12} {'category':<22} {'status':<14} {'fee':>6}  {'days':>4}  {'follow_up'}"
    print(f"  {header}")
    print("  " + "-" * len(header))
    for r in records[:5]:
        print(
            f"  {r['record_id']:<12} {r['category']:<22} {r['status']:<14}"
            f" {r['consultation_fee_inr']:>6}  {r['days_since_created']:>4}  {r['follow_up_required']}"
        )
    print()

    # --- Hard assertions ---
    errors: list[str] = []
    if len(records) < 40:
        errors.append(f"Dataset too small: {len(records)} < 40")
    for cat in CATEGORIES:
        if cat_counts.get(cat, 0) < 3:
            errors.append(f"Category '{cat}' has fewer than 3 records")
    for st in STATUSES:
        if status_counts.get(st, 0) < 1:
            errors.append(f"Status '{st}' has 0 records")
    if not in_range:
        errors.append(
            f"follow_up_required % = {follow_up_pct:.1f} is outside [10, 30]"
        )

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All validation constraints PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    _validate_and_report(APPOINTMENTS)
