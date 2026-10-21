-- CausalDSP/Safety.lean
-- Formal proofs of mathematical risk invariants, numerical safety,
-- and stoploss ratcheting mechanisms.
-- Standard axioms only, ZERO sorry, ZERO custom axioms.

namespace CausalDSP.Safety

/-! ### 1. Guarded Division (Zero-Division Immunity) -/

/-- Guarded division returning fallback whenever denominator is zero -/
def guarded_div (num den fallback : Nat) : Nat :=
  if den == 0 then fallback else num / den

theorem guarded_div_zero_den (num fallback : Nat) :
    guarded_div num 0 fallback = fallback := by
  dsimp [guarded_div]

theorem guarded_div_positive_den (num den fallback : Nat) (hPos : den > 0) :
    guarded_div num den fallback = num / den := by
  dsimp [guarded_div]
  have hNotZero : (den == 0) = false := by
    cases den with
    | zero => contradiction
    | succ n => rfl
  rw [hNotZero]
  rfl

theorem guarded_div_bound (num den : Nat) (hPos : den > 0) :
    guarded_div num den 0 * den ≤ num := by
  rw [guarded_div_positive_den num den 0 hPos]
  exact Nat.div_mul_le_self num den

/-! ### 2. Bounded Leverage Constraints -/

def clamp_leverage (req minLev maxLev : Nat) : Nat :=
  if req < minLev then minLev
  else if req > maxLev then maxLev
  else req

theorem clamp_leverage_ge_min (req minLev maxLev : Nat) (hOrder : minLev ≤ maxLev) :
    clamp_leverage req minLev maxLev ≥ minLev := by
  dsimp [clamp_leverage]
  split
  · omega
  · split <;> omega

theorem clamp_leverage_le_max (req minLev maxLev : Nat) (hOrder : minLev ≤ maxLev) :
    clamp_leverage req minLev maxLev ≤ maxLev := by
  dsimp [clamp_leverage]
  split
  · omega
  · split <;> omega

/-! ### 3. Monotonic Stop-Loss Ratchet -/

/-- For long positions, stoploss can only monotonically increase (tighten) -/
def ratchet_long (curStop candidateStop : Nat) : Nat :=
  if candidateStop > curStop then candidateStop else curStop

theorem ratchet_long_monotonic (curStop candidateStop : Nat) :
    ratchet_long curStop candidateStop ≥ curStop := by
  dsimp [ratchet_long]
  split
  · omega
  · omega

/-- For short positions, stoploss can only monotonically decrease (tighten) -/
def ratchet_short (curStop candidateStop : Nat) : Nat :=
  if candidateStop < curStop then candidateStop else curStop

theorem ratchet_short_monotonic (curStop candidateStop : Nat) :
    ratchet_short curStop candidateStop ≤ curStop := by
  dsimp [ratchet_short]
  split
  · omega
  · omega

/-! ### 4. Drawdown Safety Invariant -/

/-- Absolute drawdown in fixed-point integer basis points -/
def calc_drawdown (peak current : Nat) : Nat :=
  if current < peak then peak - current else 0

theorem drawdown_non_negative (peak current : Nat) :
    calc_drawdown peak current ≥ 0 := by
  omega

theorem drawdown_zero_at_new_peak (peak current : Nat) (hPeak : current ≥ peak) :
    calc_drawdown peak current = 0 := by
  dsimp [calc_drawdown]
  split
  · omega
  · rfl

/-! ### 5. Bounded Kelly Criterion -/

/-- Scaled Kelly fraction clamped to maximum safety fraction cap (in basis points) -/
def bounded_kelly (rawKelly maxKelly : Nat) : Nat :=
  if rawKelly > maxKelly then maxKelly else rawKelly

theorem kelly_bounded_by_max (rawKelly maxKelly : Nat) :
    bounded_kelly rawKelly maxKelly ≤ maxKelly := by
  dsimp [bounded_kelly]
  split
  · omega
  · omega

end CausalDSP.Safety
