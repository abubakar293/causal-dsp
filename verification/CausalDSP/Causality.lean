-- CausalDSP/Causality.lean
-- Mathematical proof of prefix causality and lookahead-freedom for signal transforms
-- Standard Lean 4 axioms only, ZERO sorry, ZERO custom axioms.

namespace CausalDSP.Causality

/-! ### 1. Infinite Temporal Signals -/

/-- A temporal signal parameterized by discrete time Nat -/
def Signal (α : Type) := Nat → α

/-- Agreement of two input series up to time t -/
def AgreementUpTo {α : Type} (x y : Signal α) (t : Nat) : Prop :=
  ∀ s : Nat, s ≤ t → x s = y s

/-- Formal definition of a strictly causal signal transform:
    If two input streams agree on all points up to time t,
    then the transform evaluated at time t must be identical. -/
def IsCausal {α β : Type} (f : Signal α → Signal β) : Prop :=
  ∀ (x y : Signal α) (t : Nat), AgreementUpTo x y t → f x t = f y t

/-! ### Theorem 1: Identity and Past Lookups are Causal -/

theorem id_is_causal (α : Type) : IsCausal (fun (x : Signal α) => x) := by
  intro x y t hAgree
  exact hAgree t (Nat.le_refl t)

theorem const_is_causal {α β : Type} (c : β) : IsCausal (fun (_ : Signal α) (_ : Nat) => c) := by
  intro _ _ _ _
  rfl

/-- Lag operator L^k where L^k(x)[t] = x(t - k) -/
def lag {α : Type} (k : Nat) (x : Signal α) : Signal α :=
  fun t => x (t - k)

theorem lag_is_causal {α : Type} (k : Nat) : IsCausal (@lag α k) := by
  intro x y t hAgree
  dsimp [lag]
  apply hAgree
  exact Nat.sub_le t k

/-! ### Theorem 2: Composition and Pointwise Combination -/

/-- Pointwise combination of two causal operators is causal -/
theorem pointwise_op_is_causal {α β γ : Type}
    (f g : Signal α → Signal β)
    (op : β → β → γ)
    (hf : IsCausal f)
    (hg : IsCausal g) :
    IsCausal (fun x t => op (f x t) (g x t)) := by
  intro x y t hAgree
  dsimp
  rw [hf x y t hAgree, hg x y t hAgree]

/-- Pointwise scalar scaling of a causal operator is causal -/
theorem scale_is_causal {α : Type}
    (f : Signal α → Signal Float)
    (w : Float)
    (hf : IsCausal f) :
    IsCausal (fun x t => w * (f x t)) := by
  intro x y t hAgree
  dsimp
  rw [hf x y t hAgree]

/-- Composition of two causal operators is causal -/
theorem compose_is_causal {α β γ : Type}
    (f : Signal α → Signal β)
    (g : Signal β → Signal γ)
    (hf : IsCausal f)
    (hg : IsCausal g) :
    IsCausal (fun x => g (f x)) := by
  intro x y t hAgree
  apply hg
  intro s hs
  apply hf
  intro u hu
  apply hAgree
  exact Nat.le_trans hu hs

/-! ### Theorem 3: Causal Wavelet Operations (Haar MODWT) -/

/-- Causal Haar Approximation Filter: A1[t] = 0.5 * (x[t] + x[t - 1]) -/
def causal_haar_approx (x : Signal Float) : Signal Float :=
  fun t => 0.5 * (x t + x (t - 1))

/-- Causal Haar Detail Filter: D1[t] = 0.5 * (x[t] - x[t - 1]) -/
def causal_haar_detail (x : Signal Float) : Signal Float :=
  fun t => 0.5 * (x t - x (t - 1))

theorem causal_haar_approx_is_causal : IsCausal causal_haar_approx := by
  intro x y t hAgree
  dsimp [causal_haar_approx]
  have ht : x t = y t := hAgree t (Nat.le_refl t)
  have ht1 : x (t - 1) = y (t - 1) := hAgree (t - 1) (Nat.sub_le t 1)
  rw [ht, ht1]

theorem causal_haar_detail_is_causal : IsCausal causal_haar_detail := by
  intro x y t hAgree
  dsimp [causal_haar_detail]
  have ht : x t = y t := hAgree t (Nat.le_refl t)
  have ht1 : x (t - 1) = y (t - 1) := hAgree (t - 1) (Nat.sub_le t 1)
  rw [ht, ht1]

/-! ### Theorem 4: Finite Impulse Response (FIR) Convolution -/

/-- An FIR filter with coefficients w_k evaluated up to order K:
    y[t] = sum_{k=0}^K w_k * x(t - k) -/
def fir_step (w : List Float) (x : Signal Float) (t : Nat) : Float :=
  let rec loop (weights : List Float) (k : Nat) (acc : Float) : Float :=
    match weights with
    | [] => acc
    | wk :: rest => loop rest (k + 1) (acc + wk * x (t - k))
  loop w 0 0.0

theorem fir_loop_causal (w : List Float) (k : Nat) (acc : Float)
    (x y : Signal Float) (t : Nat) (hAgree : AgreementUpTo x y t) :
    fir_step.loop x t w k acc = fir_step.loop y t w k acc := by
  induction w generalizing k acc with
  | nil => rfl
  | cons wk rest ih =>
    dsimp [fir_step.loop]
    have hxk : x (t - k) = y (t - k) := hAgree (t - k) (Nat.sub_le t k)
    rw [hxk]
    exact ih (k + 1) (acc + wk * y (t - k))

theorem causal_fir_is_causal (w : List Float) :
    IsCausal (fun x t => fir_step w x t) := by
  intro x y t hAgree
  dsimp [fir_step]
  exact fir_loop_causal w 0 0.0 x y t hAgree

/-! ### 2. Finite Lists (Prefix Causality) -/

/-- Formal definition of prefix causality on finite series:
    If two input lists agree on the first t elements (xs.take t = ys.take t),
    then any causal operator applied to them must also agree on the first t elements. -/
def is_prefix_causal {α β : Type} (f : List α → List β) : Prop :=
  ∀ (xs ys : List α) (t : Nat), xs.take t = ys.take t → (f xs).take t = (f ys).take t

theorem id_is_prefix_causal (α : Type) : is_prefix_causal (fun (xs : List α) => xs) := by
  intro xs ys t hAgree
  exact hAgree

theorem map_take {α β : Type} (g : α → β) (t : Nat) (xs : List α) :
    (xs.map g).take t = (xs.take t).map g := by
  induction xs generalizing t with
  | nil =>
    cases t <;> rfl
  | cons x xs ih =>
    cases t with
    | zero => rfl
    | succ t' =>
      dsimp [List.map, List.take]
      rw [ih t']

theorem map_is_prefix_causal {α β : Type} (g : α → β) :
    is_prefix_causal (fun (xs : List α) => xs.map g) := by
  intro xs ys t hAgree
  dsimp
  rw [map_take, map_take]
  rw [hAgree]

end CausalDSP.Causality
