import CausalDSP.Causality
import CausalDSP.Safety

-- Causality theorems
#print axioms CausalDSP.Causality.id_is_causal
#print axioms CausalDSP.Causality.const_is_causal
#print axioms CausalDSP.Causality.lag_is_causal
#print axioms CausalDSP.Causality.pointwise_op_is_causal
#print axioms CausalDSP.Causality.scale_is_causal
#print axioms CausalDSP.Causality.compose_is_causal
#print axioms CausalDSP.Causality.causal_haar_approx_is_causal
#print axioms CausalDSP.Causality.causal_haar_detail_is_causal
#print axioms CausalDSP.Causality.causal_fir_is_causal
#print axioms CausalDSP.Causality.id_is_prefix_causal
#print axioms CausalDSP.Causality.map_is_prefix_causal

-- Safety invariants
#print axioms CausalDSP.Safety.guarded_div_zero_den
#print axioms CausalDSP.Safety.guarded_div_positive_den
#print axioms CausalDSP.Safety.guarded_div_bound
#print axioms CausalDSP.Safety.clamp_leverage_ge_min
#print axioms CausalDSP.Safety.clamp_leverage_le_max
#print axioms CausalDSP.Safety.ratchet_long_monotonic
#print axioms CausalDSP.Safety.ratchet_short_monotonic
#print axioms CausalDSP.Safety.drawdown_non_negative
#print axioms CausalDSP.Safety.drawdown_zero_at_new_peak
#print axioms CausalDSP.Safety.kelly_bounded_by_max
