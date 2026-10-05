# Method

## Single-frame localization

Let an initial map-frame pose be `T0` and let a candidate correction be
`delta = [dx, dy, dyaw]` in the initial ego frame. The candidate vehicle pose
is `T = T0 compose delta`. Vector-map segments are transformed by `delta`,
sampled in the BEV tensor, and matched to type-aware map embeddings. Scores on
the Cartesian candidate grid become a probability distribution through
softmax. The reported correction is the circular posterior expectation.

## Temporal localization

For frame `t`, the prior posterior is transported by the relative ego motion:

`T_t = T_(t-1) compose u_(t-1,t)`.

The transport is implemented as trilinear forward splatting on the discrete
pose grid followed by a diagonal Gaussian uncertainty blur. The current-frame
likelihood and motion prior are combined with a tempered product of experts:

`log p_t = log p_obs,t + log(alpha p_motion,t + (1-alpha) uniform) + constant`.

`alpha` is a bounded reliability weight. This implementation uses no future
measurements, oracle selection, or adaptive analysis policy.
