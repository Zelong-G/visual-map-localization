# Method

TemporalMapLoc represents local pose correction as a probability distribution over a discrete SE(2) search grid. The single-frame model produces an observation posterior; the temporal model transports the previous posterior with vehicle motion and fuses both distributions causally.

## 1. Problem formulation

Let the initial map-frame vehicle pose be `T0` and a local correction be

`delta = [dx, dy, dyaw]`.

The corrected vehicle pose is

`T = T0 compose delta`.

The search space is a Cartesian grid over longitudinal translation, lateral translation, and yaw. Keeping the full discrete distribution is useful because local map matching may be multi-modal or uncertain even when the posterior mean is accurate.

## 2. BEV and vector-map representation

The public model expects BEV features with shape `[B, C, H, W]`. Metric-to-grid conversion is explicit: `x` points forward, `y` points left, and yaw is counter-clockwise.

The map is represented as typed line segments. Each segment stores two endpoints plus a semantic/type ID. A type-aware map encoder converts the padded segment batch into map queries, and the cross-modal decoder conditions these queries on BEV features.

The BEV generator itself is intentionally modular. The built-in interface accepts already-computed BEV tensors; an external camera/LiDAR backend can be connected through `BEVFusionAdapter` or another callable adapter without copying that backend into this repository.

## 3. Single-frame hypothesis scoring

For every candidate correction `delta_k`:

1. transform the vector-map segments into the candidate ego alignment;
2. sample BEV evidence along each segment;
3. compare sampled BEV features with the refined map queries;
4. aggregate segment evidence into a scalar pose score.

Candidates are processed in chunks so the full SE(2) grid can be evaluated without materializing all candidate-feature interactions at once.

The scores `s_k` are normalized with softmax:

`p_obs(delta_k) = softmax(s)_k`.

Translation is decoded with the posterior expectation. Yaw uses a circular expectation (`atan2(E[sin yaw], E[cos yaw])`) to avoid the discontinuity at `+pi/-pi`.

## 4. Temporal posterior transport

Assume a posterior `p_(t-1)` from the previous frame and a relative motion estimate `u_(t-1,t)` with covariance `Sigma_u`.

The previous probability mass is first transported through the SE(2) motion model. In the discrete implementation this is performed by trilinear forward splatting onto the current pose grid. Motion uncertainty is then represented by a diagonal Gaussian diffusion with configurable covariance floors.

Conceptually:

`p_motion,t = Transport(p_(t-1), u_(t-1,t), Sigma_u)`.

The transport is strictly causal: only the preceding posterior and the current relative motion are used.

## 5. Robust probabilistic fusion

The observation posterior and transported motion prior are combined in log space. A uniform fallback prevents an over-confident or partially invalid motion prior from suppressing the observation completely:

`p_t(delta) proportional_to p_obs,t(delta) * [alpha * p_motion,t(delta) + (1 - alpha) * U(delta)]`.

`alpha` is bounded to `[0, 1]` and can be scaled by the reliability attached to the relative-motion estimate. A minimum fallback weight keeps the fusion numerically safe.

The fused posterior is normalized and decoded with the same translation/circular-yaw expectation as the single-frame posterior.

## 6. Design properties

- **Causal:** no future frames, oracle selection, or post-hoc best-frame policy is used.
- **Probabilistic:** ambiguity is preserved as a posterior instead of being immediately reduced to a point estimate.
- **Modular:** BEV generation is separated from map matching and temporal fusion.
- **Memory-aware:** SE(2) hypotheses are scored in chunks.
- **Inspectable:** the public API exposes BEV features, map queries, score/posterior tensors, and temporal propagation outputs for analysis.

The compact reference configuration lives in `configs/single_frame.yaml` and `configs/temporal.yaml`.
