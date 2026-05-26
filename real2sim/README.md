# FR5 Real2Sim Appearance Pipeline

This `real2sim` package is now organized around appearance acquisition, not
single-image mesh reconstruction.

The intended data flow is:

```text
calibrated real RGB
  -> known MuJoCo/CAD geometry
  -> project visible mesh/table/base samples into the RGB image
  -> build partial texture atlas
  -> compute UV visibility mask / coverage
  -> complete missing atlas regions with placeholder inpainting
  -> optionally apply texture/material to FR5 MuJoCo XML
  -> optionally recolor mesh-bound 3DGS assets for GS-Playground rendering
```

Geometry and physics stay in the existing FR5 MuJoCo scene. 3DGS remains a
visual layer only and is not used as collision geometry.

## Main Command

Dry run only. This writes outputs under `real2sim_output/appearance/` and does
not modify `demo/fr5_demo/assets/fr5/mjmodel.xml`.

```bash
cd ~/gs_playground

uv run python -m real2sim.examples.run_fr5_appearance_pipeline \
  --image path/to/real_rgb.png \
  --camera-config demo/fr5_demo/configs/astra_camera.json \
  --config real2sim/examples/fr5_appearance_config.yaml
```

Apply to the FR5 demo after checking the debug output:

```bash
uv run python -m real2sim.examples.run_fr5_appearance_pipeline \
  --image path/to/real_rgb.png \
  --camera-config demo/fr5_demo/configs/astra_camera.json \
  --config real2sim/examples/fr5_appearance_config.yaml \
  --apply-to-fr5
```

Regenerate and recolor mesh-bound 3DGS assets:

```bash
uv run python -m real2sim.examples.run_fr5_appearance_pipeline \
  --image path/to/real_rgb.png \
  --camera-config demo/fr5_demo/configs/astra_camera.json \
  --config real2sim/examples/fr5_appearance_config.yaml \
  --apply-to-fr5 \
  --regenerate-gs
```

## Current Default Scope

The first working scope is static scene appearance:

```yaml
targets:
  - grooved_table
  - fr5_fixed_base
```

The robot arm is intentionally adapter-only for now. It is known geometry, but
proper arm appearance transfer should handle articulated links and self
occlusion more carefully than the table/base pass.

## Outputs

```text
real2sim_output/appearance/
  appearance_result.json
  visual/
    texture_atlas.png
    partial_texture_atlas.png
    uv_visibility_mask.png
  sim/
    mujoco_appearance_assets.yaml
  debug/
    projected_overlay.png
    projection_report.md
```

Check these first:

- `debug/projected_overlay.png`: projected table/base samples should land on
  the real table/base in the RGB photo.
- `visual/uv_visibility_mask.png`: white means observed texture regions.
- `visual/partial_texture_atlas.png`: only directly observed texture.
- `visual/texture_atlas.png`: completed texture atlas after placeholder
  inpainting.

## 3DGS Controls

The FR5 task config stores display density and point size controls:

```json
"fr5_3dgs": {
  "points_per_geom": 12000,
  "scale": 0.00085,
  "opacity": 0.58
}
```

You can override from the command line:

```bash
uv run python -m real2sim.examples.run_fr5_appearance_pipeline \
  --image path/to/real_rgb.png \
  --points-per-geom 20000 \
  --gs-scale 0.00065 \
  --gs-opacity 0.55
```

Smaller `gs-scale` reduces visible point size. Larger `points-per-geom`
increases density.

## Verify In Simulation

After applying:

```bash
uv run python demo/fr5_demo/arm_control.py --check-only

uv run python demo/fr5_demo/arm_control.py \
  --sim-gs-widget \
  --free-view
```

## Dynamic Objects

This package keeps the dynamic-object path separate. Later, when you add a
grasp object, use the same project structure but a separate target adapter:

```text
known/static scene appearance -> table/base/robot visual realism
dynamic object adapter        -> object pose/texture/optional 3DGS visual
MuJoCo collision              -> always simplified geometry, never raw 3DGS
```
