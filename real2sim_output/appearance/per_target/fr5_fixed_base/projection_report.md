# Projection Report: fr5_fixed_base

- status: ok
- samples_total: 6144
- samples_visible: 5608
- visible_ratio: 0.9128
- uv_coverage_ratio: 0.0114
- mask_used: False
- mask_dir_used: `None`
- texture_atlas: `/home/sanchez/gs_playground/real2sim_output/appearance/per_target/fr5_fixed_base/texture_atlas.png`
- partial_texture_atlas: `/home/sanchez/gs_playground/real2sim_output/appearance/per_target/fr5_fixed_base/partial_texture_atlas.png`
- visibility_mask: `/home/sanchez/gs_playground/real2sim_output/appearance/per_target/fr5_fixed_base/uv_visibility_mask.png`
- projected_overlay: `/home/sanchez/gs_playground/real2sim_output/appearance/per_target/fr5_fixed_base/projected_overlay.png`

Interpretation:
- If projected_overlay does not align with the real object, fix camera/model pose first.
- If mask_used=false, colors may bleed from neighboring objects.
- This is still a projection fallback, not true mesh UV baking.
