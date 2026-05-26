# Projection Report: grooved_table

- status: ok
- samples_total: 313344
- samples_visible: 202198
- visible_ratio: 0.6453
- uv_coverage_ratio: 0.0085
- mask_used: False
- mask_dir_used: `None`
- texture_atlas: `/home/sanchez/gs_playground/real2sim_output/appearance/per_target/grooved_table/texture_atlas.png`
- partial_texture_atlas: `/home/sanchez/gs_playground/real2sim_output/appearance/per_target/grooved_table/partial_texture_atlas.png`
- visibility_mask: `/home/sanchez/gs_playground/real2sim_output/appearance/per_target/grooved_table/uv_visibility_mask.png`
- projected_overlay: `/home/sanchez/gs_playground/real2sim_output/appearance/per_target/grooved_table/projected_overlay.png`

Interpretation:
- If projected_overlay does not align with the real object, fix camera/model pose first.
- If mask_used=false, colors may bleed from neighboring objects.
- This is still a projection fallback, not true mesh UV baking.
