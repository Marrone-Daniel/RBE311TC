# Mesh-Conditioned Appearance Projection Report

- image: `demo/fr5_demo/data/astra_captures/astra_rgb_00000.png`
- camera_config: `demo/fr5_demo/configs/astra_camera.json`
- model_xml: `/home/sanchez/gs_playground/demo/fr5_demo/assets/fr5/mjmodel.xml`
- targets: ['grooved_table', 'fr5_fixed_base']
- total_samples: 319488
- visible_samples: 207806
- visible_ratio: 0.6504
- uv_coverage_ratio: 0.0199
- texture_atlas: `/home/sanchez/gs_playground/demo/fr5_demo/assets/fr5/textures/fr5_static_appearance_atlas.png`
- partial_texture_atlas: `/home/sanchez/gs_playground/real2sim_output/appearance/visual/partial_texture_atlas.png`
- visibility_mask: `/home/sanchez/gs_playground/real2sim_output/appearance/visual/uv_visibility_mask.png`
- projected_overlay: `/home/sanchez/gs_playground/real2sim_output/appearance/debug/projected_overlay.png`
- apply_to_fr5: False

Important notes:
- This is a safer per-target debugging version of the previous global projection pipeline.
- The old version projected all sampled box geometry into one atlas, which can easily cause color bleeding.
- Manual masks are supported via `<image_dir>/masks/<target>.png` or `<output_dir>/masks/<target>.png`.
- If overlay alignment is bad, do not train with this output. Fix camera extrinsic / object pose first.
- True high-quality result requires real mesh UV baking or vertex-color export for mesh geoms.

Per-target reports:
- grooved_table: {'target': 'grooved_table', 'status': 'ok', 'samples_total': 313344, 'samples_visible': 202198, 'visible_ratio': 0.6452907986111112, 'uv_coverage_ratio': 0.008527278900146484, 'mask_used': False, 'mask_dir_used': None, 'texture_atlas': '/home/sanchez/gs_playground/real2sim_output/appearance/per_target/grooved_table/texture_atlas.png', 'partial_texture_atlas': '/home/sanchez/gs_playground/real2sim_output/appearance/per_target/grooved_table/partial_texture_atlas.png', 'visibility_mask': '/home/sanchez/gs_playground/real2sim_output/appearance/per_target/grooved_table/uv_visibility_mask.png', 'projected_overlay': '/home/sanchez/gs_playground/real2sim_output/appearance/per_target/grooved_table/projected_overlay.png', 'masked_source': '/home/sanchez/gs_playground/real2sim_output/appearance/per_target/grooved_table/masked_source.png'}
- fr5_fixed_base: {'target': 'fr5_fixed_base', 'status': 'ok', 'samples_total': 6144, 'samples_visible': 5608, 'visible_ratio': 0.9127604166666666, 'uv_coverage_ratio': 0.01136636734008789, 'mask_used': False, 'mask_dir_used': None, 'texture_atlas': '/home/sanchez/gs_playground/real2sim_output/appearance/per_target/fr5_fixed_base/texture_atlas.png', 'partial_texture_atlas': '/home/sanchez/gs_playground/real2sim_output/appearance/per_target/fr5_fixed_base/partial_texture_atlas.png', 'visibility_mask': '/home/sanchez/gs_playground/real2sim_output/appearance/per_target/fr5_fixed_base/uv_visibility_mask.png', 'projected_overlay': '/home/sanchez/gs_playground/real2sim_output/appearance/per_target/fr5_fixed_base/projected_overlay.png', 'masked_source': '/home/sanchez/gs_playground/real2sim_output/appearance/per_target/fr5_fixed_base/masked_source.png'}

Warnings:
