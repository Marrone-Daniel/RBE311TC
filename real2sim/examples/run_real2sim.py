from __future__ import annotations

import argparse

from real2sim.pipeline import Real2SimPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hybrid Real2Sim pipeline.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    result = Real2SimPipeline.from_config_file(args.config).run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    print("Real2Sim finished.")
    if result.get("target_object_enabled", True):
        print(f"  mask_pixels={result['mask_pixels']}")
        print(f"  center_world={result['center_world']}")
        print(f"  size_world={result['size_world']}")
    if "scene_binding" in result:
        print(f"  scene_assets={result['scene_binding'].get('assets', 0)}")
        if result["scene_binding"].get("config"):
            print(f"  scene_binding={result['scene_binding']['config']}")
    if result["warnings"]:
        print("  warnings:")
        for warning in result["warnings"]:
            print(f"    - {warning}")


if __name__ == "__main__":
    main()
