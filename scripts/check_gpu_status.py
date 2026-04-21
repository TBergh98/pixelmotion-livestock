#!/usr/bin/env python
"""
Check GPU and CUDA availability in the current environment.
Usage: python scripts/check_gpu_status.py
"""

import sys
import cv2
import platform


def main():
    print("=" * 60)
    print("GPU and CUDA Status Check")
    print("=" * 60)
    print()

    # Environment info
    print("Environment:")
    print(f"  Python:       {sys.version.split()[0]}")
    print(f"  Platform:     {platform.platform()}")
    print(f"  OpenCV:       {cv2.__version__}")
    print()

    # Check CUDA in build
    print("OpenCV Build Information:")
    info = cv2.getBuildInformation()
    cuda_in_build = "CUDA" in info
    print(f"  CUDA in build: {'YES' if cuda_in_build else 'NO'}")
    if not cuda_in_build:
        print("  → To enable CUDA, rebuild OpenCV using:")
        print("      conda env create -f environment-gpu.yml")
        print("      conda activate pixelmotion-livestock-gpu")
        print("      .\\scripts\\build_opencv_cuda.ps1")
    print()

    # Check CUDA module
    print("CUDA Module:")
    cuda_module = getattr(cv2, "cuda", None)
    if cuda_module is None:
        print("  cv2.cuda: NOT AVAILABLE")
    else:
        print("  cv2.cuda: AVAILABLE")

        try:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"  Devices:  {device_count}")

            if device_count > 0:
                for i in range(device_count):
                    try:
                        props = cv2.cuda.DeviceInfo(i)
                        name = props.name() if hasattr(props, "name") else f"Device {i}"
                        compute_capability = (
                            f"{props.majorVersion()}.{props.minorVersion()}"
                            if hasattr(props, "majorVersion")
                            else "unknown"
                        )
                        print(f"    [{i}] {name} (Compute: {compute_capability})")
                    except Exception as e:
                        print(f"    [{i}] Error: {e}")
            else:
                print()
                print("  ⚠️  No CUDA devices found.")
                print()
                print("  Possible causes:")
                print("    - OpenCV was not built with CUDA support")
                print("    - NVIDIA CUDA Toolkit is not installed")
                print("    - GPU is not recognized by the system")
                print("    - Driver version is incompatible")
        except Exception as e:
            print(f"  Error querying devices: {e}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
