# import sys
# import subprocess
# import importlib.metadata
# from packaging import version

# # Define colors for terminal output
# GREEN = "\033[92m"
# RED = "\033[91m"
# YELLOW = "\033[93m"
# RESET = "\033[0m"

# def check_system_dependency(command, name):
#     """Checks if a system tool (like ffmpeg) is available in the path."""
#     try:
#         subprocess.run([command, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
#         print(f"[{GREEN}OK{RESET}] System Tool: {name} found.")
#         return True
#     except (subprocess.CalledProcessError, FileNotFoundError):
#         print(f"[{RED}MISSING{RESET}] System Tool: {name} NOT found.")
#         print(f"      {YELLOW}Hint: Install it via 'sudo apt install {name}' or download from website.{RESET}")
#         return False

# def check_python_dependencies(req_file='requirements.txt'):
#     """Checks Python packages listed in requirements.txt."""
#     print(f"\nScanning {req_file}...")
    
#     try:
#         with open(req_file, 'r') as f:
#             requirements = f.readlines()
#     except FileNotFoundError:
#         print(f"[{RED}ERROR{RESET}] {req_file} not found!")
#         return

#     all_passed = True

#     for req in requirements:
#         req = req.strip()
#         if not req or req.startswith('#'):
#             continue

#         # Split package name and version (e.g., "Flask==3.0.0" -> "Flask", "3.0.0")
#         if '==' in req:
#             pkg_name, required_ver = req.split('==')
#         else:
#             pkg_name, required_ver = req, None

#         try:
#             # Get installed version
#             installed_ver = importlib.metadata.version(pkg_name)
            
#             if required_ver:
#                 if version.parse(installed_ver) == version.parse(required_ver):
#                     print(f"[{GREEN}OK{RESET}] {pkg_name:<15} (Installed: {installed_ver})")
#                 else:
#                     print(f"[{YELLOW}MISMATCH{RESET}] {pkg_name:<15} (Wanted: {required_ver}, Got: {installed_ver})")
#                     # We don't fail for mismatch, just warn, as newer versions might work.
#             else:
#                  print(f"[{GREEN}OK{RESET}] {pkg_name:<15} (Installed: {installed_ver})")

#         except importlib.metadata.PackageNotFoundError:
#             print(f"[{RED}MISSING{RESET}] {pkg_name:<15} is not installed.")
#             all_passed = False

#     return all_passed

# def main():
#     print("--- Respi-View Dependency Check ---\n")
    
#     # 1. Check Python Packages
#     packages_ok = check_python_dependencies()
    
#     print("\n--- System Audio Libraries ---")
#     # 2. Check FFmpeg (Crucial for Librosa loading mp3/wav on some systems)
#     ffmpeg_ok = check_system_dependency("ffmpeg", "FFmpeg")
    
#     print("\n" + "="*30)
#     if packages_ok and ffmpeg_ok:
#         print(f"{GREEN}✅ SUCCESS: Environment is ready to run!{RESET}")
#     else:
#         print(f"{RED}❌ FAILURE: Please fix the missing dependencies above.{RESET}")
#         print(f"Run: pip install -r requirements.txt")

# if __name__ == "__main__":
#     main()

