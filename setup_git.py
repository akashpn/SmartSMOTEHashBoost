"""
Automated script to initialize git and push to GitHub.

This script will:
1. Initialize git repository (if not already done)
2. Add all files
3. Make initial commit
4. Guide you through connecting to GitHub

Usage:
    python setup_git.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=check
        )
        return result.stdout.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout.strip() + "\n" + e.stderr.strip(), e.returncode


def check_git_installed():
    """Check if git is installed"""
    stdout, returncode = run_command("git --version", check=False)
    if returncode != 0:
        print("❌ Git is not installed or not in PATH")
        print("   Please install Git from: https://git-scm.com/downloads")
        return False
    print(f"✅ Git is installed: {stdout}")
    return True


def is_git_repo():
    """Check if current directory is a git repository"""
    stdout, returncode = run_command("git rev-parse --git-dir", check=False)
    return returncode == 0


def get_remote_url():
    """Get the remote GitHub URL if it exists"""
    stdout, returncode = run_command("git remote get-url origin", check=False)
    if returncode == 0:
        return stdout
    return None


def main():
    print("=" * 70)
    print("GIT SETUP & GITHUB PUSH AUTOMATION")
    print("=" * 70)
    print()

    # Check if git is installed
    if not check_git_installed():
        sys.exit(1)

    # Check if already a git repo
    if is_git_repo():
        print("✅ Git repository already initialized")
        remote_url = get_remote_url()
        if remote_url:
            print(f"✅ Remote already configured: {remote_url}")
        else:
            print("⚠️  No remote configured yet")
    else:
        print("\n📦 Initializing git repository...")
        stdout, returncode = run_command("git init")
        if returncode == 0:
            print("✅ Git repository initialized")
        else:
            print(f"❌ Failed to initialize git: {stdout}")
            sys.exit(1)

    # Check what files need to be added
    print("\n📋 Checking files to commit...")
    stdout, returncode = run_command("git status --short", check=False)
    
    if returncode == 0 and stdout:
        print("Files to be committed:")
        print(stdout)
    else:
        stdout, returncode = run_command("git status", check=False)
        if "nothing to commit" in stdout.lower():
            print("✅ All files already committed")
        else:
            print(stdout)

    # Add all files
    print("\n📝 Adding all files to git...")
    stdout, returncode = run_command("git add .")
    if returncode == 0:
        print("✅ All files added")
    else:
        print(f"❌ Failed to add files: {stdout}")
        sys.exit(1)

    # Check if there's anything to commit
    stdout, returncode = run_command("git status --short", check=False)
    if not stdout:
        print("\n✅ Nothing new to commit (everything already committed)")
    else:
        # Make commit
        print("\n💾 Creating commit...")
        commit_message = """Add cost-sensitive Borderline-SMOTE HashBoost implementation

- Implemented CostSensitiveBorderlineSMOTEHashBoost class
- Added CSV experiment runner (run_csv_experiment.py)
- Added dataset downloader (download_datasets.py)
- Updated requirements.txt and README.md
- Added utility functions (src/utils.py)"""
        
        stdout, returncode = run_command(f'git commit -m "{commit_message}"')
        if returncode == 0:
            print("✅ Commit created successfully")
        else:
            print(f"❌ Failed to commit: {stdout}")
            sys.exit(1)

    # Check remote
    remote_url = get_remote_url()
    
    if remote_url:
        print(f"\n✅ Remote repository configured: {remote_url}")
        print("\n🚀 Pushing to GitHub...")
        
        # Try to push
        stdout, returncode = run_command("git branch -M main", check=False)
        
        stdout, returncode = run_command("git push -u origin main", check=False)
        if returncode == 0:
            print("✅ Successfully pushed to GitHub!")
            print(f"\n🌐 Your repository is now at: {remote_url.replace('.git', '')}")
        else:
            if "fatal: no upstream branch" in stdout or "fatal: The current branch" in stdout:
                print("⚠️  Branch 'main' doesn't exist on remote yet")
                print("   Run this command manually:")
                print("   git push -u origin main")
            elif "fatal: could not read Username" in stdout or "Authentication failed" in stdout:
                print("⚠️  Authentication required")
                print("   You may need to:")
                print("   1. Set up a Personal Access Token (GitHub Settings > Developer settings)")
                print("   2. Or use GitHub CLI: gh auth login")
                print("   3. Or push manually: git push -u origin main")
            else:
                print(f"⚠️  Push failed: {stdout}")
                print("   Try pushing manually: git push -u origin main")
    else:
        print("\n" + "=" * 70)
        print("NEXT STEPS: Connect to GitHub")
        print("=" * 70)
        print("\n1️⃣  Create a new repository on GitHub:")
        print("   Go to: https://github.com/new")
        print("   - Name it: SmartSMOTEHashBoost")
        print("   - Don't initialize with README (you already have one)")
        print("   - Click 'Create repository'")
        print()
        print("2️⃣  Copy the repository URL (e.g., https://github.com/YOUR_USERNAME/SmartSMOTEHashBoost.git)")
        print()
        print("3️⃣  Run this command (replace YOUR_USERNAME and REPO_NAME):")
        print("   git remote add origin https://github.com/YOUR_USERNAME/SmartSMOTEHashBoost.git")
        print("   git branch -M main")
        print("   git push -u origin main")
        print()
        print("Or run this script again after adding the remote!")

    print("\n" + "=" * 70)
    print("✅ Git setup complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
