import os
import platform

TRINETRA_ART = r'''
         _____
     _.-'     `'-._
   ,'               '.
  /                   \
 |    ___________     |
 |   |           |    |
 |   |    ( )    |    |
 |   |___________| 👁️  |
  \                   /
   '.               ,'
     '-.._______.-'

 Trinetra watches everything
'''

def generate_unix_script():
    script = f'''#!/bin/bash

# Create and activate the Conda environment
conda create --name trinetra python=3.8 -y
source activate trinetra

# Install requirements
pip install -r requirements.txt

# Install CLIP
pip install git+https://github.com/openai/CLIP.git

echo "Trinetra environment setup complete!"

# Display Trinetra ASCII art
cat << EOT
{TRINETRA_ART}
EOT
'''
    with open('setup_trinetra.sh', 'w') as f:
        f.write(script)
    os.chmod('setup_trinetra.sh', 0o755)
    print("Generated setup_trinetra.sh")
    print("Run './setup_trinetra.sh' to set up the environment")

def generate_windows_script():
    script = f'''@echo off

REM Create and activate the Conda environment
call conda create --name trinetra python=3.8 -y
call conda activate trinetra

REM Install requirements
pip install -r requirements.txt

REM Install CLIP
pip install git+https://github.com/openai/CLIP.git

echo Trinetra environment setup complete!

REM Display Trinetra ASCII art
echo.
echo {TRINETRA_ART.replace('"', '^"')}
echo.

pause
'''
    with open('setup_trinetra.bat', 'w') as f:
        f.write(script)
    print("Generated setup_trinetra.bat")
    print("Run 'setup_trinetra.bat' to set up the environment")

if __name__ == "__main__":
    system = platform.system()
    if system == "Windows":
        generate_windows_script()
    elif system in ["Linux", "Darwin"]:
        generate_unix_script()
    else:
        print(f"Unsupported operating system: {system}")