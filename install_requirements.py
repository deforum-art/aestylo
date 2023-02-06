import platform
import subprocess


def pip_install_packages(packages,extra_index_url=None):
    for package in packages:
        try:
            print(f"..installing {package}")
            if extra_index_url is not None:
                running = subprocess.call(["pip", "install", "-q", package,  "--extra-index-url", extra_index_url], shell=False)
            else:
                running = subprocess.call(["pip", "install", "-q", package],shell=False)
        except Exception as e:
            print(f"failed to install {package}: {e}")
    return


def install_requirements():
    # Detect System
    os_system = platform.system()
    print(f"system detected: {os_system}")


    # Install pytorch
    torch = [
        "torch",
        "torchvision",
    ]


    extra_index_url = "https://download.pytorch.org/whl/cu117" if os_system == 'Windows' else None
    pip_install_packages(torch,extra_index_url=extra_index_url)


    # List of common packages to install
    common = [
        "flask",
        "pandas",
        "git+https://github.com/openai/CLIP.git",
        "pytorch_lightning",
    ]

    pip_install_packages(common)



if __name__ == "__main__":
    install_requirements()