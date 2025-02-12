# Air-Gapped Installation

You can install GPUStack in an air-gapped environment. An air-gapped environment refers to a setup where GPUStack will be installed offline, behind a firewall, or behind a proxy.

The following methods are available for installing GPUStack in an air-gapped environment:

- [Docker Installation](#docker-installation)
- [Manual Installation](#manual-installation)

## Docker Installation

When running GPUStack with Docker, it works out of the box in an air-gapped environment as long as the Docker images are available. To do this, follow these steps:

1. Pull GPUStack Docker images in an online environment.
2. Publish Docker images to a private registry.
3. Refer to the [Docker Installation](docker-installation.md) guide to run GPUStack using Docker.

## Manual Installation

For manual installation, you need to prepare the required packages and tools in an online environment and then transfer them to the air-gapped environment.

### Prerequisites

Set up an online environment identical to the air-gapped environment, including **OS**, **architecture**, and **Python version**.

### Step 1: Download the Required Packages

Run the following commands in an online environment:

```bash
# On Windows (PowerShell):
# $PACKAGE_SPEC = "gpustack"

# Optional: To include extra dependencies (vllm, audio, all) or install a specific version
# PACKAGE_SPEC="gpustack[all]"
# PACKAGE_SPEC="gpustack==0.4.0"
PACKAGE_SPEC="gpustack"

# Download all required packages
pip wheel $PACKAGE_SPEC -w gpustack_offline_packages

# Install GPUStack to access its CLI
pip install gpustack

# Download dependency tools and save them as an archive
gpustack download-tools --save-archive gpustack_offline_tools.tar.gz
```

Optional: Additional Dependencies for macOS.

```bash
# Deploying the speech-to-text CosyVoice model on macOS requires additional dependencies.
brew_install_with_version() {
  BREW_APP_NAME="$1"
  BREW_APP_VERSION="$2"
  BREW_APP_NAME_WITH_VERSION="$BREW_APP_NAME@$BREW_APP_VERSION"
  TAP_NAME="$USER/local-$BREW_APP_NAME-$BREW_APP_VERSION"

  # Check current installed versions
  echo "Checking installed versions of $BREW_APP_NAME."
  INSTALLED_VERSIONS=$(brew list --versions | grep "$BREW_APP_NAME" || true)
  INSTALLED_VERSION_COUNT=$(brew list --versions | grep -c "$BREW_APP_NAME" || true)

  if [ -n "$INSTALLED_VERSIONS" ]; then
    # Check if the target version is already installed
    if echo "$INSTALLED_VERSIONS" | grep -q "$BREW_APP_VERSION"; then
      if [ "$INSTALLED_VERSION_COUNT" -eq 1 ]; then
        echo "$BREW_APP_NAME $BREW_APP_VERSION is already installed."
        return 0
      elif [ "$INSTALLED_VERSION_COUNT" -gt 1 ]; then
        SINGLE_LINE_INSTALLED_VERSIONS=$(echo "$INSTALLED_VERSIONS" | tr '\n' ' ')
        echo "Installed $BREW_APP_NAME versions: $SINGLE_LINE_INSTALLED_VERSIONS"
        echo "Multiple versions of $BREW_APP_NAME are installed, relink the target version."
        echo "$INSTALLED_VERSIONS" | awk '{print $1}' | while read -r installed_version; do
            brew unlink "$installed_version"
        done

        NEED_VERSION=$(echo "$INSTALLED_VERSIONS" | grep "$BREW_APP_VERSION" | cut -d ' ' -f 1)
        brew link --overwrite "$NEED_VERSION"
        return 0
      fi
    fi
  fi

  # Create a new Homebrew tap
  if brew tap-info "$TAP_NAME" 2>/dev/null | grep -q "Installed"; then
      echo "Tap $TAP_NAME already exists. Skipping tap creation."
  else
      echo "Creating a new tap: $TAP_NAME..."
      if ! brew tap-new "$TAP_NAME"; then
          echo "Failed to create the tap $TAP_NAME." && exit 1
      fi
  fi

  # Extract the history version of the app
  echo "Extracting $BREW_APP_NAME version $BREW_APP_VERSION."
  brew tap homebrew/core --force
  brew extract --force --version="$BREW_APP_VERSION" "$BREW_APP_NAME" "$TAP_NAME"

  # Install the specific version of the application
  echo "Unlinking before install $BREW_APP_NAME."
  echo "$INSTALLED_VERSIONS" | awk '{print $1}' | while read -r installed_version; do
    brew unlink "$installed_version" 2>/dev/null || true
  done

  echo "Installing $BREW_APP_NAME version $BREW_APP_VERSION."
  if ! brew install "$TAP_NAME/$BREW_APP_NAME_WITH_VERSION"; then
      echo "Failed to install $BREW_APP_NAME version $BREW_APP_VERSION." && exit 1
  fi

  echo "Installed and linked $BREW_APP_NAME version $BREW_APP_VERSION."
}
brew_install_with_version openfst 1.8.3
CPLUS_INCLUDE_PATH=$(brew --prefix openfst@1.8.3)/include
LIBRARY_PATH=$(brew --prefix openfst@1.8.3)/lib

AUDIO_DEPENDENCY_PACKAGE_SPEC="wetextprocessing==1.0.4.1"
pip wheel $AUDIO_DEPENDENCY_PACKAGE_SPEC -w gpustack_audio_dependency_offline_packages
mv gpustack_audio_dependency_offline_packages/* gpustack_offline_packages/ && rm -rf gpustack_audio_dependency_offline_packages
```

!!!note

    This instruction assumes that the online environment uses the same GPU type as the air-gapped environment. If the GPU types differ, use the `--device` flag to specify the device type for the air-gapped environment. Refer to the [download-tools](../cli-reference/download-tools.md) command for more information.

### Step 2: Transfer the Packages

Transfer the following files from the online environment to the air-gapped environment.

- `gpustack_offline_packages` directory.
- `gpustack_offline_tools.tar.gz` file.

### Step 3: Install GPUStack

In the air-gapped environment, run the following commands:

```bash
# Install GPUStack from the downloaded packages
pip install --no-index --find-links=gpustack_offline_packages gpustack

# Load and apply the pre-downloaded tools archive
gpustack download-tools --load-archive gpustack_offline_tools.tar.gz
```

Optional: Additional Dependencies for macOS.

```bash
# Install the additional dependencies for speech-to-text CosyVoice model on macOS.
brew install openfst

pip install --no-index --find-links=gpustack_offline_packages wetextprocessing
```

Now you can run GPUStack by following the instructions in the [Manual Installation](manual-installation.md#run-gpustack) guide.
