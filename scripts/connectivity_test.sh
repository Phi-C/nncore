#! /bin/bash
set -euo pipefail

TARGET_IP=${TARGET_IP:-"127.0.0.1"}
PORT=${PORT:-"23456"}

is_installed() {
    command -v "$1" &> /dev/null
}

# Install package if not already installed
install_package() {
  local package_name=$1
  local install_cmd=$2

  if is_installed "$package_name"; then
    echo "$package_name is already installed. Skipping..."
  else
    echo "Installing $package_name..."
    eval "$install_cmd"
  fi
}

connectivity_test() {
    ping -c 1 "$TARGET_IP" &> /dev/null
    if [ $? -eq 0 ]; then
        echo "Ping successful"
    else
        echo "Ping failed"
        exit 1
    fi

    nc -z "$TARGET_IP" "$PORT" &> /dev/null
    if [ $? -eq 0 ]; then
        echo "Connection successful"
    else
        echo "Connection failed"
        exit 1
    fi
}

# OS Detection
if [[ -f "/etc/os-release" ]] && grep -qi "ubuntu" /etc/os-release; then
    # apt update
    GENERAL_INSTALLER="apt install -y"
elif [[ -f "/etc/os-release" ]] && (grep -qi "centos" /etc/os-release || grep -qi "tencentos" /et/os-release); then
    # yum update
    GENERAL_INSTALLER="yum install -y"
else
    echo "Unsupported OS"
    exit 1
fi


install_package "nc" "$GENERAL_INSTALLER nc"
install_package "ping" "$GENERAL_INSTALLER iputils-ping"

connectivity_test