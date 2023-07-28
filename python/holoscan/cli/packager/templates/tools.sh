#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#===============================================================================
# Default values for environment variables.
#===============================================================================

init_globals() {
    if [ "$0" != "/bin/bash" ] && [ "$0" != "bash" ]; then
        SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
        export RUN_SCRIPT_FILE="$(readlink -f "$0")"
    else
        export RUN_SCRIPT_FILE="$(readlink -f "${BASH_SOURCE[0]}")"
    fi

    export TOP=$(dirname "${RUN_SCRIPT_FILE}")

    export HOLOSCAN_APPLICATION="${HOLOSCAN_APPLICATION:=/opt/holoscan/app}"
    export HOLOSCAN_INPUT_PATH="${HOLOSCAN_INPUT_PATH:=/var/holoscan/input}"
    export HOLOSCAN_OUTPUT_PATH="${HOLOSCAN_OUTPUT_PATH:=/var/holoscan/output}"
    export HOLOSCAN_WORKDIR="${HOLOSCAN_WORKDIR:=/var/holoscan/}"
    export HOLOSCAN_MODEL_PATH="${HOLOSCAN_MODEL_PATH:=/opt/holoscan/models/}"
    export HOLOSCAN_DOCS="${HOLOSCAN_DOCS:=/opt/holoscan/docs/}"
    export HOLOSCAN_CONFIG_PATH="${HOLOSCAN_CONFIG_PATH:=/var/holoscan/app.config}"
    export HOLOSCAN_APP_MANIFEST_PATH="${HOLOSCAN_APP_MANIFEST_PATH:=/etc/holoscan/app.json}"
    export HOLOSCAN_PKG_MANIFEST_PATH="${HOLOSCAN_PKG_MANIFEST_PATH:=/etc/holoscan/pkg.json}"
}

################################################################################
# Utility functions
################################################################################


c_str() {
    local old_color=39
    local old_attr=0
    local color=39
    local attr=0
    local text=""
    local mode="color"
    if [ "${1:-}" = "color" ]; then
        mode="color"
        shift
    elif [ "${1:-}" = "nocolor" ]; then
        mode="nocolor"
        shift
    fi

    for i in "$@"; do
        case "$i" in
            r|R)
                color=31
                ;;
            g|G)
                color=32
                ;;
            y|Y)
                color=33
                ;;
            b|B)
                color=34
                ;;
            p|P)
                color=35
                ;;
            c|C)
                color=36
                ;;
            w|W)
                color=37
                ;;

            z|Z)
                color=0
                ;;
        esac
        case "$i" in
            l|L|R|G|Y|B|P|C|W)
                attr=1
                ;;
            n|N|r|g|y|b|p|c|w)
                attr=0
                ;;
            z|Z)
                attr=0
                ;;
            *)
                text="${text}$i"
        esac
        if [ "${mode}" = "color" ]; then
            if [ ${old_color} -ne ${color} ] || [ ${old_attr} -ne ${attr} ]; then
                text="${text}\033[${attr};${color}m"
                old_color=$color
                old_attr=$attr
            fi
        fi
    done
    /bin/echo -en "$text"
}

c_echo() {
    # Select color/nocolor based on the first argument
    local mode="color"
    if [ "${1:-}" = "color" ]; then
        mode="color"
        shift
    elif [ "${1:-}" = "nocolor" ]; then
        mode="nocolor"
        shift
    else
        if [ ! -t 1 ]; then
            mode="nocolor"
        fi
    fi

    local old_opt="$(shopt -op xtrace)" # save old xtrace option
    set +x # unset xtrace

    if [ "${mode}" = "color" ]; then
        local text="$(c_str color "$@")"
        /bin/echo -e "$text\033[0m"
    else
        local text="$(c_str nocolor "$@")"
        /bin/echo -e "$text"
    fi
    eval "${old_opt}" # restore old xtrace option
}

echo_err() {
    >&2 echo "$@"
}

c_echo_err() {
    >&2 c_echo "$@"
}

newline() {
    echo
}

info() {
    c_echo_err W "$(date -u '+%Y-%m-%d %H:%M:%S') [INFO] " Z "$@"
}

error() {
    c_echo_err R "$(date -u '+%Y-%m-%d %H:%M:%S') [ERROR] " Z "$@"
}

fatal() {
    if [ -n "$*" ]; then
        c_echo_err R "$(date -u '+%Y-%m-%d %H:%M:%S') [FATAL] " Z "$@"
        echo_err
    fi
    if [ -n "${SCRIPT_DIR}" ]; then
        exit 1
    else
        kill -INT $$  # kill the current process instead of exit in shell environment.
    fi
}

run_command() {
    local status=0
    local cmd="$*"

    [ "$(echo -n "$@")" = "" ] && return 1 # return 1 if there is no command available

    if [ "${DO_DRY_RUN}" != "true" ]; then
        "$@"
        status=$?
    fi

    return $status
}


#===============================================================================
# Section: Show
#===============================================================================

print() {
    # Parse CLI arguments next
    ARGS=("$@")
    if [ "$ARGS" = "" ]; then
        ARGS=*
    fi

    local i
    local arg
    for i in "${!ARGS[@]}"; do
        arg="${ARGS[i]}"
        if [[ "$arg" = "*" || "$arg" = "" ]]; then
           opt=all
        fi
        if [ "$arg" = "app" ]; then
           opt=app
        fi
        if [ "$arg" = "pkg" ]; then
           opt=pkg
        fi
    done
    newline
    if [[ "$opt" = "all" || "$opt" = "app" ]]; then
        if [ -f "${HOLOSCAN_APP_MANIFEST_PATH}" ]; then
            c_echo "============================== app.json =============================="
            run_command cat ${HOLOSCAN_APP_MANIFEST_PATH} | jq
        else
            c_echo_err "${HOLOSCAN_APP_MANIFEST_PATH} does not exists!"
        fi
    fi
    newline
    if [[ "$opt" = "all" || "$opt" = "pkg" ]]; then
        if [ -f "${HOLOSCAN_PKG_MANIFEST_PATH}" ]; then
            c_echo "============================== pkg.json =============================="
            run_command cat ${HOLOSCAN_PKG_MANIFEST_PATH} | jq
        else
            c_echo_err "${HOLOSCAN_PKG_MANIFEST_PATH} does not exists!"
        fi
    fi
    newline
}

print_env() {
    run_command env -0 | sort -z | tr '\0' '\n'
}
#===============================================================================
# Section: Export
#===============================================================================

copy_app() {
    info "Copying application from ${HOLOSCAN_APPLICATION} to ${EXPORT_ROOT}/app"
    if [ -d "${HOLOSCAN_APPLICATION}" ]; then
        run_command cp -r ${HOLOSCAN_APPLICATION} ${EXPORT_ROOT}/app
    else
        info "'${HOLOSCAN_APPLICATION}' cannot be found."
    fi
    newline
}

copy_configs() {
    if [ ! -d "${EXPORT_ROOT}/config/" ] 
    then
        mkdir -p ${EXPORT_ROOT}/config/
    fi

    info "Copying application manifest file from ${HOLOSCAN_APP_MANIFEST_PATH} to ${EXPORT_ROOT}/config/app.json"
    if [ -f "${HOLOSCAN_APP_MANIFEST_PATH}" ]; then
        run_command cp ${HOLOSCAN_APP_MANIFEST_PATH} ${EXPORT_ROOT}/config/app.json
    else
        info "application manifest file '${HOLOSCAN_APP_MANIFEST_PATH}' cannot be found."
    fi

    info "Copying pkg manifest file from ${HOLOSCAN_PKG_MANIFEST_PATH} to ${EXPORT_ROOT}/config/pkg.json"
    if [ -f "${HOLOSCAN_PKG_MANIFEST_PATH}" ]; then
        run_command cp ${HOLOSCAN_PKG_MANIFEST_PATH} ${EXPORT_ROOT}/config/pkg.json
    else
        info "package manifest file '${HOLOSCAN_PKG_MANIFEST_PATH}' cannot be found."
    fi

    local CONFIG_FILE_NAME=$(basename ${HOLOSCAN_CONFIG_PATH})
    info "Copying application configuration from ${HOLOSCAN_CONFIG_PATH} to ${EXPORT_ROOT}/config/$CONFIG_FILE_NAME"
    if [ -f "${HOLOSCAN_CONFIG_PATH}" ]; then
        run_command cp ${HOLOSCAN_CONFIG_PATH} ${EXPORT_ROOT}/config/$CONFIG_FILE_NAME
    else
        info "Application configuration'${HOLOSCAN_CONFIG_PATH}' cannot be found."
    fi
    newline
}

copy_models() {
    info "Copying models from ${HOLOSCAN_MODEL_PATH} to ${EXPORT_ROOT}/models"
    if [ -d "${HOLOSCAN_MODEL_PATH}" ]; then
        run_command cp -r ${HOLOSCAN_MODEL_PATH} ${EXPORT_ROOT}/models
    else
        info "'${HOLOSCAN_MODEL_PATH}' cannot be found."
    fi
    newline
}

copy_docs() {
    info "Copying documentation from ${HOLOSCAN_DOCS} to ${EXPORT_ROOT}/docs"
    if [ -d "${HOLOSCAN_DOCS}" ]; then
        run_command cp -r ${HOLOSCAN_DOCS} ${EXPORT_ROOT}/docs
    else
        info "'${HOLOSCAN_DOCS}' cannot be found."
    fi
    newline
}

check_permission(){
    test -w $@
    if [ $? -ne 0 ]; then
        fatal "Permission error: ensure the directory is created on the host first before using the 'extract' command."
    fi
}


copy() {
    EXPORT_ROOT=/var/run/holoscan/export

    if [ -d "${EXPORT_ROOT}" ]
    then
        if [ -d "${EXPORT_ROOT}/app" ]; then
            check_permission "${EXPORT_ROOT}/app"
            copy_app
        elif [ -d "${EXPORT_ROOT}/config" ]; then
            check_permission "${EXPORT_ROOT}/config"
            copy_configs
        elif [ -d "${EXPORT_ROOT}/models" ]; then
            check_permission "${EXPORT_ROOT}/models"
            copy_models
        elif [ -d "${EXPORT_ROOT}/docs" ]; then
            copy_docs "${EXPORT_ROOT}/docs"
        else
            check_permission "${EXPORT_ROOT}"
            copy_app
            copy_configs
            copy_models
            copy_docs
        fi
    else
        fatal "No volume mount found ${EXPORT_ROOT}."
    fi
}

#===============================================================================


print_usage() {
    local YELLOW='\033[0;93m' # YELLOW
    local RED='\033[0;91m' # YELLOW
    local NC='\033[0m' # No Color
    c_echo "USAGE: /var/holoscan/tools [command] [arguments]..."
    c_echo " Command List"
    c_echo "    ${YELLOW}extract${NC}  ---------------------------  Extract data based on mounted volume paths."
    c_echo "        /var/run/holoscan/export/app        extract the application"
    c_echo "        /var/run/holoscan/export/config     extract app.json and pkg.json manifest files and application YAML."
    c_echo "        /var/run/holoscan/export/models     extract models"
    c_echo "        /var/run/holoscan/export/docs       extract documentation files"
    c_echo "        /var/run/holoscan/export            extract all of the above"
    c_echo "        ${RED}IMPORTANT${NC}: ensure the directory to be mounted for data extraction is created first on the host system"
    c_echo "                   and has the correct permissions. If the directory had been created by the container previously"
    c_echo "                   with the user and group being root, please delete it and manually create it again."
    c_echo "    ${YELLOW}show${NC}  -----------------------------  Print manifest file(s): [app|pkg] to the terminal."
    c_echo "        ${YELLOW}app${NC}                                 print app.json"
    c_echo "        ${YELLOW}pkg${NC}                                 print pkg.json"
    c_echo "    ${YELLOW}env${NC}  -------------------------  Print all environment variables to the terminal."
}


main() {
    if [ "$1" = "show" ]; then
        shift
        print "$@"
    elif  [ "$1" = "env" ]; then
        shift
        print_env
    elif  [ "$1" = "extract" ]; then
        shift
        copy
    elif  [ "$1" = "help" ]; then
        shift
        print_usage
    else # all other commands will launch the application
        local command=$(jq -r '.command | fromjson | join(" ")' ${HOLOSCAN_APP_MANIFEST_PATH} 2>/dev/null)
        if [ -n "${command}" ]; then
            info "Launching application ${command} ${@}..."
            eval ${command} "$@"
            exit_code=$?
            if [ $exit_code -ne 0 ] && [ -f /.dockerenv ] && [ "$HOLOSCAN_HOSTING_SERVICE" != "HOLOSCAN_RUN" ]; then
                newline
                c_echo "================================================================================================"
                c_echo "Application exited with ${exit_code}."
                newline
                newline
                c_echo nocolor "When running inside docker, ensure that the runtime is set to nvidia with required arguments."
                newline
                c_echo nocolor "For example:"
                c_echo nocolor "docker run --runtime nvidia \\"
                c_echo nocolor "           --gpus all \\"
                c_echo nocolor "           -it \\"
                c_echo nocolor "           -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \\"
                c_echo nocolor "           -e DISPLAY=\$DISPLAY \\"
                c_echo nocolor "           -v /tmp/.X11-unix:/tmp/.X11-unix \\"
                c_echo nocolor "           -v /usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json \\"
                c_echo nocolor "           -v /etc/vulkan/icd.d/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json \\"
                c_echo nocolor "           -v \${MY-INPUT-DATA}:/var/holoscan/input \\"
                c_echo nocolor "           my-container-image[:tag]"
                newline
                c_echo "================================================================================================"
                newline
            fi
        else
            fatal "Failed to launch application; failed to read/parse command from ${HOLOSCAN_APP_MANIFEST_PATH}."
        fi
    fi
}

init_globals

if [ -n "${SCRIPT_DIR}" ]; then
    main "$@"
fi
