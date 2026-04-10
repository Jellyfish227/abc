#!/bin/bash

# ---------- defaults ----------
VERSION=""
TEST_FOLDER=""
RUN_CEC=0       # default: skip CEC verification

# ---------- argument parsing ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -if)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: -if requires an argument (4 or 6)"
                exit 1
            fi
            VERSION="$2"
            shift 2
            ;;
        -cec)
            RUN_CEC=1
            shift
            ;;
        -*)
            echo "Unknown flag: $1"
            echo "Usage: $0 -if <4|6> [-cec] [test_folder_path]"
            exit 1
            ;;
        *)
            if [[ -n "$TEST_FOLDER" ]]; then
                echo "Error: unexpected extra argument '$1'"
                exit 1
            fi
            TEST_FOLDER="$1"
            shift
            ;;
    esac
done

# ---------- validate -if ----------
if [[ -z "$VERSION" ]]; then
    echo "Error: -if <n> is required"
    echo "Usage: $0 -if <4|6> [-cec] [test_folder_path]"
    exit 1
fi

if [[ "$VERSION" != "4" && "$VERSION" != "6" ]]; then
    echo "Error: -if value must be 4 or 6 (got '$VERSION')"
    exit 1
fi

# ---------- set version-dependent variables ----------
if [[ "$VERSION" == "4" ]]; then
    TEST_FOLDER="${TEST_FOLDER:-../mappedaig_v4}"
    POST_DIR="post_fx4"
    POST_PREFIX="post4"
else
    TEST_FOLDER="${TEST_FOLDER:-../mappedaig_v6}"
    POST_DIR="post_fx6"
    POST_PREFIX="post6"
fi

echo "Version  : $VERSION"
echo "Folder   : $TEST_FOLDER"
echo "Output   : $POST_DIR/$POST_PREFIX-*"
echo "CEC check: $([ $RUN_CEC -eq 1 ] && echo enabled || echo disabled)"
echo

# ---------- main loop ----------
total=$(find "$TEST_FOLDER" -maxdepth 1 -name "*.aig" -type f | wc -l)
counter=0

find "$TEST_FOLDER" -maxdepth 1 -name "*.aig" -type f -printf '%s\t%p\n' | \
sort -n | \
while IFS=$'\t' read -r size file; do
    ((counter++))
    filename="${file##*/}"

    if   [ "$size" -lt 1024 ];       then readable_size="${size}B"
    elif [ "$size" -lt 1048576 ];    then readable_size="$(awk "BEGIN {printf \"%.3f\", $size/1024}")K"
    elif [ "$size" -lt 1073741824 ]; then readable_size="$(awk "BEGIN {printf \"%.3f\", $size/1048576}")M"
    else                                  readable_size="$(awk "BEGIN {printf \"%.3f\", $size/1073741824}")G"
    fi

    echo "========================================================================================="
    echo "Processing [$counter/$total]: $file (actual size: $readable_size)"
    echo "-----------------------------------------------------------------------------------------"
    ABC_CMD="&r -s $file; &ps; &fx -v; &ps;"
    [[ $RUN_CEC -eq 1 ]] && \
        ABC_CMD="$ABC_CMD; &put; &w $POST_DIR/$POST_PREFIX-$filename"

    CMD_PREFIX=""
    [[ $RUN_CEC -eq 0 ]] && CMD_PREFIX="time"

    $CMD_PREFIX ./abc -c "$ABC_CMD"
    echo "========================================================================================="

    if [[ $RUN_CEC -eq 1 ]]; then
        echo "========================================================================================="
        echo "Verifying [$counter/$total]: $file (actual size: $readable_size)"
        echo "-----------------------------------------------------------------------------------------"
        ./abc -c "cec $POST_DIR/$POST_PREFIX-$filename $file"
        echo "========================================================================================="
    fi

    echo
    echo
done