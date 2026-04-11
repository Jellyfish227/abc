#!/bin/bash

# ---------- defaults ----------
VERSION=""
TEST_FOLDER=""
RUN_CEC=0
PART_MODE="default"   # default | exact | multi | sweep
PART_N=0
PART_M=0
NPROC=$(nproc)

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
        -p*)
            PART_VAL="${1#-p}"
            if [[ "$PART_VAL" == "sweep" ]]; then
                PART_MODE="sweep"
            elif [[ "$PART_VAL" =~ ^([0-9]+)x$ ]]; then
                PART_MODE="multi"
                PART_M="${BASH_REMATCH[1]}"
            elif [[ "$PART_VAL" =~ ^[0-9]+$ ]]; then
                PART_MODE="exact"
                PART_N="$PART_VAL"
            else
                echo "Error: invalid -p value '$PART_VAL'"
                echo "       Use: -p<n>  | -p<m>x  | -psweep"
                exit 1
            fi
            shift
            ;;
        -*)
            echo "Unknown flag: $1"
            echo "Usage: $0 -if <4|6> [-cec] [-p<n|mx|sweep>] [test_folder_path]"
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
    echo "Usage: $0 -if <4|6> [-cec] [-p<n|mx|sweep>] [test_folder_path]"
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

# ---------- build sweep list of "count:label" pairs ----------
declare -a SWEEP_LIST=()

case "$PART_MODE" in
    default)
        SWEEP_LIST=("0:default")
        ;;
    exact)
        SWEEP_LIST=("${PART_N}:p${PART_N}")
        ;;
    multi)
        SWEEP_LIST=("$((PART_M * NPROC)):p${PART_M}x")
        ;;
    sweep)
        for ((i=1; i<=NPROC; i++)); do
            SWEEP_LIST+=("${i}:p${i}")
        done
        for ((m=1; m<=16; m++)); do
            SWEEP_LIST+=("$((m * NPROC)):p${m}x")
        done
        ;;
esac

# ---------- summary ----------
echo "Version  : $VERSION"
echo "Folder   : $TEST_FOLDER"
echo "Output   : $POST_DIR/$POST_PREFIX-*"
echo "CEC check: $([ $RUN_CEC -eq 1 ] && echo enabled || echo disabled)"
echo "nproc    : $NPROC"
case "$PART_MODE" in
    default) echo "Partition: default (no -p)" ;;
    exact)   echo "Partition: exact n=$PART_N" ;;
    multi)   echo "Partition: ${PART_M}x nproc = $((PART_M * NPROC))" ;;
    sweep)   echo "Partition: sweep — n=1..${NPROC} then m=1x..16x (${#SWEEP_LIST[@]} passes)" ;;
esac
echo

# ---------- per-pass processing function ----------
run_pass() {
    local part_count="$1"
    local part_label="$2"

    local fx_arg=""
    [[ $part_count -gt 0 ]] && fx_arg=" -p $part_count"

    local total
    total=$(find "$TEST_FOLDER" -maxdepth 1 -name "*.aig" -type f | wc -l)
    local counter=0

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
        echo "Processing [$counter/$total][$part_label]: $file (actual size: $readable_size)"
        echo "-----------------------------------------------------------------------------------------"

        ABC_CMD="&r -s $file; &ps; &fx${fx_arg}; &ps"
        [[ $RUN_CEC -eq 1 ]] && \
            ABC_CMD="$ABC_CMD; &put; &w $POST_DIR/${POST_PREFIX}-${part_label}-$filename"

        if [[ $RUN_CEC -eq 0 ]]; then
            time ./abc -c "$ABC_CMD"
        else
            ./abc -c "$ABC_CMD"
        fi
        echo "========================================================================================="

        if [[ $RUN_CEC -eq 1 ]]; then
            echo "========================================================================================="
            echo "Verifying [$counter/$total][$part_label]: $file (actual size: $readable_size)"
            echo "-----------------------------------------------------------------------------------------"
            ./abc -c "cec $POST_DIR/${POST_PREFIX}-${part_label}-$filename $file"
            echo "========================================================================================="
        fi

        echo
        echo
    done
}

# ---------- main ----------
for entry in "${SWEEP_LIST[@]}"; do
    part_count="${entry%%:*}"
    part_label="${entry##*:}"

    if [[ "$PART_MODE" == "sweep" ]]; then
        echo "####################################################"
        echo "# Sweep pass : $part_label  (partitions = $part_count)"
        echo "####################################################"
        echo
    fi

    run_pass "$part_count" "$part_label"
done