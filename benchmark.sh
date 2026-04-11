#!/bin/bash
# Intentionally no 'set -e': continue the sweep if one design or cec fails.

# ---------- locate repo root (directory of this script) ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
ABC_BIN="${ABC_BIN:-$SCRIPT_DIR/abc}"

# ---------- defaults ----------
VERSION=""
TEST_FOLDER=""
RUN_CEC=0
PART_MODE="default"   # default | exact | multi | sweep
PART_N=0
PART_M=0
NPROC=$(nproc)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NPROC}"

# ---------- argument parsing ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -if)
            if [[ -z "${2:-}" || "${2:-}" == -* ]]; then
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
    TEST_FOLDER="${TEST_FOLDER:-$SCRIPT_DIR/../mappedaig_v4}"
    POST_DIR="post_fx4"
    POST_PREFIX="post4"
else
    TEST_FOLDER="${TEST_FOLDER:-$SCRIPT_DIR/../mappedaig_v6}"
    POST_DIR="post_fx6"
    POST_PREFIX="post6"
fi

# Resolve to absolute path for find / messages
TEST_FOLDER="$(cd "$TEST_FOLDER" 2>/dev/null && pwd || true)"
if [[ -z "$TEST_FOLDER" || ! -d "$TEST_FOLDER" ]]; then
    echo "Error: test folder not found or not a directory"
    echo "       (set an explicit path as the last argument, or fix the default next to the repo)"
    exit 1
fi

if [[ ! -x "$ABC_BIN" ]]; then
    echo "Error: ABC binary not found or not executable: $ABC_BIN"
    echo "       Build with 'make' or set ABC_BIN=/path/to/abc"
    exit 1
fi

mkdir -p "$SCRIPT_DIR/$POST_DIR"

# ---------- build sweep list: "pOpt:label" (pOpt is ABC &fx -p argument, empty = no -p) ----------
declare -a SWEEP_LIST=()

case "$PART_MODE" in
    default)
        SWEEP_LIST+=(":default")
        ;;
    exact)
        SWEEP_LIST+=("${PART_N}:p${PART_N}")
        ;;
    multi)
        SWEEP_LIST+=("${PART_M}x:p${PART_M}x")
        ;;
    sweep)
        SWEEP_LIST+=( "1:p1" "20:p20" "40:p40" "1x:p1x" "8x:p8x" "16x:p16x" )
        ;;
esac

# ---------- summary ----------
echo "Version  : $VERSION"
echo "Folder   : $TEST_FOLDER"
echo "ABC      : $ABC_BIN"
echo "Output   : $SCRIPT_DIR/$POST_DIR/$POST_PREFIX-*"
echo "CEC check: $([ "$RUN_CEC" -eq 1 ] && echo enabled || echo disabled)"
echo "nproc    : $NPROC (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
case "$PART_MODE" in
    default) echo "Partition: default (no -p)" ;;
    exact)   echo "Partition: exact -p $PART_N" ;;
    multi)   echo "Partition: -p ${PART_M}x  (${PART_M}× nproc jobs, capped by nodes)" ;;
    sweep)   echo "Partition: sweep — -p 1, 20, 40, 1x, 8x, 16x (${#SWEEP_LIST[@]} passes)" ;;
esac
echo

# ---------- per-pass processing function ----------
run_pass() {
    local p_opt="$1"    # empty, or e.g. 64, or 8x (passed to &fx -p)
    local part_label="$2"

    local fx_arg=""
    [[ -n "$p_opt" ]] && fx_arg=" -p${p_opt}"

    mapfile -t files < <(find "$TEST_FOLDER" -maxdepth 1 -name '*.aig' -type f -printf '%s\t%p\n' 2>/dev/null | sort -n -k1,1)
    local total=${#files[@]}
    local counter=0

    if [[ "$total" -eq 0 ]]; then
        echo "Warning: no .aig files in $TEST_FOLDER — skipping pass [$part_label]"
        echo
        return 0
    fi

    for line in "${files[@]}"; do
        IFS=$'\t' read -r size file <<<"$line" || true
        ((++counter))
        filename="${file##*/}"

        if   [[ "$size" -lt 1024 ]];       then readable_size="${size}B"
        elif [[ "$size" -lt 1048576 ]];    then readable_size="$(awk "BEGIN {printf \"%.3f\", $size/1024}")K"
        elif [[ "$size" -lt 1073741824 ]]; then readable_size="$(awk "BEGIN {printf \"%.3f\", $size/1048576}")M"
        else                                  readable_size="$(awk "BEGIN {printf \"%.3f\", $size/1073741824}")G"
        fi

        echo "========================================================================================="
        echo "Processing [$counter/$total][$part_label]: $file (actual size: $readable_size)"
        echo "-----------------------------------------------------------------------------------------"

        local out_path="$SCRIPT_DIR/$POST_DIR/${POST_PREFIX}-${part_label}-${filename}"

        ABC_CMD="&r -s $file; &ps; &fx${fx_arg}; &ps"
        if [[ "$RUN_CEC" -eq 1 ]]; then
            ABC_CMD="$ABC_CMD; &put; &w $out_path"
        fi

        if [[ "$RUN_CEC" -eq 0 ]]; then
            time "$ABC_BIN" -c "$ABC_CMD"
        else
            "$ABC_BIN" -c "$ABC_CMD"
        fi
        echo "========================================================================================="

        if [[ "$RUN_CEC" -eq 1 ]]; then
            echo "========================================================================================="
            echo "Verifying [$counter/$total][$part_label]: $file (actual size: $readable_size)"
            echo "-----------------------------------------------------------------------------------------"
            "$ABC_BIN" -c "cec $out_path $file"
            echo "========================================================================================="
        fi

        echo
        echo
    done
}

# ---------- main ----------
for entry in "${SWEEP_LIST[@]}"; do
    p_opt="${entry%%:*}"
    part_label="${entry#*:}"

    if [[ "$PART_MODE" == "sweep" ]]; then
        echo "####################################################"
        if [[ -z "$p_opt" ]]; then
            echo "# Sweep pass : $part_label  (&fx default)"
        else
            echo "# Sweep pass : $part_label  (&fx -p${p_opt})"
        fi
        echo "####################################################"
        echo
    fi

    run_pass "$p_opt" "$part_label"
done
