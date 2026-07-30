#!/bin/bash
# 从 /tmp/GC 复制文档到对应目录
# 用法:
#   ./copy_gc_docs.sh              # 复制全部格式
#   ./copy_gc_docs.sh pdf          # 仅 PDF
#   ./copy_gc_docs.sh xls ppt      # 仅 Excel + PPT
#   ./copy_gc_docs.sh word pdf     # 仅 Word + PDF
# 支持的格式别名: xls|excel, ppt|powerpoint, word|doc, pdf

set -euo pipefail

SRC_ROOT="/tmp/GC"
XLS_TARGET="/data/xls"
PPT_TARGET="/data/ppt"
WORD_TARGET="/data/word"
PDF_TARGET="/data/raw/pdf"

DIR_LIST=(
    "00-优秀教材评审材料"
    "光刻系-Litho"
    "干刻系-Etch"
    "研磨系-CMP"
    "管理系-安全环境职业健康专业-EHS"
    "管理系-行政管理专业-临港综合"
    "薄膜系-TF"
    "00-课程共享-临港site级"
    "厂务系-FAC"
    "扩散系-Diff"
    "管理系-企业管理专业-HR"
    "管理系-工程支持专业-IE"
    "管理系-质量管理专业-QR"
    "00-课程共享-公司级"
    "工艺集成系-PIE"
    "湿法系-WET"
    "管理系-信息技术专业-IT"
    "管理系-生产管理专业-MFG"
    "管理系-采购管理专业-制造采购"
)

# 解析要复制的格式；无参数则全部复制
COPY_XLS=0
COPY_PPT=0
COPY_WORD=0
COPY_PDF=0

normalize_format() {
    case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
        xls|xlsx|excel) echo "xls" ;;
        ppt|pptx|powerpoint) echo "ppt" ;;
        word|doc|docx) echo "word" ;;
        pdf) echo "pdf" ;;
        *) echo "" ;;
    esac
}

if [[ $# -eq 0 ]]; then
    COPY_XLS=1
    COPY_PPT=1
    COPY_WORD=1
    COPY_PDF=1
else
    for arg in "$@"; do
        fmt="$(normalize_format "$arg")"
        if [[ -z "${fmt}" ]]; then
            echo "错误：不支持的格式 '${arg}'"
            echo "支持: xls|excel, ppt|powerpoint, word|doc, pdf"
            exit 1
        fi
        case "${fmt}" in
            xls) COPY_XLS=1 ;;
            ppt) COPY_PPT=1 ;;
            word) COPY_WORD=1 ;;
            pdf) COPY_PDF=1 ;;
        esac
    done
fi

mkdir_targets=()
[[ "${COPY_XLS}" -eq 1 ]] && mkdir_targets+=("${XLS_TARGET}")
[[ "${COPY_PPT}" -eq 1 ]] && mkdir_targets+=("${PPT_TARGET}")
[[ "${COPY_WORD}" -eq 1 ]] && mkdir_targets+=("${WORD_TARGET}")
[[ "${COPY_PDF}" -eq 1 ]] && mkdir_targets+=("${PDF_TARGET}")
mkdir -p "${mkdir_targets[@]}"

TIMESTAMP=$(date +%Y%m%d%H%M%S)

copy_by_ext() {
    local full_src="$1"
    local target_dir="$2"
    shift 2
    local -a patterns=("$@")

    local find_expr=()
    local i=0
    for pat in "${patterns[@]}"; do
        [[ $i -gt 0 ]] && find_expr+=(-o)
        find_expr+=(-iname "${pat}")
        i=$((i + 1))
    done

    find "${full_src}" -type f \( "${find_expr[@]}" \) \
        -exec sh -c 'cp "$1" "'"${target_dir}"'/'"${TIMESTAMP}"'_$(basename "$1")"' _ {} \;
}

echo "========== 开始复制文档 =========="
echo "时间戳前缀：${TIMESTAMP}"
selected=()
[[ "${COPY_XLS}" -eq 1 ]] && selected+=("Excel→${XLS_TARGET}")
[[ "${COPY_PPT}" -eq 1 ]] && selected+=("PPT→${PPT_TARGET}")
[[ "${COPY_WORD}" -eq 1 ]] && selected+=("Word→${WORD_TARGET}")
[[ "${COPY_PDF}" -eq 1 ]] && selected+=("PDF→${PDF_TARGET}")
echo "本次格式：${selected[*]}"
echo

for sub_dir in "${DIR_LIST[@]}"; do
    full_src="${SRC_ROOT}/${sub_dir}"
    if [[ ! -d "${full_src}" ]]; then
        echo "警告：${full_src} 不存在，跳过"
        continue
    fi
    echo "正在处理：${full_src}"

    if [[ "${COPY_XLS}" -eq 1 ]]; then
        copy_by_ext "${full_src}" "${XLS_TARGET}" "*.xls" "*.xlsx"
    fi
    if [[ "${COPY_PPT}" -eq 1 ]]; then
        copy_by_ext "${full_src}" "${PPT_TARGET}" "*.ppt" "*.pptx"
    fi
    if [[ "${COPY_WORD}" -eq 1 ]]; then
        copy_by_ext "${full_src}" "${WORD_TARGET}" "*.doc" "*.docx"
    fi
    if [[ "${COPY_PDF}" -eq 1 ]]; then
        copy_by_ext "${full_src}" "${PDF_TARGET}" "*.pdf"
    fi
done

echo
echo "========== 复制完成 =========="
[[ "${COPY_XLS}" -eq 1 ]] && echo "Excel：${XLS_TARGET}"
[[ "${COPY_PPT}" -eq 1 ]] && echo "PPT：${PPT_TARGET}"
[[ "${COPY_WORD}" -eq 1 ]] && echo "Word：${WORD_TARGET}"
[[ "${COPY_PDF}" -eq 1 ]] && echo "PDF：${PDF_TARGET}"
