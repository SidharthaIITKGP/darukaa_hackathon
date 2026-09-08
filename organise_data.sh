#!/usr/bin/env bash
# Reorganise the flat Data/ folder into the pipeline structure.
#
#   cd ~/Coding/Darukaa\ Task
#   bash organise_data.sh
#
# Pass a different source dir as $1 if yours isn't called Data.
set -uo pipefail

SRC="${1:-Data}"
[ -d "$SRC" ] || { echo "No such directory: $SRC"; exit 1; }

mkdir -p data/corpus/assessments data/corpus/meta_analyses \
         data/structured/takola_2023 data/rasters/gsocseq \
         data/cache data/derived

# take <glob> <destination-path>   - globs so odd filenames still match
take () {
  local pattern="$1" dest="$2" found=0
  for f in $SRC/$pattern; do
    if [ -e "$f" ]; then
      mv "$f" "$dest"
      echo "  OK      $(basename "$dest")"
      found=1
      break
    fi
  done
  [ "$found" -eq 1 ] || echo "  MISSING  $pattern"
}

echo "== Assessments =="
take 'SRCCL_Chapter_6*.pdf'        data/corpus/assessments/IPCC_2019_SRCCL_Ch6_response_options.pdf
take 'IPCC_AR6_WGIII_Chapter07*'   data/corpus/assessments/IPCC_2022_AR6_WG3_Ch7_AFOLU.pdf
take 'ipbes-6-15*'                 data/corpus/assessments/IPBES_2018_land_degradation_spm.pdf
take 'cb1928en*'                   data/corpus/assessments/FAO_2020_soil_biodiversity.pdf
take 'cb9002en*'                   data/corpus/assessments/FAO_2022_GSOCseq_technical_report.pdf
take 'i6874en*'                    data/corpus/assessments/FAO_2017_VGSSM.pdf

echo "== Meta-analyses =="
take 'Agronomy*'                   data/corpus/meta_analyses/Joshi_2023_covercrops_SOC_corn.pdf
take 'Ecological*'                 data/corpus/meta_analyses/McClelland_2020_covercrops_SOC_temperate.pdf
take '12862_2021*'                 data/corpus/meta_analyses/Mupepele_2021_agroforestry_biodiv_europe.pdf
take '12862_2022*'                 data/corpus/meta_analyses/Boinot_2022_agroforestry_critique.pdf
take '41467_2019*'                 data/corpus/meta_analyses/Woodcock_2019_pollinator_diversity_yield.pdf
take 'GCB-31*'                     data/corpus/meta_analyses/GCB_2025_agroforestry_multifunctionality.pdf

echo "== Structured (Takola) =="
ZIP=""
for f in $SRC/7s9r4*.zip; do [ -e "$f" ] && ZIP="$f" && break; done
if [ -n "$ZIP" ]; then
  unzip -o -q "$ZIP" -d data/structured/takola_2023/
  # drop the systematic-review audit trail; keep only what we use
  rm -f data/structured/takola_2023/query_search_title-abstract_screening_*.xls
  rm -f data/structured/takola_2023/primarystudies_de-dupl_*.csv
  rm -f data/structured/takola_2023/overlap_matrix_primary_percentage.csv
  rm -f data/structured/takola_2023/script_*.txt
  mv "$ZIP" data/structured/takola_2023/_original_archive.zip
  ls data/structured/takola_2023/ | sed 's/^/  OK      /'
else
  echo "  MISSING  7s9r4*.zip"
fi

echo "== Rasters =="
n=0
for f in $SRC/*.tif $SRC/GSOCseq*; do
  [ -e "$f" ] && mv "$f" data/rasters/gsocseq/ && echo "  OK      $(basename "$f")" && n=$((n+1))
done
[ "$n" -gt 0 ] || echo "  none (GSOCseq rasters not downloaded - optional)"

if [ ! -f .gitignore ]; then
cat > .gitignore <<'EOF'
.venv/
venv/
__pycache__/
*.pyc
.env

# Generated artifacts - rebuildable from data/corpus
data/derived/

# Large binaries - document how to fetch in README instead
data/rasters/*.tif
EOF
echo "== Wrote .gitignore =="
fi

echo
LEFT=$(ls -A "$SRC" 2>/dev/null | wc -l)
if [ "$LEFT" -eq 0 ]; then
  rmdir "$SRC" && echo "$SRC/ was empty and has been removed."
else
  echo "Still in $SRC/ (check before deleting):"
  ls -A "$SRC" | sed 's/^/  /'
fi

echo
echo "== Final structure =="
find data -type f | sort | sed 's/^/  /'
