#!/bin/bash

# 1. Navegue para a pasta correta (ajuste o caminho se necessário)
cd /home/vfrocha/VibNet-1D/data/processed

echo "Iniciando a padronização das pastas de classes Normais..."

# 2. Renomeia Class_0 para Class_Normal em todas as subpastas do CWRU (12k e 48k)
for d in CWRU*/Load_*; do [ -d "$d/Class_0" ] && mv "$d/Class_0" "$d/Class_Normal"; done
echo " -> CWRU concluído."

# 3. Renomeia Class_40 para Class_Normal em todas as subpastas da HUST (Bearing)
for d in HUST/Load_*; do [ -d "$d/Class_40" ] && mv "$d/Class_40" "$d/Class_Normal"; done
echo " -> HUST Bearing concluído."

# 4. Renomeia Class_26 para Class_Normal em todas as subpastas da PU
for d in PU/C*; do [ -d "$d/Class_26" ] && mv "$d/Class_26" "$d/Class_Normal"; done
echo " -> PU concluído."

# --- NOVO: HUST GEARBOX ---
# 5. Renomeia Class_Healthy para Class_Normal nas 30 condições da HUST Gearbox
for d in HUST_Gearbox/Cond_*; do [ -d "$d/Class_Healthy" ] && mv "$d/Class_Healthy" "$d/Class_Normal"; done
echo " -> HUST Gearbox concluído."
# --------------------------

echo "Renomeação concluída com sucesso!"
