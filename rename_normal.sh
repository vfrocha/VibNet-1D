# 1. Navegue para a pasta correta (ajuste o caminho se necessário)
cd /home/vfrocha/VibNet-1D/data/processed

# 2. Renomeia Class_0 para Class_Normal em todas as subpastas do CWRU (12k e 48k)
for d in CWRU*/Load_*; do [ -d "$d/Class_0" ] && mv "$d/Class_0" "$d/Class_Normal"; done

# 3. Renomeia Class_40 para Class_Normal em todas as subpastas da HUST
for d in HUST/Load_*; do [ -d "$d/Class_40" ] && mv "$d/Class_40" "$d/Class_Normal"; done

# 4. Renomeia Class_26 para Class_Normal em todas as subpastas da PU
for d in PU/C*; do [ -d "$d/Class_26" ] && mv "$d/Class_26" "$d/Class_Normal"; done

print "Renomeação concluída!"
