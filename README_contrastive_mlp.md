# VibNet-1D — experimentos finais com MLP e contrastive loss

## Arquivos

- `scripts/tl/exp_tl_contrastive_mlp.py`: runner LOCO que reutiliza o pipeline de extração de features do experimento de balanceamento.
- `src/models/build_mlp_contrastive.py`: MLP convencional com treinamento contrastivo conjunto ou pré-treinamento contrastivo.

O módulo original `src/models/build_tabnet_resnet.py` e o módulo de balanceamento anterior não são modificados.

## Protocolo

- Fontes: `CWRU_12k` e `MFPT`.
- Target: `CWRU_48k`.
- Variante padrão: `detrended_rms`.
- Avaliação: LOCO por condição, com a mesma regra de validade do experimento original.
- Métrica primária: Macro-F1; métricas adicionais: ACC, Balanced Accuracy e ROC-AUC.
- Seed padrão: 42.
- O scaler é ajustado separadamente em cada domínio usando apenas dados de treino; o scaler do target nunca vê a condição LOCO de teste.
- Rótulos e pares contrastivos são tratados por domínio. Não se comparam diretamente rótulos entre fontes diferentes, pois os vocabulários de classes podem não coincidir.
- O conjunto de teste não participa do treinamento contrastivo, da seleção de hiperparâmetros nem da escolha do checkpoint.

## Modos de contrastive loss

- `joint`: treinamento supervisionado multi-domínio com `CrossEntropy + lambda * SupervisedContrastiveLoss` em cada minibatch. A perda contrastiva usa rótulos daquele domínio apenas.
- `pretrain`: pré-treinamento contrastivo supervisionado apenas nas fontes, seguido por treinamento multi-domínio com cross-entropy e fine-tuning supervisionado no target. A cabeça de projeção é descartada após o pré-treinamento; o encoder é mantido.

## Arquiteturas

- `--encoder-profile baseline`: encoder `256 -> 128 -> latent(64)`.
- `--encoder-profile expanded`: encoder `512 -> 256 -> 128 -> latent(64)`.
- `--head-shape asymmetric`: cabeça pós-latente atual `128 -> 64 -> classes`.
- `--head-shape symmetric`: cabeça pós-latente em ordem inversa das larguras do encoder; por exemplo, no encoder expandido `128 -> 256 -> 512 -> classes`.
- `--head-shape linear`: classificação linear diretamente do latente; serve como ablação opcional.

O termo “head” é usado no código para evitar confusão: trata-se de uma MLP classificadora pós-latente, não de um decoder de reconstrução de autoencoder.

## Instalação

Copie os arquivos do pacote para a raiz do repositório, preservando os caminhos:

```text
scripts/tl/exp_tl_contrastive_mlp.py
src/models/build_mlp_contrastive.py
