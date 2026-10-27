# Repository Guidelines

## Contexto e estrutura

O VibNet-1D é um projeto de pesquisa para diagnóstico de falhas em máquinas rotativas com sinais de vibração, aprendizado profundo e transferência entre datasets e condições operacionais. `src/data/` contém preparação e validação de datasets; `src/features/`, extração de atributos; `src/models/`, arquiteturas e treinamento; `src/preprocessing/`, normalização; `src/experiments/`, experimentos (os anteriores ficam em `old_experiments/`). `scripts/` reúne utilitários e pontos de entrada, `notebooks/` exploração, `docs/` documentação e `results/` saídas geradas. Prefira adicionar módulos e scripts novos a sobrescrever experimentos existentes.

## Desenvolvimento e estilo

Instale dependências somente quando necessário e após avaliar o impacto: `python -m pip install -r requirements.txt`. Execute scripts a partir da raiz do repositório, por exemplo `python scripts/validate_dataset_tl.py`. Siga o estilo existente: indentação de quatro espaços, nomes `snake_case` e classes `PascalCase`; scripts experimentais costumam usar `run_*` ou `exp_*`. Não há suíte de testes, build unificado, formatter ou linter configurados. Antes de alterar código, inspecione a implementação real, configurações e testes e identifique os pontos de integração; trate-os como fonte de verdade, sem presumir que exemplos de conversas estejam completos. Preserve funcionalidades, parâmetros e saídas existentes. Prefira mudanças pequenas, reproduzíveis e revisáveis.

## Método de trabalho

Para tarefas complexas, apresente antes um plano breve com arquivos a modificar e riscos metodológicos. Execute verificações pertinentes de sintaxe e testes; informe o que foi e não foi executado. Não inicie experimentos longos, instale dependências, altere ambientes ou publique remotamente sem avaliar o impacto e obter confirmação quando necessária. Não execute comandos Git destrutivos, descarte alterações do usuário nem faça push sem autorização explícita.

## Protocolo científico

Evite vazamento entre treino, validação e teste. Em avaliações LOCO, mantenha a condição de teste isolada de ajuste, seleção de hiperparâmetros e balanceamento. Preserve a separação entre domínios e os vocabulários de classes próprios de cada dataset. Use Macro-F1 como métrica primária quando previsto pelo protocolo solicitado; reporte métricas secundárias e variabilidade quando disponíveis. Não compare protocolos distintos sem explicitar suas diferenças. Distinga resultados observados, hipóteses, limitações e recomendações. Para cada experimento, registre parâmetros, seeds, métricas, caminhos de saída e limitações. Nunca descreva testes sintéticos ou verificações de sintaxe como experimentos com dados reais.

## Dados, commits e comunicação

Trate datasets como entradas e resultados como artefatos; não inclua dados brutos grandes ou privados, credenciais ou caminhos específicos da máquina. Não há histórico de commits disponível para inferir convenção; prefira assuntos imperativos concisos, como `Add dataset validation`. PRs devem explicar objetivo, módulos afetados e verificações ou experimentos executados, além de incluir resumos de resultados relevantes e issues relacionadas quando aplicável. Responda em português brasileiro, salvo solicitação de outro idioma. Ao concluir, resuma alterações, arquivos, verificações, resultados e próximos passos.
