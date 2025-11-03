# Versionamento do MINAS

O MINAS usa **Versionamento Semântico** (Semantic Versioning): `MAJOR.MINOR.PATCH`

## Tipos de Mudança

### MAJOR (1.x.x → 2.x.x)
**Mudanças significativas / Breaking Changes**
- Mudanças na API que quebram compatibilidade
- Remoção de funcionalidades
- Reestruturação completa de módulos
- Mudanças que exigem alterações no código dos usuários

**Exemplos:**
- Renomear funções públicas
- Mudar assinatura de funções (parâmetros obrigatórios)
- Remover funções ou módulos

### MINOR (x.1.x → x.2.x)
**Novas funcionalidades (backward compatible)**
- Adicionar novas funcionalidades
- Adicionar novos parâmetros opcionais
- Melhorias de performance significativas
- Novos módulos ou classes

**Exemplos:**
- Nova função de visualização
- Novo algoritmo de predição
- Novos parâmetros opcionais em funções existentes

### PATCH (x.x.1 → x.x.2)
**Correções de bugs / Mudanças pequenas**
- Correção de bugs
- Ajustes de documentação
- Pequenos ajustes visuais
- Correções de typos no código

**Exemplos:**
- Corrigir erro nos gráficos (ticks no eixo x)
- Ajustar tamanho de fonte
- Corrigir cálculo de métricas

## Como Usar

### 1. Atualizar versão automaticamente:

```bash
# Para mudanças MAJOR (breaking changes)
python bump_version.py major

# Para novas funcionalidades (MINOR)
python bump_version.py minor

# Para correções de bugs (PATCH)
python bump_version.py patch
```

### 2. Commitar as mudanças:

```bash
git add VERSION pyproject.toml
git commit -m "Bump version to X.Y.Z"
git tag vX.Y.Z
git push && git push --tags
```

## Histórico de Versões

### v1.0.0 (2025-09-03)
- Versão inicial do MINAS
- Funcionalidades de predição com RF e XGBoost
- Gráficos de regressão e correção bolométrica
- Suporte para APOGEE, LAMOST, GALAH

### Exemplos de Próximas Versões

- **v1.0.1**: Corrigir ticks nos gráficos de resíduos
- **v1.1.0**: Adicionar suporte para novos surveys
- **v2.0.0**: Reestruturar API completamente
